"""Explicit HU preflight and frozen feature-v4 fit calibration, never auto-pilot."""
from __future__ import annotations

from collections import Counter
import hashlib
import io
import json
import math
from pathlib import Path
import random
import resource
import sys
import time
import tempfile

import torch

from ml.memory import sample_bytes
from ml.model import AdvantageNetwork, AveragePolicyNetwork
from ml.train import fit
from scripts.fit_budget_audit import EPOCHS, opening_predictions
from scripts.parallel_cfr import traversal_workers
from training.checkpoint import atomic_bytes
from training.config import load_config
from training.metrics import opening_hand_class
from training.runner import TrainingRunner


def write_json(path: Path, value: dict) -> None:
    atomic_bytes(path, (json.dumps(value, indent=2, allow_nan=False) + "\n").encode())


def replay_summary(memory) -> dict:
    classes = Counter(label for sample in memory.samples if (label := opening_hand_class(sample.state)) is not None)
    sizes = [sample_bytes(sample) for sample in memory.samples]
    return {"retained": len(memory.samples), "seen": memory.seen,
            "opening_records": sum(classes.values()), "opening_hand_classes": len(classes),
            "opening_class_counts": dict(sorted(classes.items())), "accounted_bytes": memory.used_bytes,
            "mean_sample_bytes": sum(sizes) / len(sizes), "max_sample_bytes": max(sizes)}


def choose_epoch(rows: list[dict], tolerance: float = .05) -> dict:
    """Exploratory rule; a boundary minimum explicitly does not claim convergence."""
    if not rows or not 0 <= tolerance < 1 or any(not math.isfinite(row["heldout_loss"]) or row["heldout_loss"] < 0 for row in rows):
        raise ValueError("Finite nonnegative held-out measurements and tolerance in [0,1) required")
    minimum = min(rows, key=lambda row: row["heldout_loss"])
    selected = min((row for row in rows if row["heldout_loss"] <= minimum["heldout_loss"] * (1 + tolerance)), key=lambda row: row["epoch"])
    return {"epochs": selected["epoch"], "minimum_epoch": minimum["epoch"],
            "minimum_heldout_loss": minimum["heldout_loss"], "relative_tolerance": tolerance,
            "minimum_at_last_measurement": minimum["epoch"] == max(row["epoch"] for row in rows),
            "convergence_established": False, "scope": "exploratory frozen-replay selection; validation is record-level and correlated within traversals"}


def run_preflight(output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"Preflight output already exists: {output}")
    config = load_config("configs/train_hu.json")
    config.update(output_dir=str(output), outer_iterations=2, workers=4, trainer_threads=1,
                  batch_size=64, advantage_epochs=2, average_epochs=2, checkpoint_every=1, evaluation_every=2)
    config["hu"]["traversals_per_player"] = 32
    config["hu"]["root_sampling"]["hole_card_sampling"] = "stratified"
    output.mkdir(parents=True)
    write_json(output / "config.json", config)
    runner = TrainingRunner(config)
    report = {"config": config, "iterations": [], "scope": "two-iteration throughput and coverage preflight; no poker-quality acceptance"}
    write_json(output / "report.json", report)
    with traversal_workers(4) as executor:
        for _ in range(2):
            started = time.perf_counter()
            print(f"HU preflight iteration {runner.iteration + 1}/2: 64 traversals, 4 workers, max_nodes=20000", flush=True)
            try:
                row = runner.run_iteration(executor=executor)
            except Exception as error:
                report["failure"] = {"type": type(error).__name__, "message": str(error), "seconds": time.perf_counter() - started}
                write_json(output / "report.json", report)
                write_json(Path("docs/hu-coverage-preflight.json"), report)
                raise
            solver = runner.solvers["hu"]
            metric = row["tracks"]["hu"]
            summary = {"iteration": runner.iteration, "metrics": metric,
                       "replay": {"advantage": replay_summary(solver.advantage_memory), "strategy": replay_summary(solver.strategy_memory)}, "probes": {}}
            for kind, model in (("advantage", solver.advantage_model), ("strategy", solver.average_model)):
                predictions = opening_predictions(model, 2, 256)
                filename = f"{kind}-iteration-{runner.iteration:02d}.json"
                write_json(output / filename, predictions)
                summary["probes"][kind] = {"artifact": filename, "holdings": predictions["holdings"],
                                           "action_spread": predictions["action_spread"], "max_card_order_gap": predictions["max_card_order_gap"]}
            coverage = metric["sample_coverage"]
            summary["generated_openings"] = {kind: sum(coverage.get(f"{kind}_opening_hand_class", {}).values()) for kind in ("advantage", "strategy")}
            summary["root_exact_combos_cumulative"] = len({combo for key, values in metric["hole_card_coverage"].items() if key.endswith(":SB") for combo in values})
            summary["root_exact_combos_scope"] = "traverser-targeted HU SB opening roots, union across seats and iterations"
            summary["seconds_including_probes"] = time.perf_counter() - started
            summary["parent_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
            summary["generation_traversals_per_second"] = metric["traversals"] / metric["generation_seconds"]
            report["iterations"].append(summary)
            write_json(output / "report.json", report)
            write_json(Path("docs/hu-coverage-preflight.json"), report)
            print(f"HU iteration {runner.iteration}: {summary['seconds_including_probes']:.1f}s; advantage opening generated={summary['generated_openings']['advantage']} retained={summary['replay']['advantage']['opening_records']}; root combos={summary['root_exact_combos_cumulative']}", flush=True)
    report["checkpoint_benchmark"] = benchmark_checkpoint(output / "checkpoints" / "iteration_000002.pt")
    write_json(output / "report.json", report)
    write_json(Path("docs/hu-coverage-preflight.json"), report)
    return report


def run_calibration(checkpoint: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"Calibration output already exists: {output}")
    source_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    runner = TrainingRunner.load_checkpoint(checkpoint)
    solver = runner.solvers["hu"]
    output.mkdir(parents=True)
    report = {"feature_version": 4, "source_checkpoint": str(checkpoint), "source_sha256": source_hash,
              "epochs": EPOCHS, "models": {}, "batch_benchmark": {}}
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for kind, constructor, memory in (("advantage", AdvantageNetwork, solver.advantage_memory),
                                          ("strategy", AveragePolicyNetwork, solver.strategy_memory)):
            seed = solver.seed + (10 if kind == "strategy" else 0)
            initialization = seed + (solver.version + 1) * 101
            samples = memory.samples
            order = list(range(len(samples)))
            random.Random(seed).shuffle(order)
            cut = max(1, min(len(order)-1, round(len(order)*.2)))
            timings = {}
            for batch_size in (64, 256):
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(initialization)
                    model = constructor(feature_version=4)
                    started = time.perf_counter()
                    metrics = fit(model, samples, 1, batch_size, seed, learning_rate=runner.config["learning_rate"])
                    elapsed = time.perf_counter() - started
                    timings[str(batch_size)] = {"seconds": elapsed, "training_samples_per_second": (len(samples)-cut)/elapsed,
                                                "metrics": metrics, "scope": "one epoch including validation; different batch sizes imply different SGD update counts"}
                    print(f"Batch benchmark {kind}: batch={batch_size} seconds={elapsed:.2f}", flush=True)
            report["batch_benchmark"][kind] = timings
            batch_size = min((64, 256), key=lambda size: timings[str(size)]["seconds"])
            result = {"seed": seed, "initialization_seed": initialization, "batch_size": batch_size,
                      "heldout_indices": order[:cut], "training_indices": order[cut:], "measurements": [], "replay": replay_summary(memory)}
            report["models"][kind] = result
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(initialization)
                model = constructor(feature_version=4)
                started = time.perf_counter()
                def record(epoch: int, metrics: dict[str, float]) -> None:
                    predictions = opening_predictions(model, 2, batch_size)
                    stem = f"hu-{kind}-epoch-{epoch:02d}"
                    write_json(output / f"{stem}.json", predictions)
                    buffer = io.BytesIO()
                    torch.save({"feature_version": 4, "kind": kind, "epoch": epoch, "source_sha256": source_hash,
                                "weights": model.state_dict()}, buffer)
                    atomic_bytes(output / f"{stem}.pt", buffer.getvalue())
                    result["measurements"].append({"epoch": epoch, "elapsed_seconds": time.perf_counter()-started, **metrics,
                                                    "holdings": predictions["holdings"], "action_spread": predictions["action_spread"],
                                                    "max_card_order_gap": predictions["max_card_order_gap"], "artifact": stem,
                                                    "opening_probe_gates": opening_probe_gates(predictions)})
                    write_json(output / "report.json", report)
                    print(f"Calibration {kind}: epoch={epoch} train={metrics['train_loss']:.6g} heldout={metrics['heldout_loss']:.6g} elapsed={time.perf_counter()-started:.1f}s", flush=True)
                fit(model, samples, 50, batch_size, seed, learning_rate=runner.config["learning_rate"], measurement_epochs=EPOCHS, on_measurement=record)
            result["selection"] = choose_epoch(result["measurements"])
            write_json(output / "report.json", report)
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != source_hash:
            raise RuntimeError(f"Source checkpoint changed during calibration: {checkpoint}")
        write_json(Path("docs/hu-coverage-calibration.json"), report)
        return report
    finally:
        torch.set_num_threads(previous_threads)


def benchmark_checkpoint(checkpoint: Path) -> dict:
    """Measure serialization and atomic fsync on a temporary copy; preserve source."""
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    runner = TrainingRunner.load_checkpoint(checkpoint)
    with tempfile.TemporaryDirectory(prefix="hu-checkpoint-preflight-") as temporary:
        destination = Path(temporary) / "checkpoint.pt"
        started = time.perf_counter()
        runner.save_checkpoint(destination)
        elapsed = time.perf_counter()-started
        size = destination.stat().st_size
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != digest:
        raise RuntimeError(f"Checkpoint benchmark changed source: {checkpoint}")
    return {"seconds": elapsed, "bytes": size, "source_sha256": digest,
            "scope": "one complete staged checkpoint serialization and atomic write/fsync on local temporary storage"}


def medium_estimate(preflight: dict, calibration: dict, strategy_capacity: int | None = None) -> dict:
    """Extrapolate observed ratios; explicitly report uncertainty and budget limits."""
    rows = preflight["iterations"]
    totals = {key: sum(row["metrics"][key] for row in rows) for key in ("traversals", "advantage_samples", "strategy_samples")}
    generated_openings = sum(row["generated_openings"]["advantage"] for row in rows)
    if generated_openings <= 0:
        raise ValueError("Cannot estimate medium coverage with no opening advantage records")
    opening_fraction = generated_openings / totals["advantage_samples"]
    openings_per_traversal = generated_openings / totals["traversals"]
    iterations = 6
    per_player = math.ceil(3000 / (iterations * 2 * openings_per_traversal))
    traversal_count = iterations * 2 * per_player
    capacity = math.ceil(3000 / opening_fraction * 1.25)
    generation_seconds_per_traversal = max(row["metrics"]["generation_seconds"] / row["metrics"]["traversals"] for row in rows)
    generation = traversal_count * generation_seconds_per_traversal
    selected = {kind: calibration["models"][kind]["selection"]["epochs"] for kind in ("advantage", "strategy")}
    expected_strategy_count = math.ceil(traversal_count * totals["strategy_samples"] / totals["traversals"])
    if strategy_capacity is not None and (type(strategy_capacity) is not int or strategy_capacity < 2):
        raise ValueError(f"Explicit strategy capacity must be an integer >=2: {strategy_capacity}")
    strategy_capacity = min(capacity if strategy_capacity is None else strategy_capacity, expected_strategy_count)
    fit_seconds = 0.0
    for kind, cap, sample_total in (("advantage", capacity, traversal_count*totals["advantage_samples"]/totals["traversals"]),
                                   ("strategy", strategy_capacity, expected_strategy_count)):
        result = calibration["models"][kind]
        measurement = next(row for row in result["measurements"] if row["epoch"] == selected[kind])
        per_epoch_sample = measurement["elapsed_seconds"] / (selected[kind] * result["replay"]["retained"])
        fit_seconds += sum(min(cap, sample_total * step / iterations) * selected[kind] * per_epoch_sample for step in range(1,iterations+1))
    adv_bytes = capacity * rows[-1]["replay"]["advantage"]["mean_sample_bytes"]
    strategy_bytes = strategy_capacity * rows[-1]["replay"]["strategy"]["mean_sample_bytes"]
    generated_batch_bytes = traversal_count / iterations * sum(totals[kind+"_samples"]/totals["traversals"] * rows[-1]["replay"][kind]["mean_sample_bytes"] for kind in ("advantage", "strategy"))
    root_seconds = traversal_count * max(row["metrics"]["root_seconds"]/row["metrics"]["traversals"] for row in rows)
    checkpoint_benchmark = preflight.get("checkpoint_benchmark")
    checkpoint_seconds = 0.0
    checkpoint_bytes = 0.0
    if checkpoint_benchmark is not None:
        replay_bytes = sum(rows[-1]["replay"][kind]["accounted_bytes"] for kind in ("advantage", "strategy"))
        bytes_ratio = checkpoint_benchmark["bytes"] / replay_bytes
        for step in range(1, iterations+1):
            adv_retained = min(capacity, traversal_count*totals["advantage_samples"]/totals["traversals"] * step/iterations)
            strategy_retained = min(strategy_capacity, expected_strategy_count*step/iterations)
            payload = bytes_ratio*(adv_retained*rows[-1]["replay"]["advantage"]["mean_sample_bytes"] + strategy_retained*rows[-1]["replay"]["strategy"]["mean_sample_bytes"])
            checkpoint_bytes += payload
            checkpoint_seconds += payload/checkpoint_benchmark["bytes"]*checkpoint_benchmark["seconds"]
    total = generation + fit_seconds + root_seconds + checkpoint_seconds
    pessimistic = total * 2
    estimated_peak = 3*(adv_bytes + strategy_bytes) + 2*generated_batch_bytes + 4*max(row["metrics"]["worker_peak_rss_bytes"] for row in rows) + max(row["parent_peak_rss_bytes"] for row in rows)
    return {"targets": {"retained_advantage_openings": 3000, "generated_exact_root_combos": 1000, "retained_hand_classes": 150, "iterations": iterations},
            "assumptions": ["Checkpoint time and disk bytes scale with accounted replay from one measured save; evaluation/probe overhead is outside this component model.",
                            "Observed opening/sample ratios persist; six iterations change strategy and may change branching substantially.",
                            "25% replay-capacity margin targets expected retention only, not a coverage guarantee.",
                            "Twice measured extrapolation is a scenario, not a proven upper bound; max_nodes remains a hard per-traversal guard.",
                            "Exact-combo coverage counts traverser-targeted SB roots, not canonical replay card IDs.",
                            "3000 opening draws make all 169 hand classes likely under unbiased dealing; actual held classes require measurement."],
            "traversals_per_player_per_iteration": per_player, "total_traversals": traversal_count,
            "advantage_capacity": capacity, "strategy_capacity": strategy_capacity, "selected_epochs": selected,
            "expected_unique_SB_combos_iid_reference": 1326*(1-(1-1/1326)**3000),
            "expected_strategy_opening_records": strategy_capacity * sum(row["generated_openings"]["strategy"] for row in rows)/totals["strategy_samples"],
            "observed_opening_fraction": opening_fraction, "observed_openings_per_traversal": openings_per_traversal,
            "generation_seconds": generation, "fit_seconds": fit_seconds, "root_seconds": root_seconds, "checkpoint_seconds": checkpoint_seconds,
            "estimated_six_checkpoint_disk_bytes": checkpoint_bytes,
            "estimated_seconds": total, "pessimistic_seconds": pessimistic, "runtime_budget_seconds":1800,
            "estimated_accounted_replay_bytes": adv_bytes+strategy_bytes, "estimated_generated_batch_bytes": generated_batch_bytes,
            "estimated_peak_memory_bytes": estimated_peak, "memory_budget_bytes":24*1024**3,
            "meets_estimated_time_budget": total <=1800, "meets_pessimistic_time_budget":pessimistic <=1800,
            "meets_memory_budget":estimated_peak<=24*1024**3, "medium_pilot_launched":False}


def prepare_medium_config(preflight: dict, estimate: dict, output_dir: str) -> dict:
    """Prepare explicit measured byte headroom; execution is a separate decision."""
    from training.config import validate_config
    config = json.loads(json.dumps(preflight["config"]))
    config.update(output_dir=output_dir, outer_iterations=6, batch_size=256,
                  advantage_epochs=estimate["selected_epochs"]["advantage"], average_epochs=estimate["selected_epochs"]["strategy"],
                  checkpoint_every=1, evaluation_every=1, strategy_capacity=estimate["strategy_capacity"])
    config["hu"]["traversals_per_player"] = estimate["traversals_per_player_per_iteration"]
    config["hu"]["advantage_capacity"] = estimate["advantage_capacity"]
    replay = preflight["iterations"][-1]["replay"]
    config["hu"]["advantage_byte_budget"] = math.ceil(1.25*estimate["advantage_capacity"]*replay["advantage"]["max_sample_bytes"])
    config["memory_byte_budget"] = max(config["memory_byte_budget"], math.ceil(1.25*estimate["strategy_capacity"]*replay["strategy"]["max_sample_bytes"]))
    config["generation_byte_budget"] = math.ceil(2*estimate["estimated_generated_batch_bytes"])
    config["sample_byte_budget"] = max(config["sample_byte_budget"], math.ceil(1.25*config["max_nodes"]*max(replay[kind]["max_sample_bytes"] for kind in ("advantage", "strategy"))))
    return validate_config(config)


def opening_probe_gates(predictions: dict) -> dict:
    """Provisional card-signal gates on all exact combos in one fixed root."""
    if len(predictions["combos"]) != 1326 or len(predictions["probabilities"]) != 1326:
        raise ValueError("All 1326 exact-combo predictions are required")
    groups = {label: [] for label in ("AA", "KK", "72o", "32o")}
    for combo, probabilities in zip(predictions["combos"], predictions["probabilities"]):
        high, low = sorted((card // 4 for card in combo), reverse=True)
        ranks = "23456789TJQKA"
        label = ranks[high]+ranks[low]+("" if high == low else "s" if combo[0] % 4 == combo[1] % 4 else "o")
        if label in groups:
            groups[label].append(probabilities)
    means = {label: [sum(row[index] for row in rows)/len(rows) for index in range(len(predictions["actions"]))]
             for label, rows in groups.items()}
    fold = predictions["actions"].index("FOLD")
    aa_72 = sum(abs(a-b) for a,b in zip(means["AA"], means["72o"]))/2
    kk_32 = sum(abs(a-b) for a,b in zip(means["KK"], means["32o"]))/2
    return {"tv_AA_72o": aa_72, "tv_KK_32o": kk_32,
            "tv_threshold": .15, "tv_separation_pass": aa_72 >= .15 and kk_32 >= .15,
            "strong_fold_below_weak": means["AA"][fold] < means["72o"][fold] and means["KK"][fold] < means["32o"][fold],
            "card_order_exact": predictions["max_card_order_gap"] == 0,
            "class_mean_probabilities": means, "scope": "provisional engineering gates, not an EV or equilibrium guarantee"}
