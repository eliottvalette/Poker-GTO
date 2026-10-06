"""Small real iterations only; cost estimates never launch the configured pilot."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import resource
import statistics
import sys
import tempfile
import time
import torch
from features import FEATURE_SCHEMA_VERSION
from features.neural import neural_observation
from infoset import observe
from ml.memory import TrainingSample, sample_bytes
from ml.model import AdvantageNetwork, AveragePolicyNetwork, MODEL_ARCHITECTURE
from training.checkpoint import atomic_bytes
from training.config import config_hash, validate_config
from training.runner import TrainingRunner, source_commit, source_metadata

PREFLIGHT_VERSION = 1


def percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("Percentile requires measured values")
    ordered = sorted(values)
    return ordered[math.ceil((len(ordered) - 1) * probability)]


def run_preflight(config: dict, *, iterations: int = 2, traversals_per_player: int = 4) -> dict:
    config = validate_config(config)
    if type(iterations) is not int or not 1 <= iterations <= 3 or type(traversals_per_player) is not int or not 2 <= traversals_per_player <= 8:
        raise ValueError(f"Preflight bounds: iterations=1..3, traversals/player=2..8; got {iterations}, {traversals_per_player}")
    original_threads = torch.get_num_threads()
    child_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    child_cpu_started = child_usage.ru_utime + child_usage.ru_stime
    cpu_started, started = time.process_time(), time.perf_counter()
    report = {**source_metadata(), "version": PREFLIGHT_VERSION, "git_commit": source_commit(), "config_hash": config_hash(config),
              "feature_schema": FEATURE_SCHEMA_VERSION, "model_architecture": MODEL_ARCHITECTURE,
              "parameters_per_model": sum(p.numel() for p in AdvantageNetwork().parameters()),
              "objective": "hand_chip_delta", "traversal_mode": config["traversal_mode"],
              "workers": config["workers"], "worker_torch_threads": 1, "trainer_threads": config["trainer_threads"],
              "gpu_usage": "none", "outer_iterations_proposed": config["outer_iterations"],
              "root_sampler_distribution": {name: config[name]["root_sampling"] for name in ("3max", "hu") if config[name]["enabled"]},
              "traversal_unit": "one regret traversal plus one independent average-policy pass per task",
              "preflight_iterations_requested": iterations, "preflight_traversals_per_player": traversals_per_player,
              "measured_iterations": [], "failure": None}
    report["traversals_per_iteration_proposed"] = sum((3 if name == "3max" else 2) * config[name]["traversals_per_player"] for name in ("3max", "hu") if config[name]["enabled"])
    report["expected_total_traversals"] = report["traversals_per_iteration_proposed"] * config["outer_iterations"]
    try:
        with tempfile.TemporaryDirectory(prefix="poker-deep-cfr-preflight-") as directory:
            measured = deepcopy(config)
            measured.update(output_dir=directory, outer_iterations=iterations, checkpoint_every=1, evaluation_every=1)
            for name in ("3max", "hu"):
                measured[name]["traversals_per_player"] = traversals_per_player
            runner = TrainingRunner(measured)
            observations = [observe(hand) for roots in runner.probes.values() for _, hand in roots]
            report["observation_bytes_before"] = statistics.mean(sample_bytes(o) for o in observations)
            report["observation_bytes_after"] = statistics.mean(sample_bytes(neural_observation(o)) for o in observations)
            references = [TrainingSample(1, o.hero, o, tuple(float(m) / sum(o.legal_mask) for m in o.legal_mask),
                                         1.0, "strategy", 0) for o in observations]
            report["replay_sample_reference"] = {
                "before_mean": statistics.mean(sample_bytes(s) - sample_bytes(s.state) + sample_bytes(o)
                                               for s, o in zip(references, observations)),
                "after_mean": statistics.mean(sample_bytes(s) for s in references),
                "method": "matched fixed states and metadata; raw Observation versus compact NeuralObservation with identical conservative accounting"}

            for _ in range(iterations):
                try:
                    row = runner.run_iteration()
                    report["measured_iterations"].append(row)
                except Exception as error:
                    report["failure"] = {"iteration": runner.iteration + 1, "type": type(error).__name__, "message": str(error),
                                         "semantics": "iteration aborted atomically; no samples dropped or partial models published"}
                    break
            if runner.iteration:
                runner.export(onnx=True)
                report["checkpoint_bytes"] = (Path(directory) / "checkpoints" / f"iteration_{runner.iteration:06d}.pt").stat().st_size
                report["onnx_model_bytes"] = {name: (Path(directory) / "policy" / f"average_{name}.onnx").stat().st_size for name in runner.solvers}
                report["preflight_artifact_bytes"] = sum(p.stat().st_size for p in Path(directory).rglob('*') if p.is_file())
                samples = [s for solver in runner.solvers.values() for memory in (*solver.advantage_memory.values(), solver.strategy_memory) for s in memory.samples]
                sizes = [sample_bytes(s) for s in samples]
                report["bytes_per_sample"] = {"mean": statistics.mean(sizes), "p95": percentile(sizes, .95), "max": max(sizes),
                                             "measurement": "conservative Python-owned accounting; excludes allocator and process overhead"}
                report["bytes_per_sample"]["by_kind"] = {kind: {
                    "mean": statistics.mean(sample_bytes(s) for s in samples if s.kind == kind),
                    "count": sum(s.kind == kind for s in samples)} for kind in ("advantage", "strategy")}
                report["serialized_sample_bytes_mean"] = statistics.mean(len(json.dumps(asdict(s), default=lambda value: value.hex()).encode()) for s in samples)
                rows = [track for row in report["measured_iterations"] for track in row["tracks"].values()]
                nodes = [n for row in rows for n in row["nodes_per_traversal"]]
                generation_seconds = sum(row["generation_seconds"] + row["root_seconds"] for row in rows)
                report.update(tasks_per_second=len(nodes) / generation_seconds, traversals_per_second=len(nodes) / generation_seconds,
                              nodes_per_second=sum(nodes) / generation_seconds, average_nodes_per_traversal=statistics.mean(nodes),
                              p95_nodes_per_traversal=percentile(nodes, .95),
                              samples_per_traversal=sum(row["advantage_samples"] + row["strategy_samples"] for row in rows) / len(nodes),
                              worker_peak_rss_bytes=max(row["worker_peak_rss_bytes"] for row in rows),
                              worker_cpu_seconds=sum(row["worker_cpu_seconds"] for row in rows))
                report["policy_phases"] = [{"iteration": row["iteration"], "policy": "uniform" if row["iteration"] == 1 else "learned_nonuniform",
                                          "generation_seconds": sum(track["generation_seconds"] for track in row["tracks"].values()),
                                          "root_seconds": sum(track["root_seconds"] for track in row["tracks"].values()),
                                          "fit_seconds": sum(track["fit_seconds"] for track in row["tracks"].values()),
                                          "nodes": sum(track["nodes"] for track in row["tracks"].values())} for row in report["measured_iterations"]]
                advantage_models = sum(len(s.players) for s in runner.solvers.values())
                strategies = len(runner.solvers)
                capacity = advantage_models * config["advantage_capacity"] + strategies * config["strategy_capacity"]
                retained = sum(len(m.samples) for s in runner.solvers.values() for m in (*s.advantage_memory.values(), s.strategy_memory))
                replay_ram = capacity * report["bytes_per_sample"]["mean"]
                fit_seconds = sum(row["fit_seconds"] for row in rows) / runner.iteration
                generation_iteration = report["traversals_per_iteration_proposed"] / report["traversals_per_second"]
                full_fit = fit_seconds * capacity / retained
                evaluation_seconds = max(0, statistics.mean(row["wall_seconds"] for row in report["measured_iterations"]) - generation_seconds / runner.iteration - fit_seconds)
                iteration_estimate = generation_iteration + full_fit + evaluation_seconds / config["evaluation_every"]
                report["replay_capacities"] = {"advantage_per_model": config["advantage_capacity"], "strategy_per_track": config["strategy_capacity"], "total_samples": capacity}
                report["ram_estimate"] = {"retained_replay_bytes": replay_ram, "staged_replay_conservative_bytes": 3 * replay_ram,
                                          "configured_replay_budget_bytes": (advantage_models + strategies) * config["memory_byte_budget"],
                                          "generation_budget_bytes": config["generation_byte_budget"],
                                          "pending_worker_sample_budget_bytes": config["workers"] * config["sample_byte_budget"]}
                report["runtime_estimates_seconds"] = {"pilot": iteration_estimate * config["outer_iterations"],
                    "500_iterations": iteration_estimate * 500, "generation_per_iteration": generation_iteration,
                    "fit_at_full_capacity_per_iteration": full_fit,
                    "method": "measured throughput including root generation and spawn overhead; linear full-replay fit scaling; conservative extrapolation, not a guarantee"}
                report["expected_artifact_footprint_bytes"] = report["checkpoint_bytes"] * capacity / retained * (math.ceil(config["outer_iterations"] / config["checkpoint_every"]) + 1)
                report["artifact_estimate_method"] = "checkpoint size scaled by replay capacity and retained checkpoint count; excludes logs and allocator overhead"
            report["readiness"] = "PILOT READY" if runner.iteration == iterations and report["failure"] is None else "NOT READY"
    finally:
        torch.set_num_threads(original_threads)
    elapsed = time.perf_counter() - started
    report["preflight_wall_seconds"] = elapsed
    report["trainer_cpu_seconds"] = time.process_time() - cpu_started
    report["trainer_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
    child_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    report["child_cpu_seconds_including_spawn"] = child_usage.ru_utime + child_usage.ru_stime - child_cpu_started
    report["average_cpu_cores_used"] = (report["child_cpu_seconds_including_spawn"] + report["trainer_cpu_seconds"]) / elapsed
    report["logical_cpu_count"] = os.cpu_count()
    report["average_host_cpu_utilization_percent"] = 100 * report["average_cpu_cores_used"] / report["logical_cpu_count"]
    if "ram_estimate" in report:
        ram = report["ram_estimate"]
        ram["projected_total_bytes"] = (ram["staged_replay_conservative_bytes"] + report["trainer_peak_rss_bytes"]
                                        + config["workers"] * report["worker_peak_rss_bytes"]
                                        + ram["generation_budget_bytes"] + ram["pending_worker_sample_budget_bytes"])
        ram["method"] = "3 replay copies + measured trainer/worker RSS + configured generated/pending sample budgets; allocator growth is unmodeled"
    report["approval_required"] = "Substantial pilot/full runs require explicit approval after reviewing this report"
    destination = Path(config["output_dir"]) / "preflight.json"
    atomic_bytes(destination, (json.dumps(report, indent=2, allow_nan=False) + "\n").encode())
    return report
