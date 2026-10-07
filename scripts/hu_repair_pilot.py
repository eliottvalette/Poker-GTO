"""Bounded HU integration pilot; retains checkpoints and full opening probes."""
from __future__ import annotations

import json
from pathlib import Path
import time

from scripts.fit_budget_audit import opening_predictions
from training.checkpoint import atomic_bytes
from training.config import load_config
from training.runner import TrainingRunner


def run_hu_repair_pilot(output: Path, iterations: int = 3, traversals_per_player: int = 4,
                        epochs: int = 20) -> dict:
    """Use unchanged root sampling and replay capacities; never publish a policy."""
    if not 1 <= iterations <= 3 or not 2 <= traversals_per_player <= 8 or not 1 <= epochs <= 50:
        raise ValueError("Pilot limits: 1..3 iterations, 2..8 traversals/player, 1..50 epochs")
    if output.exists():
        raise FileExistsError(f"Pilot output already exists: {output}")
    config = load_config("configs/train_hu.json")
    config.update(output_dir=str(output), outer_iterations=iterations, workers=1,
                  advantage_epochs=epochs, average_epochs=epochs, checkpoint_every=1, evaluation_every=iterations)
    config["hu"]["traversals_per_player"] = traversals_per_player
    output.mkdir(parents=True)
    atomic_bytes(output / "config.json", (json.dumps(config, indent=2) + "\n").encode())
    runner = TrainingRunner(config)
    report = {"config": config, "scope": "integration smoke test, insufficient coverage for poker quality acceptance",
              "fit_budget": f"{epochs} epochs is exploratory; the old-replay sweep does not establish convergence",
              "iterations": []}
    for _ in range(iterations):
        started = time.perf_counter()
        row = runner.run_iteration()
        solver = runner.solvers["hu"]
        summary = {"iteration": runner.iteration, "metrics": row["tracks"]["hu"], "probes": {}}
        for kind, model in (("advantage", solver.advantage_model), ("strategy", solver.average_model)):
            predictions = opening_predictions(model, 2, config["batch_size"])
            filename = f"{kind}-iteration-{runner.iteration:02d}.json"
            atomic_bytes(output / filename, (json.dumps(predictions, allow_nan=False) + "\n").encode())
            summary["probes"][kind] = {"holdings": predictions["holdings"], "action_spread": predictions["action_spread"],
                                      "max_card_order_gap": predictions["max_card_order_gap"], "artifact": filename}
        summary["seconds"] = time.perf_counter() - started
        report["iterations"].append(summary)
        atomic_bytes(output / "report.json", (json.dumps(report, indent=2, allow_nan=False) + "\n").encode())
        metrics = summary["metrics"]["average_policy"]
        print(f"HU pilot iteration={runner.iteration}/{iterations}: seconds={summary['seconds']:.1f}, strategy retained={len(solver.strategy_memory.samples)}/{solver.strategy_memory.capacity}, ESS={metrics['weight_effective_samples']:.1f}", flush=True)
    return report
