"""Bounded comparisons of hand-terminal cEV estimators; no neural training."""
from __future__ import annotations

from dataclasses import asdict
import math
import random
import statistics
import time

from actions import ACTION_IDS
from cfr_solver import Traversal, TraversalBudgetExceeded, regret_matching
from outcome_sampling import OutcomeSamplingTraversal
from infoset import observe
from poker_game_expresso import HandState
from scripts.outcome_diagnostics import json_safe, quantiles, _raw_weights

SCENARIOS = {
    "3max_25bb": {0: 25.0, 1: 25.0, 2: 25.0},
    "3max_5bb": {0: 5.0, 1: 5.0, 2: 5.0},
    "hu_25bb": {0: 25.0, 1: 25.0},
}


def uniform(obs):
    return regret_matching([0.0] * len(ACTION_IDS), obs.legal_mask)


def _moments(values: list[float]) -> dict:
    return {"n": len(values), "mean": statistics.fmean(values) if values else None,
            "standard_error": statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else None}


def summarize(rows: list[dict]) -> dict:
    successful = [row for row in rows if row["status"] == "completed"]
    regrets = [abs(value) for row in successful for vector, mask in zip(row["regret_targets"], row["regret_masks"])
               for value, legal in zip(vector, mask) if legal]
    weights = [weight for row in successful for weight in row["average_weights"]]
    corrections = [value for row in successful for value in row["regret_correction_weights"] if value is not None]
    return {"attempts": len(rows), "completed": len(successful), "censored": sum(r["status"] == "censored" for r in rows),
            "failures": sum(r["status"] == "error" for r in rows),
            "nodes": quantiles([float(r["nodes"]) for r in rows]),
            "depth": quantiles([float(r["depth"]) for r in rows]),
            "seconds": quantiles([r["seconds"] for r in rows]),
            "regret_seconds": quantiles([r["regret_seconds"] for r in rows]),
            "absolute_regret_targets": quantiles(regrets),
            "regret_distribution_scope": "legal action slots only, including legitimate zero targets", "average_weights": quantiles(weights),
            "regret_correction_weights": quantiles(corrections),
            "value": _moments([row["value"] for row in successful]),
            "value_mean_is_uncensored": len(successful) == len(rows)}


def run_hand_probe(trials: int = 64, max_nodes: int = 2000, max_depth: int = 100,
                   epsilon: float = 0.6, deal_seed: int = 3) -> dict:
    """Compare the same fixed-deal roots for every active player and estimator.

    Fixed deals isolate action-sampling error; this is not a population poker EV
    estimate. Every budget failure remains in the report. Failed external passes
    discard all partially emitted targets before distribution summaries.
    """
    if type(trials) is not int or trials < 2:
        raise ValueError(f"At least two explicit diagnostic trials required, got {trials}")
    if type(max_nodes) is not int or max_nodes < 1 or type(max_depth) is not int or max_depth < 0:
        raise ValueError(f"Invalid traversal budgets: nodes={max_nodes}, depth={max_depth}")
    if not math.isfinite(epsilon) or not 0 < epsilon <= 1 or type(deal_seed) is not int:
        raise ValueError(f"Invalid epsilon/deal seed: {epsilon}, {deal_seed}")
    rows = []
    started = time.perf_counter()
    for name, stacks in SCENARIOS.items():
        root = HandState.start(stacks, 0, random.Random(deal_seed))
        root_observation = observe(root)
        for player in root.players:
            for seed in range(trials):
                for mode in ("external_sampling", "outcome_sampling"):
                    walk = (Traversal(uniform, random.Random(seed), max_nodes, max_depth) if mode == "external_sampling"
                            else OutcomeSamplingTraversal(uniform, random.Random(seed), max_nodes, max_depth, epsilon))
                    targets, masks, averages = [], [], []
                    def regret_sink(obs, vector, weight):
                        targets.append(list(vector))
                        masks.append(list(obs.legal_mask))
                        if obs.hero == root.current_player and len(obs.history) == len(root_observation.history):
                            row["root_regret_target"] = list(vector)
                    def average_sink(obs, vector, weight):
                        averages.append(weight)
                    before = time.perf_counter()
                    row = {"scenario": name, "player": player, "seed": seed, "mode": mode,
                           "status": "completed", "error": None, "value": None, "regret_seconds": None, "root_regret_target": None}
                    try:
                        if mode == "external_sampling":
                            row["value"] = walk.regrets(root, player, regret_sink)
                            row["regret_seconds"] = time.perf_counter() - before
                            # Distinct averaging traversal does not inflate the regret-node count.
                            average_walk = Traversal(uniform, random.Random(seed + 10000), max_nodes, max_depth)
                            average_walk.average(root, player, average_sink)
                            row["average_nodes"] = average_walk.nodes
                        else:
                            row["value"] = walk.run(root, player, regret_sink, average_sink)
                    except (TraversalBudgetExceeded, ValueError, ArithmeticError) as error:
                        row["status"] = "censored" if isinstance(error, TraversalBudgetExceeded) else "error"
                        row["error"] = str(error)
                        targets.clear()
                        masks.clear()
                        averages.clear()
                        row["value"] = None
                        row["root_regret_target"] = None
                    row["seconds"] = time.perf_counter() - before
                    if row["regret_seconds"] is None:
                        row["regret_seconds"] = row["seconds"]
                    row["nodes"] = walk.nodes
                    row["depth"] = walk.max_depth_seen
                    row["regret_targets"] = targets
                    row["regret_masks"] = masks
                    row["average_weights"] = averages
                    if mode == "outcome_sampling":
                        row["diagnostics"] = json_safe(asdict(walk.diagnostics))
                        raw = _raw_weights(walk.diagnostics.log_regret_total_weights)
                        row["regret_correction_weights"] = raw["values"]
                        row["regret_correction_representation"] = {k: v for k, v in raw.items() if k != "values"}
                    else:
                        row["regret_correction_weights"] = [1.0] * len(targets)
                    rows.append(row)
    comparisons = []
    by_scenario = {}
    root_regret_comparisons = []
    for name, stacks in SCENARIOS.items():
        by_scenario[name] = {mode: summarize([row for row in rows if row["scenario"] == name and row["mode"] == mode])
                             for mode in ("external_sampling", "outcome_sampling")}
        for player in stacks:
            estimates = {mode: summarize([r for r in rows if r["scenario"] == name and r["player"] == player and r["mode"] == mode])
                         for mode in ("external_sampling", "outcome_sampling")}
            valid = all(result["value_mean_is_uncensored"] for result in estimates.values())
            a, b = [estimates[mode]["value"] for mode in ("external_sampling", "outcome_sampling")]
            gap = abs(a["mean"] - b["mean"]) if valid else None
            paired = {mode: {r["seed"]: r["value"] for r in rows
                             if r["scenario"] == name and r["player"] == player and r["mode"] == mode}
                      for mode in estimates}
            differences = [paired["external_sampling"][seed] - paired["outcome_sampling"][seed]
                           for seed in range(trials)] if valid else []
            standard_error = statistics.stdev(differences) / math.sqrt(trials) if valid else None
            comparisons.append({"scenario": name, "player": player, "estimates": {k: v["value"] for k, v in estimates.items()},
                                "uncensored": valid, "absolute_mean_gap": gap, "paired_difference_standard_error": standard_error,
                                "within_six_standard_errors": gap <= 6 * standard_error + 1e-10 if valid else None})
        root_rows = {mode: {row["seed"]: row["root_regret_target"] for row in rows
                            if row["scenario"] == name and row["mode"] == mode and row["player"] == 0}
                     for mode in ("external_sampling", "outcome_sampling")}
        for action_index, action in enumerate(ACTION_IDS):
            if any(vector is None for group in root_rows.values() for vector in group.values()):
                continue  # No conditional successful-subset comparison on failure.
            samples = {mode: [group[seed][action_index] for seed in range(trials)] for mode, group in root_rows.items()}
            a, b = samples["external_sampling"], samples["outcome_sampling"]
            delta = [left - right for left, right in zip(a, b)]
            se = statistics.stdev(delta) / math.sqrt(trials)
            gap = abs(statistics.fmean(delta))
            root_regret_comparisons.append({"scenario": name, "player": 0, "action": action,
                                           "external": _moments(a), "outcome": _moments(b),
                                           "absolute_mean_gap": gap, "paired_difference_standard_error": se,
                                           "within_six_standard_errors": gap <= 6 * se + 1e-10})
    return {"version": 1, "objective": "hand_chip_delta", "utility_units": "initial_big_blind_chips",
            "payout_scope": "winner_take_all", "traversal_scope": "current_hand", "epsilon": epsilon,
            "trials_per_player_and_mode": trials, "deal_seed": deal_seed,
            "max_nodes": max_nodes, "max_depth": max_depth,
            "policy": "uniform", "rows": rows, "by_scenario": by_scenario, "comparisons": comparisons,
            "root_regret_comparisons": root_regret_comparisons,
            "seconds": time.perf_counter() - started,
            "interpretation": "Finite diagnostic batch, not learned-policy convergence or full-game exploitability."}
