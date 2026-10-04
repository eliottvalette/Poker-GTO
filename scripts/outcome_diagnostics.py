"""Bounded CPU outcome-sampling diagnostics, never a training workflow.

Every requested path remains in the result, including censored and numerical
failures. Full-tournament lengths are separate from traverser paths that may
stop when an eliminated player's payoff is already settled.
"""
from __future__ import annotations

from dataclasses import asdict
import math
import random
import time
from typing import Any, Callable

from actions import ACTION_IDS, apply_action
from cfr_solver import TraversalBudgetExceeded, sample_index, validate_strategy
from infoset import Observation, observe
from tournament import TournamentState
from outcome_sampling import OutcomeSamplingTraversal, behavior_policy

Strategy = Callable[[Observation], tuple[float, ...]]


def probe_strategy(name: str) -> Strategy:
    """Uniform or modest 1-to-1.5 legal-action weighting, with full target support."""
    if name not in ("uniform", "mild_skew"):
        raise ValueError(f"Unknown diagnostic policy {name!r}; expected uniform or mild_skew")

    def strategy(obs: Observation) -> tuple[float, ...]:
        legal = [i for i, mask in enumerate(obs.legal_mask) if mask]
        weights = [1.0 if name == "uniform" else 1.0 + 0.5 * n / max(1, len(legal) - 1)
                   for n in range(len(legal))]
        denominator = sum(weights)
        result = [0.0] * len(ACTION_IDS)
        for index, weight in zip(legal, weights):
            result[index] = weight / denominator
        distribution = tuple(result)
        validate_strategy(distribution, obs.legal_mask)
        return distribution

    return strategy


def quantiles(values: list[float]) -> dict[str, float | int | None]:
    """Linear empirical quantiles; zeros/-infinity are counted explicitly."""
    finite = sorted(value for value in values if math.isfinite(value))
    if any(math.isnan(value) or value == math.inf for value in values):
        raise ValueError("Quantiles require finite values or exact-zero log reach (-infinity)")

    def at(fraction: float) -> float | None:
        if not finite:
            return None
        offset = fraction * (len(finite) - 1)
        lower = math.floor(offset)
        upper = math.ceil(offset)
        return finite[lower] + (finite[upper] - finite[lower]) * (offset - lower)

    return {"count": len(values), "finite_count": len(finite),
            "zero_count": sum(value == 0.0 for value in values),
            "negative_infinity_count": sum(value == -math.inf for value in values),
            "min": at(0), "p50": at(0.5), "p90": at(0.9), "p95": at(0.95), "max": at(1)}


def json_safe(value: Any) -> Any:
    """Keep logs while rendering true zero reaches as null, never nonstandard JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        if value == -math.inf:
            return None
        raise ValueError(f"Non-JSON diagnostic number {value}")
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def full_tournament_path(seed: int, strategy: Strategy, max_nodes: int,
                         max_depth: int, epsilon: float) -> dict[str, Any]:
    """Independently sample all players under q until winner or explicit censoring."""
    _validate_probe((seed,), max_nodes, max_depth, epsilon)
    tournament = TournamentState(rng=random.Random(seed))
    rng = random.Random(seed + 100000)
    nodes = 0
    decisions = 0
    log_q = 0.0
    row: dict[str, Any] = {"kind": "full_tournament", "seed": seed,
                           "status": "censored", "failure": None}
    started = time.perf_counter()
    try:
        while not tournament.terminal:
            if nodes >= max_nodes or nodes >= max_depth:
                raise TraversalBudgetExceeded(
                    f"Full-path budget exceeded: nodes={nodes}/{max_nodes}, depth={nodes}/{max_depth}, "
                    f"hand={tournament.hand_number}")
            nodes += 1
            if tournament.hand is None or tournament.hand.terminal:
                tournament.start_hand()
                continue
            obs = observe(tournament)
            target = strategy(obs)
            behavior = behavior_policy(target, obs.legal_mask, epsilon)
            validate_strategy(behavior, obs.legal_mask)
            chosen = sample_index(behavior, rng)
            decisions += 1
            log_q += math.log(behavior[chosen])
            apply_action(tournament.hand, ACTION_IDS[chosen])
        row["status"] = "terminal"
    except TraversalBudgetExceeded as error:
        row["failure"] = str(error)
    except (ValueError, ArithmeticError) as error:
        row["status"] = "failure"
        row["failure"] = f"{type(error).__name__}: {error}"
    inverse = _raw_weights([-log_q])
    row.update(nodes=nodes, decisions=decisions, node_accounting="decisions plus hand-start transitions",
               depth=nodes, hand_number=tournament.hand_number,
               winner=tournament.winner, log_inverse_q=-log_q,
               prefix_inverse_q=inverse["values"][0],
               prefix_inverse_q_overflow=bool(inverse["overflow_count"]),
               prefix_inverse_q_underflow=bool(inverse["underflow_count"]),
               seconds=time.perf_counter() - started)
    return row


def _validate_probe(seeds: tuple[int, ...], max_nodes: int, max_depth: int, epsilon: float) -> None:
    if not seeds or any(type(seed) is not int for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError(f"Expected distinct integer diagnostic seeds, got {seeds}")
    if type(max_nodes) is not int or max_nodes < 1 or type(max_depth) is not int or max_depth < 1:
        raise ValueError(f"Diagnostic budgets must be positive integers: nodes={max_nodes}, depth={max_depth}")
    if not math.isfinite(epsilon) or not 0 < epsilon <= 1:
        raise ValueError(f"Diagnostic epsilon must be in (0, 1], got {epsilon}")


def _raw_weights(log_values: list[float]) -> dict[str, Any]:
    values: list[float | None] = []
    overflow_count = underflow_count = structural_zero_count = 0
    for log in log_values:
        if log == -math.inf:
            values.append(0.0)
            structural_zero_count += 1
            continue
        if not math.isfinite(log):
            raise ValueError(f"Invalid diagnostic log weight {log}")
        try:
            weight = math.exp(log)
        except OverflowError:
            values.append(None)
            overflow_count += 1
            continue
        if weight == 0:
            values.append(None)
            underflow_count += 1
        elif not math.isfinite(weight):
            values.append(None)
            overflow_count += 1
        else:
            values.append(weight)
    return {"values": values, "overflow_count": overflow_count,
            "underflow_count": underflow_count, "structural_zero_count": structural_zero_count,
            "finite_positive_count": sum(value is not None and value > 0 for value in values)}


def outcome_path(seed: int, player: int, strategy: Strategy, max_nodes: int,
                 max_depth: int, epsilon: float) -> dict[str, Any]:
    """A bounded current-hand path; the simulator remains independent."""
    _validate_probe((seed,), max_nodes, max_depth, epsilon)
    root = TournamentState(rng=random.Random(seed))
    root.start_hand()
    walk = OutcomeSamplingTraversal(strategy, random.Random(seed + 200000 + player),
                                   max_nodes, max_depth, epsilon)
    counts = {"regret_samples": 0, "average_samples": 0}
    average_weights: list[float] = []

    def regret_sink(obs: Observation, target: tuple[float, ...], weight: float) -> None:
        counts["regret_samples"] += 1

    def average_sink(obs: Observation, target: tuple[float, ...], weight: float) -> None:
        counts["average_samples"] += 1
        average_weights.append(weight)

    started = time.perf_counter()
    value = None
    status = "failure"
    failure = None
    try:
        value = walk.run(root, player, regret_sink, average_sink)
        status = "terminal" if walk.diagnostics.terminal_reason == "terminal" else "settled_payoff"
    except TraversalBudgetExceeded as error:
        status, failure = "censored", str(error)
    except (ValueError, ArithmeticError) as error:
        failure = f"{type(error).__name__}: {error}"
    diagnostics = asdict(walk.diagnostics)
    log_fields = [key for key in diagnostics if key.startswith("log_")]
    return {"kind": "outcome_traverser", "seed": seed, "player": player,
            "status": status, "failure": failure, "value": value,
            "nodes": diagnostics["nodes"], "decisions": diagnostics["decisions"], "hand_number": diagnostics["hand_number"],
            **counts, "average_weights": average_weights,
            "diagnostics": json_safe(diagnostics),
            "log_zero_counts": {key: sum(v == -math.inf for v in diagnostics[key]) for key in log_fields},
            "log_summaries": {key: quantiles(diagnostics[key]) for key in log_fields},
            "raw_importance_weights": {key: _raw_weights(diagnostics[key]) for key in log_fields},
            "seconds": time.perf_counter() - started}


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    full = [row for row in rows if row["kind"] == "full_tournament"]
    traverser = [row for row in rows if row["kind"] == "outcome_traverser"]
    statuses = ("terminal", "settled_payoff", "censored", "failure")
    logs: dict[str, list[float]] = {}
    for row in traverser:
        for key, values in row["diagnostics"].items():
            if key.startswith("log_"):
                logs.setdefault(key, []).extend(-math.inf if value is None else value for value in values)
    raw_groups: dict[str, dict[str, Any]] = {}
    for row in traverser:
        for key, weights in row["raw_importance_weights"].items():
            group = raw_groups.setdefault(key, {"values": [], "overflow_count": 0,
                                                "underflow_count": 0, "structural_zero_count": 0})
            group["values"].extend(value for value in weights["values"] if value is not None)
            for count in ("overflow_count", "underflow_count", "structural_zero_count"):
                group[count] += weights[count]
    raw_summaries = {
        key: {"finite_represented_quantiles": quantiles(group["values"]),
              "overflow_excluded_count": group["overflow_count"],
              "underflow_excluded_count": group["underflow_count"],
              "structural_zero_count": group["structural_zero_count"]}
        for key, group in raw_groups.items()
    }
    return {
        "counts": {kind: {status: sum(row["status"] == status for row in group) for status in statuses}
                   for kind, group in (("full_tournament", full), ("outcome_traverser", traverser))},
        "actual_terminal_hand_lengths": quantiles([float(row["hand_number"]) for row in full if row["status"] == "terminal"]),
        "actual_censored_hand_lengths": quantiles([float(row["hand_number"]) for row in full if row["status"] == "censored"]),
        "traverser_hand_lengths_including_censoring": quantiles([float(row["hand_number"]) for row in traverser]),
        "nodes_including_censoring": quantiles([float(row["nodes"]) for row in rows]),
        "decision_counts_including_censoring": quantiles([float(row["decisions"]) for row in rows]),
        "raw_importance_weights": raw_summaries,
        "logs": {key: quantiles(values) for key, values in logs.items()},
        "full_path_log_inverse_q": quantiles([row["log_inverse_q"] for row in full]),
        "full_path_prefix_inverse_q_overflow_count": sum(row["prefix_inverse_q_overflow"] for row in full),
        "full_path_prefix_inverse_q_underflow_count": sum(row["prefix_inverse_q_underflow"] for row in full),
        "full_path_prefix_inverse_q": quantiles([row["prefix_inverse_q"] for row in full
                                                if row["prefix_inverse_q"] is not None]),
        "max_abs_regret": quantiles([row["diagnostics"]["max_abs_regret"] for row in traverser]),
        "seconds": sum(row["seconds"] for row in rows),
    }


def run_probe(seeds: tuple[int, ...] = (0, 1), max_nodes: int = 300,
              max_depth: int = 300, epsilon: float = 0.6) -> dict[str, Any]:
    """Record every bounded attempt; no retries, filtering, training or file writes.

    Two seeds are the default calibration. A larger seed set must be selected
    explicitly after inspecting that calibration's CPU cost.
    """
    _validate_probe(seeds, max_nodes, max_depth, epsilon)
    rows: list[dict[str, Any]] = []
    for policy_name in ("uniform", "mild_skew"):
        strategy = probe_strategy(policy_name)
        for seed in seeds:
            full = full_tournament_path(seed, strategy, max_nodes, max_depth, epsilon)
            full["policy"] = policy_name
            rows.append(full)
            for player in (0, 1, 2):
                row = outcome_path(seed, player, strategy, max_nodes, max_depth, epsilon)
                row["policy"] = policy_name
                rows.append(row)
    report = {"version": 2, "objective": "hand_chip_delta", "traversal_scope": "current_hand", "epsilon": epsilon,
              "seeds": list(seeds), "max_nodes": max_nodes, "max_depth": max_depth,
              "rows": rows, "summary": _summarize(rows),
              "by_policy": {name: _summarize([row for row in rows if row["policy"] == name])
                            for name in ("uniform", "mild_skew")},
              "interpretation": "Censored paths remain failures; completed subset is not an unbiased training batch."}
    return json_safe(report)
