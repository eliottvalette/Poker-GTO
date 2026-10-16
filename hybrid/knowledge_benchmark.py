"""Quality-gated amortization, including measured teacher and fitting CPU costs."""
from __future__ import annotations

from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import statistics
import time

import torch

from hybrid.distillation import SolverStudents
from hybrid.knowledge import calculate, cached_policy, game_contract
from hybrid.policy_source import UniformLegalPolicy
from hybrid.range_student_experiments import solve
from hybrid.solver_experiments import bounded_roots
from hybrid.solver_inputs import RootPriorPolicy
from hybrid.state import ComputeBudget, instantiate
from infoset import observe
from training.checkpoint import atomic_bytes


def break_even(direct_cpu: float, query_cpu: float, setup_cpu: float, *, quality_pass: bool) -> dict:
    if any(not math.isfinite(v) or v < 0 for v in (direct_cpu, query_cpu, setup_cpu)):
        raise ValueError("Finite nonnegative measured CPU costs required")
    saving = direct_cpu - query_cpu
    return {"quality_pass": quality_pass, "cpu_saved_per_query": saving,
            "queries_to_amortize": math.ceil(setup_cpu / saving) if quality_pass and saving > 0 else None,
            "reason": "quality_failed" if not quality_pass else "no_online_saving" if saving <= 0 else "measured_break_even"}


def run(output: Path, *, updates: int = 64, repeats: int = 3) -> dict:
    if output.exists() or repeats < 1:
        raise ValueError("Fresh benchmark destination and positive repeat count required")
    output.mkdir(parents=True)
    report = {"scope": "small independent roots; stronger non-river forest remains approximate",
              "tracks": {}, "repeats": repeats, "fit_updates_per_specialist": updates,
              "cost_scope": "CPU including teacher generation/persistence, model initialization, fitting/evaluation and checkpoint serialization",
              "quality_gate": "mixed response loss no greater than direct within 1e-8 BB; descriptive, not statistical acceptance"}
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for count in (2, 3):
            budget = ComputeBudget(iterations=32, samples=32, max_depth=1, max_nodes=1000000, seed=100)
            paths, teacher_cpu, cached = [], 0., {}
            for split in ("train", "validation", "test"):
                for root, ranges in bounded_roots(count, split):
                    path = output / f"{count}-{split}-{root.street}.json"
                    begin = time.process_time()
                    result = calculate(path, root, ranges, budget, split=split)
                    cost = time.process_time() - begin
                    teacher_cpu += cost
                    if split == "test":
                        cached[root.street] = (path, result, cost)
                    paths.append(path)
            begin = time.process_time()
            student = SolverStudents(count, seed=321)
            fit_report = student.fit(paths, updates=updates)
            student.save(output / f"students-{count}.pt")
            fit_cpu = time.process_time() - begin
            spots = []
            for root, ranges in bounded_roots(count, "test"):
                contract = game_contract(root, ranges, budget, "test")
                prior = RootPriorPolicy(student.models["strategy"], contract, UniformLegalPolicy())
                reference = solve(root, ranges, replace(budget, iterations=256, samples=256, seed=101), UniformLegalPolicy())
                methods = {}
                for name, assistance in (("direct", None), ("nn_plus_cfr", prior)):
                    timings = []
                    for _ in range(repeats):
                        begin = time.process_time()
                        result = solve(root, ranges, budget, UniformLegalPolicy(), assistance)
                        timings.append(time.process_time() - begin)
                    methods[name] = (result, statistics.median(timings), timings)
                path, table_result, table_setup = cached[root.street]
                begin = time.process_time()
                table = cached_policy(path, root, ranges)
                load_cpu = time.process_time() - begin
                timings = []
                worlds = list(ranges.enumerate())
                for _ in range(repeats):
                    begin = time.process_time()
                    for deal in worlds:
                        table.probabilities(observe(instantiate(root, deal.hands)))
                    timings.append(time.process_time() - begin)
                # Cost includes querying all feasible deals, conservatively more
                # work than a single known-hand lookup. Values were loaded above.
                methods["cached"] = (table_result, statistics.median(timings), timings)
                rows = {}
                for name, (result, cpu, timings) in methods.items():
                    gaps, ranking_errors, losses = [], [], []
                    for hand, ev in reference.action_ev_by_hand.items():
                        chosen = max(result.action_ev_by_hand[hand], key=result.action_ev_by_hand[hand].get)
                        gap = (max(ev.values()) - ev[chosen]) / root.blinds.big
                        gaps.append(gap)
                        ranking_errors.append(float(gap > 1e-8))
                        losses.append((max(ev.values()) - sum(result.strategy_by_hand[hand][a] * v for a, v in ev.items())) / root.blinds.big)
                    rows[name] = {"cpu_seconds": cpu, "cpu_repeats": timings,
                        "ranking_error_fraction": statistics.fmean(ranking_errors),
                        "argmax_loss_bb": statistics.fmean(gaps), "mixed_loss_bb": statistics.fmean(losses)}
                rows["direct"]["total_cpu_by_queries"] = {str(n): n * rows["direct"]["cpu_seconds"] for n in (1, 10, 100, 1000)}
                for name, setup in (("cached", table_setup + load_cpu), ("nn_plus_cfr", teacher_cpu + fit_cpu)):
                    quality = rows[name]["mixed_loss_bb"] <= rows["direct"]["mixed_loss_bb"] + 1e-8
                    rows[name]["setup_cpu_seconds"] = setup
                    rows[name]["amortization"] = break_even(rows["direct"]["cpu_seconds"], rows[name]["cpu_seconds"], setup, quality_pass=quality)
                    rows[name]["total_cpu_by_queries"] = {str(n): setup + n * rows[name]["cpu_seconds"] for n in (1, 10, 100, 1000)}
                spots.append({"street": root.street, "methods": rows})
            report["tracks"][str(count)] = {"solve_budget": asdict(budget),
                "reference_budget": asdict(replace(budget, iterations=256, samples=256, seed=101)),
                "training_teacher_cpu": teacher_cpu, "fit_cpu": fit_cpu,
                "fit_metrics": fit_report, "spots": spots, "accepted": False}
        atomic_bytes(output / "report.json", json.dumps(report, indent=2, allow_nan=False).encode())
        return report
    finally:
        torch.set_num_threads(old_threads)
