"""Independent-root evaluation of exact-range conditioned optional students."""
from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
import random
import resource
import sys
import statistics
import time

import torch

from actions import ACTION_IDS, legal_actions
from evaluation import subgame_best_response
from hybrid.distillation import SolverStudents
from hybrid.knowledge import calculate, cached_policy, game_contract
from hybrid.policy_source import UniformLegalPolicy, category_policy
from hybrid.ranges import HandRange, JointRanges, combo
from hybrid.river_solver import RiverSolver
from hybrid.solver_inputs import INPUT_SCHEMA, RootPriorPolicy, continuation_id
from hybrid.state import ComputeBudget, instantiate
from infoset import observe
from poker_game_expresso import HandState
from training.checkpoint import atomic_bytes


def curriculum(count: int, split: str, roots_per_street: int):
    """Legal roots, three candidate combos per seat, paired nonuniform priors.

    Root families stay in one split. Narrow ranges make exhaustive references
    affordable; these are not representative of full preflop range coverage.
    """
    if type(roots_per_street) is not int or not 1 <= roots_per_street <= 64:
        raise ValueError("Curriculum requires 1..64 independent roots per street")
    split_id = {"train": 0, "validation": 1, "test": 2}[split]
    policies = (UniformLegalPolicy(), category_policy({"FOLD": .1, "CHECK": 1, "CALL": 2, "RAISE": .2, "ALL_IN": .1}, "passive-continuation-v1"))
    for street_index, street in enumerate(("PREFLOP", "FLOP", "TURN", "RIVER")):
        for index in range(roots_per_street):
            rng = random.Random(923000 + count * 10000 + split_id * 1000 + street_index * 100 + index)
            stack = (1.5 + (index % 4) * .1 + split_id * .03 if street == "RIVER" else
                     (5., 12., 25., 8.)[index % 4] + split_id * .17)
            root = HandState.start({p: stack * (1 + .15 * p * (index % 2)) for p in range(count)}, index % count, rng)
            while root.street != street:
                root.act("CALL" if root.to_call() else "CHECK")
            if index % 2 and street != "RIVER":
                raises = [a for a in legal_actions(root) if a.category == "RAISE" and a.action_id != "ALL_IN"]
                if not raises:
                    raise ValueError("Curriculum requires a legal opening raise")
                raises[0].apply(root)
            available = [c for c in range(52) if c not in root.board]
            support = {}
            for p, player in root.players.items():
                hands = {combo(player.cards)}
                while len(hands) < 3:
                    hands.add(combo(tuple(rng.sample(available, 2))))
                support[p] = sorted(hands)
            for variant in range(2):
                weights = (1., 2., 7.) if variant == 0 else (7., 2., 1.)
                ranges = JointRanges({p: HandRange(dict(zip(hands, weights))) for p, hands in support.items()}, tuple(root.board))
                for profile in policies[:1] if street == "RIVER" else policies:
                    name = f"{count}-{split}-{street}-{index}-range{variant}-{profile.version}"
                    yield name, root, ranges, profile


def solve(root, ranges, budget, profile, prior=None):
    solver = RiverSolver(prior)
    return solver.solve(root, ranges, budget) if root.street == "RIVER" else solver.solve_chance(root, ranges, budget, continuation=profile)


def preflight() -> list[dict]:
    budget = ComputeBudget(iterations=32, samples=64, max_depth=1, max_nodes=2000000, seed=120)
    rows = []
    for count in (2, 3):
        for name, root, ranges, profile in curriculum(count, "train", 1):
            start = time.perf_counter()
            result = solve(root, ranges, budget, profile)
            rows.append({"name": name, "seconds": time.perf_counter() - start, "nodes": result.work.nodes,
                         "infosets": len(result.policy.table), "joint_deals": len(ranges.enumerate())})
    return rows


def run(output: Path, *, train_roots: int = 4, heldout_roots: int = 2,
        iterations: int = 64, updates: int = 128) -> dict:
    if output.exists():
        raise ValueError(f"Separate experiment output required: {output}")
    output.mkdir(parents=True)
    budget = ComputeBudget(iterations=iterations, samples=64, max_depth=1, max_nodes=3000000, seed=120)
    report = {"budget": asdict(budget), "comparison_budget": asdict(replace(budget, iterations=iterations * 2, samples=128, seed=121)),
              "input_schema": INPUT_SCHEMA, "target_schema": "average-profile-root-advantage-bb-v1",
              "train_roots_per_street": train_roots, "heldout_roots_per_street": heldout_roots,
              "student_seed": 139, "updates": updates, "tracks": {},
              "scope": "root-only, narrow exact-combo ranges; not general poker-strength validation"}
    started = time.perf_counter()
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for count in (2, 3):
            paths, generation, tests = [], [], []
            vocabulary = set()
            for split, size in (("train", train_roots), ("validation", heldout_roots), ("test", heldout_roots)):
                for name, root, ranges, profile in curriculum(count, split, size):
                    path = output / f"{name}.json"
                    begin = time.perf_counter()
                    result = calculate(path, root, ranges, budget, split=split, continuation=profile)
                    contract = game_contract(root, ranges, budget, split, profile)
                    vocabulary.add(continuation_id(contract))
                    generation.append({"name": name, "seconds": time.perf_counter() - begin,
                        "nodes": result.work.nodes, "samples": result.work.samples,
                        "bytes": path.stat().st_size, "holdings": len(result.strategy_by_hand),
                        "board": root.board, "stacks": root.initial_stacks, "actor": root.current_player})
                    paths.append(path)
                    if split == "test":
                        tests.append((path, root, ranges, profile, result))
                print(f"{count}p {split}: {len(paths)} computed games", flush=True)
            variants = {}
            for conditioned in (False, True):
                label = "exact_ranges" if conditioned else "observation_only_ablation"
                learner = SolverStudents(count, seed=139, continuations=tuple(sorted(vocabulary)), conditioned=conditioned)
                begin = time.perf_counter()
                metrics = learner.fit(paths, updates=updates)
                checkpoint = output / f"students-{count}-{label}.pt"
                learner.save(checkpoint)
                fit_seconds = time.perf_counter() - begin
                decisions = []
                for path, root, ranges, profile, low in tests:
                    contract = game_contract(root, ranges, budget, "test", profile)
                    begin = time.perf_counter()
                    table = cached_policy(path, root, ranges)
                    table_load = time.perf_counter() - begin
                    high_budget = replace(budget, iterations=iterations * 2, samples=128, seed=121)
                    begin = time.perf_counter()
                    high = solve(root, ranges, high_budget, profile)
                    high_seconds = time.perf_counter() - begin
                    query_seconds = {"table": 0., "student": 0.}
                    regrets, gaps = [], []
                    used = set()
                    for deal in ranges.enumerate():
                        hand = deal.hands[root.current_player]
                        if hand in used:
                            continue
                        used.add(hand)
                        obs = observe(instantiate(root, deal.hands))
                        start = time.perf_counter()
                        bp = learner.models["strategy"].probabilities(obs, contract)
                        ap = learner.models["advantage"].predict(obs, contract)
                        query_seconds["student"] += time.perf_counter() - start
                        start = time.perf_counter()
                        tp = table.probabilities(obs)
                        query_seconds["table"] += time.perf_counter() - start
                        ev = high.action_ev_by_hand[hand]
                        baseline = sum(high.strategy_by_hand[hand][a] * v for a, v in ev.items())
                        regrets.extend((ap[ACTION_IDS.index(a)] - (v - baseline) / root.blinds.big) ** 2 for a, v in ev.items())
                        gaps.append({"hand": hand,
                            "action_ev_rmse_bb": (sum(((low.action_ev_by_hand[hand][a] - v) / root.blinds.big) ** 2 for a, v in ev.items()) / len(ev)) ** .5,
                            "root_response_shortfall_bb": {name: (max(ev.values()) - sum(prob[ACTION_IDS.index(a)] * v for a, v in ev.items())) / root.blinds.big
                                for name, prob in (("student", bp), ("table", tp))}})
                    comparisons = []
                    if root.street == "RIVER":
                        worlds = [(d.probability, instantiate(root, d.hands)) for d in ranges.enumerate()]
                        for limit in (8, 32):
                            for assisted in (False, True):
                                prior = RootPriorPolicy(learner.models["strategy"], contract, UniformLegalPolicy()) if assisted else None
                                start = time.perf_counter()
                                candidate = solve(root, ranges, replace(budget, iterations=limit), profile, prior)
                                seconds = time.perf_counter() - start
                                gain = sum(subgame_best_response(worlds, candidate.policy.probabilities, p)["best_response_gain_bb"] for p in root.players)
                                comparisons.append({"iterations": limit, "assisted": assisted, "seconds": seconds,
                                    "nodes": candidate.work.nodes, "response_gain_bb": gain})
                    decisions.append({"game": path.stem, "high_seconds": high_seconds,
                        "table_load_seconds": table_load, "query_seconds": query_seconds,
                        "advantage_rmse_bb": (sum(regrets) / len(regrets)) ** .5,
                        "holdings": gaps, "matched_iterations": comparisons})
                variants[label] = {"fit_seconds": fit_seconds, "checkpoint_bytes": checkpoint.stat().st_size,
                    "specialists": metrics, "decisions": decisions, "accepted": False}
                print(f"{count}p {label}: fit {fit_seconds:.2f}s; held-out comparisons complete", flush=True)
            report["tracks"][str(count)] = {"generation": generation, "variants": variants}
        report["seconds"] = time.perf_counter() - started
        report["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        atomic_bytes(output / "report.json", json.dumps(report, indent=2, allow_nan=False).encode())
        return report
    finally:
        torch.set_num_threads(old_threads)


def range_contrast(output: Path, *, updates: int = 256) -> dict:
    """Identifiability test, fitted on two exact labels; not held-out validation."""
    from itertools import combinations
    from utils import rank7
    from hybrid.knowledge import root_supervision
    from hybrid.solver_inputs import encode_roots, RIVER_CONTINUATION
    if output.exists():
        raise ValueError(f"Separate contrast output required: {output}")
    output.mkdir(parents=True)
    root = HandState.start({0: 1.5, 1: 1.5}, 0, random.Random(3))
    while root.street != "RIVER":
        root.act("CALL" if root.to_call() else "CHECK")
    next(a for a in legal_actions(root) if a.action_id == "ALL_IN").apply(root)
    actor, opponent = root.current_player, next(p for p in root.players if p != root.current_player)
    hands = list(combinations([c for c in range(52) if c not in root.board], 2))
    selected = None
    for hero in hands:
        rank = rank7((*hero, *root.board))
        stronger = [h for h in hands if not set(h).intersection(hero) and rank7((*h, *root.board)) > rank]
        weaker = [h for h in hands if not set(h).intersection(hero) and rank7((*h, *root.board)) < rank]
        if stronger and weaker:
            selected = hero, stronger[0], weaker[0]
            break
    if selected is None:
        raise ValueError("Contrast requires both winning and losing opponent holdings")
    hero, strong, weak = selected
    rows, contracts = [], []
    for i, weights in enumerate(((.99, .01), (.01, .99))):
        ranges = JointRanges({actor: HandRange({hero: 1}), opponent: HandRange(dict(zip((strong, weak), weights)))}, tuple(root.board))
        path = output / f"range-{i}.json"
        calculate(path, root, ranges, ComputeBudget(iterations=200), split="train")
        contract, samples = root_supervision(path, "strategy")
        contracts.append(contract)
        rows.append(samples[0])
    if rows[0].state != rows[1].state:
        raise ValueError("Range contrast accidentally changed the observable state")
    inputs = encode_roots([s.state for s in rows], contracts, (RIVER_CONTINUATION,))
    targets = torch.tensor([s.target for s in rows])
    report = {"scope": "two training examples only; exact label identifiability, not generalization",
              "updates": updates, "hero": hero, "board": root.board,
              "strong_opponent": strong, "weak_opponent": weak,
              "call_targets": targets[:, ACTION_IDS.index("CALL")].tolist(), "variants": {}}
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for conditioned in (False, True):
            learner = SolverStudents(2, seed=845, continuations=(RIVER_CONTINUATION,), conditioned=conditioned)
            model = learner.models["strategy"].specialists["RIVER"]
            optimizer = learner.optimizers["strategy", "RIVER"]
            start = time.perf_counter()
            for _ in range(updates):
                optimizer.zero_grad(set_to_none=True)
                prediction = model(inputs)
                loss = (targets * (targets.clamp_min(1e-12).log() - prediction.clamp_min(1e-12).log())).sum(1).mean()
                loss.backward()
                optimizer.step()
            learner.models["strategy"].trained[3] = True
            learner.updates["strategy", "RIVER"] = updates
            label = "conditioned" if conditioned else "observation_only"
            learner.save(output / f"{label}.pt")
            with torch.no_grad():
                prediction = model(inputs)
                kl = (targets * (targets.clamp_min(1e-12).log() - prediction.clamp_min(1e-12).log())).sum(1).mean()
            report["variants"][label] = {"kl": float(kl), "call_predictions": prediction[:, ACTION_IDS.index("CALL")].tolist(),
                                          "seconds": time.perf_counter() - start}
        atomic_bytes(output / "report.json", json.dumps(report, indent=2, allow_nan=False).encode())
        return report
    finally:
        torch.set_num_threads(old_threads)


def cpu_comparison(output: Path) -> dict:
    """Select feasible measured candidates under the same per-spot CPU ceiling.

    Repeated runs at 8/16/24/32/64/128 iterations define a cost/quality curve; this is
    an offline benchmark, not a time-interruptible production solver scheduler.
    """
    report = {"scope": "exact narrow river games, same median CPU ceiling on a discrete budget grid; optional root prior only", "repeats": 5, "tracks": {}}
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for count in (2, 3):
            name, root, ranges, profile = next(row for row in curriculum(count, "test", 1) if row[1].street == "RIVER")
            budget = ComputeBudget(max_nodes=3000000, iterations=32, samples=64, max_depth=1, seed=120)
            contract = game_contract(root, ranges, budget, "test", profile)
            student = SolverStudents.load(output / f"students-{count}-exact_ranges.pt")
            prior = RootPriorPolicy(student.models["strategy"], contract, UniformLegalPolicy())
            worlds = [(d.probability, instantiate(root, d.hands)) for d in ranges.enumerate()]
            candidates = []
            for iterations in (8, 16, 24, 32, 64, 128):
                for assisted in (False, True):
                    timings = []
                    for repeat in range(5):
                        begin = time.process_time()
                        candidate = solve(root, ranges, replace(budget, iterations=iterations), profile, prior if assisted else None)
                        timings.append(time.process_time() - begin)
                        if repeat and candidate.action_ev_by_hand != result.action_ev_by_hand:
                            raise ValueError("Repeated CPU benchmark changed numerical outputs")
                        result = candidate
                    cpu = statistics.median(timings)
                    gain = sum(subgame_best_response(worlds, result.policy.probabilities, p)["best_response_gain_bb"] for p in root.players)
                    candidates.append({"method": "student_plus_cfr" if assisted else "direct_cfr",
                        "iterations": iterations, "cpu_seconds": cpu, "cpu_repeats": timings, "nodes": result.work.nodes, "response_gain_bb": gain})
            ceiling = next(c["cpu_seconds"] for c in candidates if c["method"] == "direct_cfr" and c["iterations"] == 32)
            selected = {}
            for method in ("direct_cfr", "student_plus_cfr"):
                feasible = [c for c in candidates if c["method"] == method and c["cpu_seconds"] <= ceiling]
                selected[method] = max(feasible, key=lambda c: c["iterations"]) if feasible else {"status": "no measured candidate within budget"}
            start = time.process_time()
            cached = cached_policy(output / f"{name}.json", root, ranges)
            load_cpu = time.process_time() - start
            start = time.process_time()
            for _, world in worlds:
                cached.probabilities(observe(world))
            query_cpu = time.process_time() - start
            report["tracks"][str(count)] = {"cpu_ceiling_seconds": ceiling, "selected": selected, "curve": candidates,
                "cached_table": {"load_cpu_seconds": load_cpu, "query_cpu_seconds": query_cpu,
                    "queries": len(worlds), "generation_cost": "stored separately in main report; reuse is not free initial solving"}}
        atomic_bytes(output / "cpu-comparison.json", json.dumps(report, indent=2).encode())
        return report
    finally:
        torch.set_num_threads(old_threads)


def label_stability(output: Path) -> dict:
    """Measure finite-forest action-value dispersion before enlarging training."""
    report = {"scope": "two held-out turn roots; cross-seed dispersion is not error against truth",
              "iterations": 32, "max_depth": 1, "max_nodes": 3000000, "tracks": {}}
    for count in (2, 3):
        name, root, ranges, profile = next(row for row in curriculum(count, "test", 1) if row[1].street == "TURN")
        rows = []
        for samples in (64, 256, 1024):
            vectors, seconds, actual_samples = [], [], []
            for seed in (711, 812, 913):
                budget = ComputeBudget(iterations=32, samples=samples, max_depth=1, max_nodes=3000000, seed=seed)
                start = time.process_time()
                result = solve(root, ranges, budget, profile)
                seconds.append(time.process_time() - start)
                actual_samples.append(result.work.samples)
                vectors.append([v / root.blinds.big for h, ev in sorted(result.action_ev_by_hand.items()) for a, v in sorted(ev.items())])
            dispersion = (statistics.mean(statistics.variance(values) for values in zip(*vectors))) ** .5
            rows.append({"samples": samples, "seeds": [711, 812, 913], "cross_seed_rms_sd_bb": dispersion,
                         "cpu_seconds": seconds, "actual_worlds": actual_samples, "action_values_bb": vectors})
        report["tracks"][str(count)] = {"game": name, "budgets": rows}
    atomic_bytes(output / "label-stability.json", json.dumps(report, indent=2).encode())
    return report
