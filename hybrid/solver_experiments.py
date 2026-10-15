"""Bounded, reproducible solver-first curriculum and optional student comparison."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import random
import resource
import sys
import time

import torch

from actions import ACTION_IDS
from evaluation import subgame_best_response
from hybrid.distillation import SolverStudents
from hybrid.knowledge import calculate, game_contract
from hybrid.solver_inputs import RootPriorPolicy
from hybrid.policy_source import UniformLegalPolicy
from hybrid.ranges import HandRange, JointRanges, combo
from hybrid.river_solver import RiverSolver
from hybrid.state import ComputeBudget, instantiate
from poker_game_expresso import HandState
from infoset import observe
from training.checkpoint import atomic_bytes


def bounded_roots(count: int, split: str):
    """Deliberately shallow independent roots; not representative full NLHE data."""
    offset = {"train": 0, "validation": 1, "test": 2}[split]
    for index, street in enumerate(("PREFLOP", "FLOP", "TURN", "RIVER")):
        rng = random.Random(81000 + count * 100 + offset * 10 + index)
        root = HandState.start({p: 1.5 + .1 * offset for p in range(count)}, offset % count, rng)
        while root.street != street:
            root.act("CALL" if root.to_call() else "CHECK")
        available = [c for c in range(52) if c not in root.board]
        ranges = JointRanges({p: HandRange({combo(player.cards): 1,
            combo(tuple(rng.sample(available, 2))): 1}) for p, player in root.players.items()}, tuple(root.board))
        yield root, ranges


def run(output: Path, *, iterations: int = 32, updates: int = 64) -> dict:
    """Fresh comparison; intentionally refuse to mix runs in an existing directory."""
    if output.exists():
        raise ValueError(f"Experiment output already exists: {output}")
    output.mkdir(parents=True)
    budget = ComputeBudget(max_nodes=300000, iterations=iterations, samples=32, max_depth=1, seed=818)
    report = {"budget": asdict(budget), "updates_per_specialist": updates,
              "scope": "24 shallow roots; exact river and fixed-forest other streets; independent splits; not poker-strength certification",
              "tracks": {}}
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    started = time.perf_counter()
    try:
        for count in (2, 3):
            paths, generation = [], []
            for split in ("train", "validation", "test"):
                for root, ranges in bounded_roots(count, split):
                    path = output / f"{count}-{split}-{root.street}.json"
                    begin = time.perf_counter()
                    result = calculate(path, root, ranges, budget, split=split)
                    paths.append(path)
                    generation.append({"split": split, "street": root.street, "seconds": time.perf_counter() - begin,
                        "infosets": len(result.policy.table), "nodes": result.work.nodes,
                        "iterations": result.work.iterations, "bytes": path.stat().st_size})
                    print(f"solver {count}p {split} {root.street}: {result.work.nodes} nodes", flush=True)
            variants = {}
            for residual in (False, True):
                name = "residual" if residual else "relu_baseline"
                learner = SolverStudents(count, residual=residual, seed=715)
                begin = time.perf_counter()
                metrics = learner.fit(paths, updates=updates)
                checkpoint = output / f"{count}-{name}.pt"
                learner.save(checkpoint)
                fit_seconds = time.perf_counter() - begin
                root, ranges = list(bounded_roots(count, "test"))[-1]
                worlds = [(d.probability, instantiate(root, d.hands)) for d in ranges.enumerate()]
                computed = RiverSolver().solve(root, ranges, ComputeBudget(iterations=200, max_nodes=1000000))
                gains = {}
                for kind in ("advantage", "strategy"):
                    policy = RootPriorPolicy(learner.models[kind], game_contract(root, ranges, budget, "test"), UniformLegalPolicy())
                    gains[kind + "_root_uniform_downstream"] = sum(subgame_best_response(worlds, policy.probabilities, p)["best_response_gain_bb"] for p in root.players)
                gains["computed_no_model"] = sum(subgame_best_response(worlds, computed.policy.probabilities, p)["best_response_gain_bb"] for p in root.players)
                gains["uniform"] = sum(subgame_best_response(worlds, UniformLegalPolicy().probabilities, p)["best_response_gain_bb"] for p in root.players)
                assistance = []
                prior = RootPriorPolicy(learner.models["strategy"], game_contract(root, ranges, budget, "test"), UniformLegalPolicy())
                for limit in (10, 50):
                    for use_prior in (False, True):
                        begin = time.perf_counter()
                        candidate = RiverSolver(prior if use_prior else None).solve(
                            root, ranges, ComputeBudget(iterations=limit, max_nodes=1000000))
                        seconds = time.perf_counter() - begin
                        gain = sum(subgame_best_response(worlds, candidate.policy.probabilities, p)["best_response_gain_bb"] for p in root.players)
                        assistance.append({"iterations": limit, "neural_prior": use_prior, "seconds": seconds,
                                           "nodes": candidate.work.nodes, "response_gain_sum_bb": gain})
                street_quality = []
                for held_root, held_ranges in bounded_roots(count, "test"):
                    high = ComputeBudget(iterations=128, samples=128, max_depth=1, max_nodes=1000000, seed=819)
                    reference = (RiverSolver().solve(held_root, held_ranges, high) if held_root.street == "RIVER" else
                                 RiverSolver().solve_chance(held_root, held_ranges, high))
                    low = (RiverSolver().solve(held_root, held_ranges, budget) if held_root.street == "RIVER" else
                           RiverSolver().solve_chance(held_root, held_ranges, budget))
                    actor = held_root.current_player
                    contexts = []
                    used = set()
                    for deal in held_ranges.enumerate():
                        hand = deal.hands[actor]
                        if hand in used:
                            continue
                        used.add(hand)
                        obs = observe(instantiate(held_root, deal.hands))
                        ev = reference.action_ev_by_hand[hand]
                        contract = game_contract(held_root, held_ranges, budget, "test")
                        a = learner.models["advantage"].probabilities(obs, contract)
                        b = learner.models["strategy"].probabilities(obs, contract)
                        policies = {"computed_no_model": low.strategy_by_hand[hand],
                                    "advantage": dict(zip(ACTION_IDS, a)), "strategy": dict(zip(ACTION_IDS, b))}
                        contexts.append({"holding": hand, "reference_ev_bb": ev,
                            "root_response_shortfall_bb": {label: max(ev.values()) - sum(p.get(action, 0) * value for action, value in ev.items())
                                                            for label, p in policies.items()}})
                    street_quality.append({"street": held_root.street, "holdings": contexts,
                        "scope": "one-step deviation under independently seeded stronger finite-forest continuation; not full-game exploitability"})
                variants[name] = {"fit_seconds": fit_seconds, "checkpoint_bytes": checkpoint.stat().st_size,
                                  "specialists": metrics, "held_out_river_response_gain_sum_bb": gains,
                                  "matched_node_budget_prior_ablation": assistance, "held_out_street_decisions": street_quality,
                                  "accepted": False}
                print(f"students {count}p {name}: {fit_seconds:.2f}s, response gains {gains}", flush=True)
            report["tracks"][str(count)] = {"generation": generation, "variants": variants}
        report["seconds"] = time.perf_counter() - started
        report["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        atomic_bytes(output / "report.json", json.dumps(report, indent=2, allow_nan=False).encode())
        return report
    finally:
        torch.set_num_threads(old_threads)
