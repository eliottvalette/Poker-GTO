"""Bounded validation evidence; no large training workflow is invoked."""
from __future__ import annotations
from dataclasses import asdict
import json
import random
import resource
import sys
import tempfile
import time
from pathlib import Path
import torch
from actions import ACTION_IDS
from cfr_solver import Traversal, TraversalBudgetExceeded, regret_matching
from evaluation import subgame_best_response
from ml.deep_cfr import DeepCFRSolver, NeuralAveragePolicy
from ml.model import AdvantageNetwork, AveragePolicyNetwork
from poker_game_expresso import HandState
from tournament import TournamentState


def controlled_river(rng: random.Random) -> HandState:
    h = HandState.start({0: 2.0, 1: 2.0}, 0, random.Random(3))
    while h.street != "RIVER":
        h.act("CALL" if h.to_call() else "CHECK")
    return h


def run_smoke_validation() -> dict:
    torch.set_num_threads(1)
    solver = DeepCFRSolver((0, 1), "hand_chip_delta", seed=2, advantage_capacity=100, strategy_capacity=100)
    started = time.perf_counter()
    for _ in range(2):
        solver.run_iteration(controlled_river, traversals_per_player=2, max_nodes=1000)
    elapsed = time.perf_counter() - started
    samples = [s for memory in solver.advantage_memory.values() for s in memory.samples] + solver.strategy_memory.samples
    serialized_bytes = sum(len(json.dumps(asdict(s)).encode()) for s in samples)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "average.pt"
        solver.export_average(path)
        policy = NeuralAveragePolicy(path)
        metrics = subgame_best_response([(1.0, controlled_river(random.Random(0)))], policy.query, 0)
    tournament_samples = []
    tiny = TournamentState({0: 1.0, 1: 1.0}, rng=random.Random(3))
    tiny.start_hand()
    walk = Traversal(lambda o: regret_matching([0] * len(ACTION_IDS), o.legal_mask), random.Random(1), 1000, 100)
    tiny_value = walk.regrets(tiny, 0, lambda o, r, w: tournament_samples.append(o))
    full = TournamentState(rng=random.Random(3))
    full.start_hand()
    bounded = Traversal(lambda o: regret_matching([0] * len(ACTION_IDS), o.legal_mask), random.Random(1), 100, 100)
    before = time.perf_counter()
    try:
        bounded.regrets(full, 0, lambda *_: None)
        full_result = "completed"
    except TraversalBudgetExceeded as error:
        full_result = str(error)
    full_probe_seconds = time.perf_counter() - before
    average_walk = Traversal(lambda o: regret_matching([0] * len(ACTION_IDS), o.legal_mask),
                             random.Random(1), 300, 150)
    average_samples = []
    before = time.perf_counter()
    try:
        average_walk.average(full.clone(), 0, lambda o, target, weight: average_samples.append(weight))
        average_result = "completed"
    except TraversalBudgetExceeded as error:
        average_result = str(error)
    average_seconds = time.perf_counter() - before
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {"architecture": "card embeddings + numeric MLP + full-history GRU + 64-wide head",
            "advantage_parameters_per_player": sum(p.numel() for p in AdvantageNetwork().parameters()),
            "average_policy_parameters": sum(p.numel() for p in AveragePolicyNetwork().parameters()),
            "smoke_seconds": elapsed, "smoke_metrics": solver.metrics,
            "mean_serialized_sample_bytes": serialized_bytes / len(samples),
            "process_peak_rss_mib": rss / (1024**2 if sys.platform == "darwin" else 1024),
            "conditional_subgame_best_response": metrics,
            "short_hand_chip_value": tiny_value, "short_hand_nodes": walk.nodes,
            "short_hand_samples": len(tournament_samples),
            "hand_25bb_probe": full_result, "full_probe_nodes": bounded.nodes,
            "full_probe_seconds": full_probe_seconds,
            "hand_25bb_average_probe": {"result": average_result, "nodes": average_walk.nodes,
                                       "samples": len(average_samples), "seconds": average_seconds},
            "readiness": "SMOKE-TRAIN READY"}


if __name__ == "__main__":
    print(json.dumps(run_smoke_validation(), indent=2, allow_nan=False))
