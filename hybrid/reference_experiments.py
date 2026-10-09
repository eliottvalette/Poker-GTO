"""Exact finite-game quality and model-prior ablations with independent roots."""

import json
from pathlib import Path
import random
import time

from evaluation import subgame_best_response
from hybrid.experiments import small_game
from hybrid.policy_source import NeuralAveragePolicy, UniformLegalPolicy
from hybrid.reference import MixturePolicy, ProfilePolicy, ReferencePolicy, PROFILES
from hybrid.ranges import HandRange, JointRanges, combo
from hybrid.river_solver import RiverSolver
from hybrid.state import ComputeBudget, instantiate
from ml.deep_cfr import NeuralAveragePolicy as PublishedPolicy
from training.workflow import active_checkpoint, digest


def reference_ablation(directory: Path, *, roots_per_count: int = 4) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    pool = {name: ProfilePolicy(name) for name in PROFILES}
    pool["uniform"] = UniformLegalPolicy()
    components = (pool["tight_passive"], pool["loose_aggressive"])
    pool["mixed"] = MixturePolicy(components, (0.5, 0.5))
    pool["adaptive"] = MixturePolicy(components, (0.5, 0.5), adaptive=True)
    hash_checks = {}
    for count, track in ((2, "hu"), (3, "3max")):
        checkpoint = active_checkpoint(track)
        if checkpoint is None:
            raise ValueError(f"No published policy for requested track {track}")
        path = checkpoint.parent / f"average_{track}.pt"
        before = digest(path)
        artifact = PublishedPolicy(path)
        neural = NeuralAveragePolicy(
            artifact.query, f"published-{track}-iteration{artifact.iteration}"
        )
        for seed in range(roots_per_count):
            root, _ = small_game(count, seed=4100 + seed)
            rng = random.Random(5100 + seed)
            cards = [c for c in range(52) if c not in root.board]
            rng.shuffle(cards)
            ranges = JointRanges(
                {
                    p: HandRange(
                        {
                            combo(tuple(cards[i * 4 : i * 4 + 2])): 1.0,
                            combo(tuple(cards[i * 4 + 2 : i * 4 + 4])): 2.0,
                        }
                    )
                    for i, p in enumerate(root.players)
                },
                tuple(root.board),
            )
            root = instantiate(root, ranges.enumerate()[0].hands)
            worlds = [
                (d.probability, instantiate(root, d.hands)) for d in ranges.enumerate()
            ]
            candidates = {"published-direct": neural}
            previous = None
            for iterations, prior, name in (
                (5, None, "previous-local"),
                (50, None, "local-50"),
                (50, neural, "neural-prior-50"),
                (500, None, "reference-500"),
            ):
                started = time.perf_counter()
                result = RiverSolver(prior).solve(
                    root,
                    ranges,
                    ComputeBudget(iterations=iterations, max_nodes=1000000),
                )
                responses = {
                    p: subgame_best_response(worlds, result.policy.probabilities, p)
                    for p in root.players
                }
                gain = sum(r["best_response_gain_bb"] for r in responses.values())
                stability = (
                    None
                    if previous is None
                    else sum(
                        sum(abs(v - previous[h][a]) for a, v in mix.items()) / 2
                        for h, mix in result.strategy_by_hand.items()
                    )
                    / len(result.strategy_by_hand)
                )
                rows.append(
                    {
                        "count": count,
                        "seed": seed,
                        "candidate": name,
                        "response_gain_sum": gain,
                        "nodes": result.work.nodes,
                        "seconds": time.perf_counter() - started,
                        "root_policy_tv_from_previous": stability,
                    }
                )
                candidates[name] = result.policy
                previous = result.strategy_by_hand
                if name == "reference-500" and gain <= 0.05:
                    policy = ReferencePolicy(
                        dict(result.policy.table),
                        {
                            "accepted": True,
                            "gain": gain,
                            "count": count,
                            "seed": seed,
                            "scope": "exact specified finite river ranges and action abstraction",
                            "iterations": iterations,
                        },
                    )
                    policy.save(directory / f"reference-{count}-{seed}.json")
            for opponent_name, opponent in {
                **pool,
                "published": neural,
                "previous_local": candidates["previous-local"],
            }.items():
                for name, candidate in candidates.items():
                    hero = root.current_player

                    def strategy(obs):
                        return (
                            candidate if obs.hero == hero else opponent
                        ).probabilities(obs)

                    evaluation = subgame_best_response(
                        worlds, strategy, hero, max_nodes=100000
                    )
                    rows.append(
                        {
                            "count": count,
                            "seed": seed,
                            "candidate": name,
                            "opponent": opponent_name,
                            "exact_chip_ev": evaluation["policy_value_bb"],
                            "bounded_response_gain": evaluation[
                                "best_response_gain_bb"
                            ],
                        }
                    )
        after = digest(path)
        if before != after:
            raise AssertionError("Published model changed during evaluation")
        hash_checks[track] = {"sha256": after, "unchanged": True}
    report = {
        "budget": {
            "roots_per_count": roots_per_count,
            "iterations": [5, 50, 500],
            "max_nodes": 1000000,
        },
        "rows": rows,
        "published_hash_checks": hash_checks,
        "scope": "exact finite-game expectations, no MC error; not unrestricted poker exploitability",
    }
    (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
