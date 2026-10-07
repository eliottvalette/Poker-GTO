"""Exact reach-weighted references and sampled collectors on fixed river trees.

The deals are fixed to isolate action sampling. These are conditional average
policies, not an exploitability estimate or a full-game convergence claim.
"""
from __future__ import annotations

import math
import random
from collections.abc import Callable

from actions import ACTION_IDS, legal_actions
from cfr_solver import Traversal, regret_matching
from infoset import Observation, observe
from poker_game_expresso import HandState

COLLECTORS = ("uniform_importance", "opponent_nodes")


def varying_policy(iteration: int) -> Callable[[Observation], tuple[float, ...]]:
    def policy(obs: Observation) -> tuple[float, ...]:
        exponent = (1.0, -1.0, 0.5)[iteration - 1]
        return regret_matching([(i + 1.0) ** exponent for i in range(len(ACTION_IDS))], obs.legal_mask)
    return policy


def small_root(count: int) -> HandState:
    root = HandState.start({i: 1.5 for i in range(count)}, 0, random.Random(3))
    while root.street != "RIVER":
        root.act("CALL" if root.to_call() else "CHECK")
    return root


def add_mass(rows: dict, key: str, strategy: tuple[float, ...], weight: float) -> None:
    row = rows.setdefault(key, {"mass": 0.0, "action_mass": [0.0] * len(ACTION_IDS),
                                "squared_weights": 0.0, "records": 0})
    row["mass"] += weight
    row["squared_weights"] += weight * weight
    row["records"] += 1
    for index, value in enumerate(strategy):
        row["action_mass"][index] += weight * value


def distribution(row: dict) -> list[float]:
    return [v / row["mass"] for v in row["action_mass"]]


def exact_reference(root: HandState) -> dict:
    """Enumerate all actions; calculate each collector's exact inclusion mass.

    Uniform paths contribute q * (own_reach/q). Opponent-node collection during
    traverser p contributes product(reach[j] for j != p). In HU this is exactly
    the recording player's own reach; in 3-max it also contains a third reach.
    """
    reference: dict = {}
    expected = {name: {} for name in COLLECTORS}
    node_counts = []
    for iteration in (1, 2, 3):
        policy = varying_policy(iteration)
        nodes = 0

        def visit(state: HandState, reach: dict[int, float], uniform_reach: float) -> None:
            nonlocal nodes
            nodes += 1
            if nodes > 10000:
                raise RuntimeError("Exact collector reference exceeded 10000 nodes")
            if state.terminal:
                return
            obs = observe(state)
            key, actor = obs.key(), obs.hero
            strategy = policy(obs)
            add_mass(reference, key, strategy, iteration * reach[actor])
            add_mass(expected["uniform_importance"], key, strategy,
                     iteration * uniform_reach * (reach[actor] / uniform_reach))
            inclusion = sum(math.prod(reach[j] for j in reach if j != p) for p in reach if p != actor)
            add_mass(expected["opponent_nodes"], key, strategy, iteration * inclusion)
            actions = legal_actions(state)
            for action in actions:
                updated = dict(reach)
                updated[actor] *= strategy[ACTION_IDS.index(action.action_id)]
                visit(Traversal.child(state, action), updated, uniform_reach / len(actions))

        visit(root, dict.fromkeys(root.players, 1.0), 1.0)
        node_counts.append(nodes)
    return {"reference": reference, "expected": expected, "nodes_per_iteration": node_counts}


def audit_collectors(trials: int = 256) -> dict:
    if type(trials) is not int or trials < 1:
        raise ValueError(f"Positive explicit trial count required: {trials}")
    report = {"iterations": [1, 2, 3], "iteration_weighting": "linear", "trials_per_player_iteration": trials,
              "scope": "fixed-deal river trees; exact inclusion expectations plus production traversal samples",
              "ess_scope": "weight concentration only; correlated records are not independent observations", "tracks": {}}
    for count, name in ((2, "hu"), (3, "3max")):
        root = small_root(count)
        exact = exact_reference(root)
        reference = exact["reference"]
        result = {"nodes_per_iteration": exact["nodes_per_iteration"], "infosets": len(reference), "collectors": {}}
        report["tracks"][name] = result
        for collector in COLLECTORS:
            sampled: dict = {}
            weights = []
            nodes = 0
            for iteration in (1, 2, 3):
                policy = varying_policy(iteration)
                for player in root.players:
                    for trial in range(trials):
                        walk = Traversal(policy, random.Random(iteration * 100000 + player * trials + trial), max_nodes=10000)

                        def sink(obs: Observation, target: tuple[float, ...], weight: float) -> None:
                            weighted = weight * iteration
                            add_mass(sampled, obs.key(), target, weighted)
                            weights.append(weighted)

                        if collector == "uniform_importance":
                            walk.average(root, player, sink)
                        else:
                            walk.regrets(root, player, lambda *_: None, strategy_sink=sink)
                        nodes += walk.nodes
            rows = []
            for key, exact_row in reference.items():
                truth = distribution(exact_row)
                expectation = distribution(exact["expected"][collector][key])
                empirical = distribution(sampled[key]) if key in sampled else None
                rows.append({"infoset": key, "exact_average": truth, "collector_expectation": expectation,
                             "absolute_bias": max(abs(a - b) for a, b in zip(truth, expectation)),
                             "sampled_average": empirical, "sampled_accumulation": sampled.get(key)})
            total = math.fsum(weights)
            result["collectors"][collector] = {
                "exact_max_absolute_bias": max(row["absolute_bias"] for row in rows),
                "records": len(weights), "nodes": nodes, "covered_infosets": len(sampled),
                "weight_ess": total * total / math.fsum(w * w for w in weights),
                "largest_weight_share": max(weights) / total,
                "top10_weight_share": math.fsum(sorted(weights, reverse=True)[:10]) / total,
                "rows": rows}
            print(f"{name}/{collector}: exact bias={result['collectors'][collector]['exact_max_absolute_bias']:.6g}, records={len(weights)}, coverage={len(sampled)}/{len(reference)}", flush=True)
    return report
