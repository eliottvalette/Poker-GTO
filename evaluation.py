"""Bounded exact policy values and infoset-consistent best responses on supplied deals.

This measures only the specified finite conditional subgame, not full-game exploitability.
Opponent hidden deals must be represented by weighted roots, never chosen per infoset.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from cfr_solver import GameState, Strategy, Traversal, TraversalBudgetExceeded, validate_strategy
from actions import ACTION_IDS, legal_actions
from poker_game_expresso import HandState
from infoset import observe


@dataclass
class Node:
    state: GameState
    external_reach: float
    depth: int
    probability: float
    children: dict[str, Node] = field(default_factory=dict)
    strategy: tuple[float, ...] = ()
    key: str | None = None
    value: float = 0.0


def subgame_best_response(roots: list[tuple[float, GameState]], strategy: Strategy,
                          hero: int, max_nodes: int = 10000) -> dict[str, float]:
    if not roots or abs(sum(p for p, _ in roots) - 1) > 1e-8 or any(p <= 0 for p, _ in roots):
        raise ValueError("Best-response roots need positive probabilities summing to one")
    count = 0
    infosets: dict[str, list[Node]] = {}
    levels: dict[int, list[Node]] = {}
    def build(state, reach, depth, probability):
        nonlocal count
        count += 1
        if count > max_nodes:
            raise TraversalBudgetExceeded(f"Best-response enumeration exceeded {max_nodes} nodes")
        if not isinstance(state, HandState):
            raise ValueError("Exact subgame evaluation currently requires hand states with fixed deals")
        node = Node(state, reach, depth, probability)
        levels.setdefault(depth, []).append(node)
        if state.terminal:
            node.value = state.utility(hero)
            return node
        obs = observe(state)
        node.strategy = strategy(obs)
        validate_strategy(node.strategy, obs.legal_mask)
        if state.current_player == hero:
            node.key = obs.key()
            infosets.setdefault(node.key, []).append(node)
        for action in legal_actions(state):
            p = node.strategy[ACTION_IDS.index(action.action_id)]
            next_reach = reach if state.current_player == hero else reach * p
            node.children[action.action_id] = build(Traversal.child(state, action), next_reach, depth + 1, p)
        return node
    trees = [(p, build(state, p, 0, p)) for p, state in roots]
    def policy_value(node):
        if not node.children:
            return node.value
        return sum(child.probability * policy_value(child) for child in node.children.values())
    base_value = sum(p * policy_value(n) for p, n in trees)
    choices = {}
    for depth in sorted(levels, reverse=True):
        for node in levels[depth]:
            if not node.children:
                continue
            if node.key is None:
                node.value = sum(c.probability * c.value for c in node.children.values())
            else:
                if node.key not in choices:
                    group = infosets[node.key]
                    if any(n.depth != depth for n in group):
                        raise ValueError("Perfect-recall infoset occurs at inconsistent depths")
                    choices[node.key] = max(node.children, key=lambda a: sum(n.external_reach * n.children[a].value for n in group))
                node.value = node.children[choices[node.key]].value
    best = sum(p * n.value for p, n in trees)
    return {"policy_value_bb": base_value, "best_response_value_bb": best,
            "best_response_gain_bb": best - base_value, "nodes": float(count)}
