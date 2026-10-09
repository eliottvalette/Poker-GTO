"""Explicit budgets and information-safe public state transport."""
from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Callable, Mapping

from poker_game_expresso import HandState
from hybrid.ranges import Combo, combo


class BudgetExceeded(RuntimeError):
    """Computation stopped without inventing an unresolved terminal payoff."""


@dataclass(frozen=True)
class ComputeBudget:
    max_nodes: int = 100000
    iterations: int = 100
    samples: int = 64
    max_depth: int = 4
    max_joint_candidates: int = 10000
    seed: int = 0
    strict: bool = True

    def __post_init__(self) -> None:
        if any(type(v) is not int or v < 1 for v in (self.max_nodes, self.iterations, self.samples,
                                                     self.max_depth, self.max_joint_candidates)):
            raise ValueError(f"Budget limits must be positive integers: {self}")
        if type(self.seed) is not int:
            raise ValueError("Seed must be an integer")
        if type(self.strict) is not bool:
            raise ValueError("Budget strict mode must be a boolean")


FAST = ComputeBudget(5000, 10, 16, 1, 1000)
NORMAL = ComputeBudget()
DEEP = ComputeBudget(1000000, 1000, 512, 6, 100000)


@dataclass
class WorkCounter:
    budget: ComputeBudget
    cancelled: Callable[[], bool] = lambda: False
    nodes: int = 0
    samples: int = 0
    iterations: int = 0
    terminal_leaves: int = 0
    rollout_leaves: int = 0
    learned_leaves: int = 0

    def visit(self) -> None:
        if self.cancelled():
            raise BudgetExceeded("Analysis cancelled")
        if self.nodes >= self.budget.max_nodes:
            raise BudgetExceeded(f"Analysis node limit reached: {self.nodes}")
        self.nodes += 1


def instantiate(public: HandState, hands: Mapping[int, Combo], rng: random.Random | None = None) -> HandState:
    """Discard all original hidden cards/deck before constructing a possible world."""
    if set(hands) != set(public.players):
        raise ValueError("Every seated player, including folded players, needs a private deal")
    result = public.clone()
    for p, h in hands.items():
        result.players[p].cards = combo(h)
    dead = [*public.board, *(c for h in hands.values() for c in h)]
    if len(set(dead)) != len(dead):
        raise ValueError("Private deal conflicts with public board or another hand")
    result.deck = [c for c in range(52) if c not in dead]
    if rng is not None:
        rng.shuffle(result.deck)
    result.assert_invariants()
    return result
