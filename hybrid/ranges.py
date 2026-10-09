"""Exact private-hand factors conditioned on joint card compatibility."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import accumulate, combinations, product
import math
import random
from types import MappingProxyType
from typing import Mapping

from features.range_features import normalize_range

Combo = tuple[int, int]


def combo(cards: tuple[int, ...]) -> Combo:
    if len(cards) != 2 or len(set(cards)) != 2 or any(type(c) is not int or not 0 <= c < 52 for c in cards):
        raise ValueError(f"Invalid private combination: {cards}")
    return tuple(sorted(cards))


def validate_board(board: tuple[int, ...]) -> None:
    if len(board) not in (0, 3, 4, 5) or len(set(board)) != len(board) or any(type(c) is not int or not 0 <= c < 52 for c in board):
        raise ValueError(f"Invalid public board: {board}")


@dataclass(frozen=True)
class HandRange:
    """A normalized factor, not a multi-player compatible marginal."""

    weights: Mapping[Combo, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "weights", MappingProxyType(normalize_range(self.weights)))

    @classmethod
    def uniform(cls, dead: tuple[int, ...] = ()) -> HandRange:
        if len(set(dead)) != len(dead) or any(type(c) is not int or not 0 <= c < 52 for c in dead):
            raise ValueError(f"Invalid dead cards: {dead}")
        return cls({h: 1.0 for h in combinations(range(52), 2) if not set(h).intersection(dead)})

    def exclude(self, dead: tuple[int, ...]) -> HandRange:
        return HandRange(normalize_range(self.weights, dead))

    def probability(self, cards: Combo) -> float:
        return self.weights.get(combo(cards), 0.0)


@dataclass(frozen=True)
class JointDeal:
    hands: Mapping[int, Combo]
    probability: float


class JointRanges:
    """Product factors conditioned on disjoint cards; folds retain their factor.

    Updating one factor by a private-hand/public-history action likelihood is
    exact within this representation. Marginals are derived, never fed back as
    independent factors. Rejection sampling uses the entire product proposal;
    sequentially renormalizing each player's remaining hands would be biased.
    """

    def __init__(self, ranges: Mapping[int, HandRange], board: tuple[int, ...] = ()) -> None:
        validate_board(board)
        if len(ranges) not in (2, 3) or any(type(p) is not int for p in ranges):
            raise ValueError("Joint ranges require two or three explicitly identified seats")
        self.board = board
        self.ranges = MappingProxyType({p: r.exclude(board) for p, r in ranges.items()})
        self._proposals = tuple((p, tuple(r.weights), tuple(accumulate(r.weights.values())))
                                for p, r in self.ranges.items())

    def conditioned(self, player: int, cards: Combo) -> JointRanges:
        cards = combo(cards)
        if player not in self.ranges or self.ranges[player].probability(cards) <= 0:
            raise ValueError(f"Private query outside public support: player={player}, cards={cards}")
        return JointRanges({p: HandRange({cards: 1.0}) if p == player else r.exclude(cards)
                            for p, r in self.ranges.items()}, self.board)

    @property
    def candidate_count(self) -> int:
        return math.prod(len(r.weights) for r in self.ranges.values())

    def enumerate(self, max_candidates: int = 100000) -> tuple[JointDeal, ...]:
        if max_candidates < 1 or self.candidate_count > max_candidates:
            raise ValueError(f"Joint enumeration needs {self.candidate_count} candidates; limit={max_candidates}")
        seats = tuple(self.ranges)
        rows = []
        for entries in product(*(r.weights.items() for r in self.ranges.values())):
            cards = [c for h, _ in entries for c in h]
            if len(set(cards)) == len(cards):
                rows.append((dict(zip(seats, (h for h, _ in entries))), math.prod(w for _, w in entries)))
        total = math.fsum(w for _, w in rows)
        if total <= 0:
            raise ValueError("Ranges have no compatible joint private deal")
        return tuple(JointDeal(MappingProxyType(h), w / total) for h, w in rows)

    def sample(self, rng: random.Random, max_attempts: int = 10000) -> Mapping[int, Combo]:
        if max_attempts < 1:
            raise ValueError("Joint rejection budget must be positive")
        for _ in range(max_attempts):
            hands = {p: rng.choices(h, cum_weights=w, k=1)[0] for p, h, w in self._proposals}
            cards = [c for h in hands.values() for c in h]
            if len(set(cards)) == len(cards):
                return hands
        raise RuntimeError(f"No compatible deal in {max_attempts} product proposals; inspect range overlap")

    def assert_compatible(self, max_candidates: int = 100000) -> None:
        """Find a witness without normalizing or enumerating a large joint table."""
        factors = sorted(self.ranges.values(), key=lambda r: len(r.weights))
        attempts = 0

        def witness(index: int, used: frozenset[int]) -> bool:
            nonlocal attempts
            if index == len(factors):
                return True
            for hand in factors[index].weights:
                attempts += 1
                if attempts > max_candidates:
                    raise RuntimeError(f"Joint compatibility witness budget exceeded: {max_candidates}")
                if not used.intersection(hand) and witness(index + 1, used.union(hand)):
                    return True
            return False

        if not witness(0, frozenset(self.board)):
            raise ValueError("Ranges have no compatible joint private deal")

    def marginals(self, max_candidates: int = 100000) -> dict[int, HandRange]:
        result: dict[int, dict[Combo, float]] = {p: {} for p in self.ranges}
        for deal in self.enumerate(max_candidates):
            for p, h in deal.hands.items():
                result[p][h] = result[p].get(h, 0.0) + deal.probability
        return {p: HandRange(w) for p, w in result.items()}
