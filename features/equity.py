"""Showdown equity, distinct from action EV; enumeration or explicitly seeded MC."""
from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations, product
import math
import random
from typing import Mapping
from features.range_features import normalize_range
from utils import rank7


@dataclass(frozen=True)
class EquityResult:
    equity: float
    exact: bool
    evaluations: int
    sampling_count: int
    standard_error: float | None


def equity(hero: tuple[int, int], opponents: tuple[Mapping[tuple[int, int], float], ...],
           board: tuple[int, ...] = (), *, max_exact: int = 100000, samples: int = 10000,
           seed: int = 0) -> EquityResult:
    known = (*hero, *board)
    if len(hero) != 2 or len(board) not in (0, 3, 4, 5) or len(set(known)) != len(known) or any(c not in range(52) for c in known):
        raise ValueError(f"Invalid equity cards: hero={hero}, board={board}")
    if len(opponents) not in (1, 2) or max_exact < 1 or samples < 2:
        raise ValueError(f"Invalid equity budget/opponents: {len(opponents)}, {max_exact}, {samples}")
    ranges = [normalize_range(r, known) for r in opponents]
    # Joint independent priors are conditioned on mutually disjoint hands.
    combo_count = math.prod(len(r) for r in ranges)
    runouts = math.comb(52 - len(known) - 2 * len(ranges), 5 - len(board))
    rng = random.Random(seed)

    def payoff(hands: tuple[tuple[int, int], ...], runout: tuple[int, ...]) -> float:
        public = (*board, *runout)
        ranks = [rank7((*h, *public)) for h in (hero, *hands)]
        best = max(ranks)
        return 1 / ranks.count(best) if ranks[0] == best else 0.0

    if combo_count * runouts <= max_exact:
        numerator = mass = 0.0
        evaluations = 0
        for rows in product(*(r.items() for r in ranges)):
            hands = tuple(h for h, _ in rows)
            dead = (*known, *(c for h in hands for c in h))
            if len(set(dead)) != len(dead):
                continue
            weight = math.prod(w for _, w in rows)
            deck = [c for c in range(52) if c not in dead]
            value = 0.0
            for runout in combinations(deck, 5 - len(board)):
                value += payoff(hands, runout)
                evaluations += 1
            numerator += weight * value / runouts
            mass += weight
        if mass <= 0:
            raise ValueError("Opponent ranges have no mutually compatible joint hands")
        return EquityResult(numerator / mass, True, evaluations, 0, None)
    choices = [list(r) for r in ranges]
    weights = [list(r.values()) for r in ranges]
    values = []
    attempts = 0
    while len(values) < samples:
        attempts += 1
        if attempts > samples * 1000:
            raise RuntimeError(f"Joint range rejection budget exceeded: accepted={len(values)}, attempts={attempts}")
        hands = tuple(rng.choices(h, weights=w, k=1)[0] for h, w in zip(choices, weights))
        dead = (*known, *(c for h in hands for c in h))
        if len(set(dead)) != len(dead):
            continue
        runout = tuple(rng.sample([c for c in range(52) if c not in dead], 5 - len(board)))
        values.append(payoff(hands, runout))
    mean = math.fsum(values) / samples
    error = math.sqrt(math.fsum((v - mean) ** 2 for v in values) / (samples - 1) / samples)
    return EquityResult(mean, False, samples, samples, error)


def hand_equity(hero: tuple[int, int], opponent: tuple[int, int], board: tuple[int, ...] = (), **budgets: int) -> EquityResult:
    return equity(hero, ({opponent: 1.0},), board, **budgets)
