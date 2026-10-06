"""Explicit range validation; no inferred or fabricated ranges enter baseline training."""
from __future__ import annotations
import math
from typing import Mapping


def normalize_range(weights: Mapping[tuple[int, int], float], dead: tuple[int, ...] = ()) -> dict[tuple[int, int], float]:
    normalized: dict[tuple[int, int], float] = {}
    for hand, weight in weights.items():
        if len(hand) != 2 or len(set(hand)) != 2 or any(type(c) is not int or c not in range(52) for c in hand):
            raise ValueError(f"Invalid range hand: {hand}")
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"Invalid range weight: hand={hand}, weight={weight}")
        key = tuple(sorted(hand))
        if key in normalized:
            raise ValueError(f"Duplicate unordered range hand: {hand}")
        normalized[key] = 0.0 if set(hand).intersection(dead) else weight
    total = math.fsum(normalized.values())
    if not math.isfinite(total) or total <= 0:
        raise ValueError(f"Range has no positive unblocked mass: dead={dead}, total={total}")
    return {h: w / total for h, w in normalized.items() if w > 0}
