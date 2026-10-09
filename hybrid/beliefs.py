"""Bayesian public reach factors and a separate private conditioning view."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

from hybrid.ranges import Combo, HandRange, JointRanges
import math


class IncompatibleObservation(ValueError):
    """The stated behavior model assigns no feasible mass to an observed action."""


@dataclass(frozen=True)
class BeliefState:
    public: JointRanges
    provenance: tuple[str, ...] = ("explicit prior",)
    upstream_counterfactual_values: Mapping[int, Mapping[Combo, float]] | None = None

    def update(self, player: int, likelihood: Callable[[Combo], float], *, model_version: str) -> BeliefState:
        if player not in self.public.ranges:
            raise ValueError(f"Unseated observed player: {player}")
        weights = {}
        for hand, prior in self.public.ranges[player].weights.items():
            value = likelihood(hand)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Invalid action likelihood for {hand}: {value}")
            weights[hand] = prior * value
        if not any(weights.values()):
            raise IncompatibleObservation(f"Zero-likelihood observation for player={player}, model={model_version}")
        factors = dict(self.public.ranges)
        factors[player] = HandRange(weights)
        posterior = JointRanges(factors, self.public.board)
        try:
            posterior.assert_compatible()
        except ValueError as error:
            raise IncompatibleObservation(f"Action leaves no compatible joint support: {error}") from error
        return BeliefState(posterior,
                           (*self.provenance, f"action likelihood: player={player}, model={model_version}"),
                           self.upstream_counterfactual_values)

    def reveal(self, board: tuple[int, ...]) -> BeliefState:
        if board[:len(self.public.board)] != self.public.board:
            raise ValueError("A board reveal must extend the existing ordered public board")
        return BeliefState(JointRanges(self.public.ranges, board), (*self.provenance, "public board reveal"),
                           self.upstream_counterfactual_values)

    def private_view(self, hero: int, cards: Combo) -> JointRanges:
        return self.public.conditioned(hero, cards)
