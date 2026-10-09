"""Range-conditioned holding values with independent exact-river label provenance."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
from typing import Mapping
import torch

from hybrid.learning import Learner, Supervision, observation_features
from hybrid.ranges import Combo, JointRanges
from hybrid.river_solver import RiverSolver
from hybrid.state import ComputeBudget
from infoset import observe
from poker_game_expresso import HandState


def range_features(public: HandState, ranges: JointRanges) -> tuple[float, ...]:
    """169-class factors in actor-relative position order; suit detail is lossy.

    This explicitly lossy approximation is only eligible within measured
    validation scope. Exact ranges remain the solver's authority.
    """
    positions = ["BTN", "SB", "BB"] if len(public.players) == 3 else ["SB", "BB"]
    offset = positions.index(public.actor.position)
    ordered = positions[offset:] + positions[:offset]
    features = []
    for position in ordered:
        player = next(p for p, value in public.players.items() if value.position == position)
        values = [0.0] * 169
        for (a, b), probability in ranges.ranges[player].weights.items():
            high, low = sorted((a // 4, b // 4), reverse=True)
            index = high * 13 + low if a % 4 == b % 4 or high == low else low * 13 + high
            values[index] += probability
        features.extend(values)
    return tuple(features + [0.0] * (169 * (3 - len(public.players))))


def continuation_features(public: HandState, ranges: JointRanges, holding: Combo) -> tuple[float, ...]:
    if set(public.players) != set(ranges.ranges) or tuple(public.board) != ranges.board:
        raise ValueError("Continuation range/state mismatch")
    if ranges.ranges[public.current_player].probability(holding) <= 0:
        raise ValueError("Continuation holding has no public prior support")
    state = public.clone()
    state.actor.cards = holding
    return (*observation_features(observe(state)), *range_features(state, ranges))


def exact_river_labels(public: HandState, ranges: JointRanges, budget: ComputeBudget,
                       split_group: str) -> list[Supervision]:
    solved = RiverSolver().solve(public, ranges, budget)
    records = []
    for holding, policy in solved.strategy_by_hand.items():
        value = sum(probability * solved.action_ev_by_hand[holding][action] for action, probability in policy.items())
        records.append(Supervision(continuation_features(public, ranges, holding), (value,), (),
                                   f"exact-terminal-river-CFR/{solved.complete_iterations}-iterations/finite-iteration-error-unbounded",
                                   len(public.players), {**asdict(budget), "street": public.street}, None, split_group))
    return records


class LearnedContinuation:
    def __init__(self, learner: Learner) -> None:
        if learner.kind != "continuation" or not learner.accepted:
            raise ValueError("Continuation model has not passed held-out value acceptance")
        self.learner = learner
        schemas = {r.compute_budget.get("feature_schema", "class169-v1") for r in learner.records}
        if len(schemas) != 1 or not schemas.issubset({"class169-v1", "combo1326-v2"}):
            raise ValueError(f"Unsupported continuation feature schema: {schemas}")
        self.feature_schema = next(iter(schemas))
        digest = hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes() for p in learner.model.parameters())).hexdigest()[:16]
        self.version = f"continuation/{self.feature_schema}/players={learner.records[0].player_count}/epoch={learner.epochs}/{digest}"

    def features(self, public, ranges, holding):
        if self.feature_schema == "combo1326-v2":
            from hybrid.value_features import suit_aware_features
            return suit_aware_features(public, ranges, holding)
        return continuation_features(public, ranges, holding)

    def value(self, public: HandState, ranges: JointRanges, holding: Combo) -> float:
        if len(public.players) != self.learner.records[0].player_count:
            raise ValueError("Continuation model player-count mismatch")
        streets = {r.compute_budget.get("street") for r in self.learner.records}
        if public.street not in streets:
            raise ValueError(f"Continuation model has no validation-domain labels for street={public.street}")
        with torch.no_grad():
            output = self.learner.model(torch.tensor([self.features(public, ranges, holding)], dtype=torch.float32))[0, 0]
        if not torch.isfinite(output):
            raise ValueError("Nonfinite learned continuation")
        value = float(output)
        initial = public.initial_stacks[public.current_player]
        if not -initial <= value <= public.total_chips - initial:
            raise ValueError(f"Learned continuation violates physical chip bounds: {value}")
        return value

    def values(self, public: HandState, ranges: JointRanges) -> Mapping[Combo, float]:
        if len(public.players) != self.learner.records[0].player_count:
            raise ValueError("Continuation model player-count mismatch")
        if public.street not in {r.compute_budget.get("street") for r in self.learner.records}:
            raise ValueError(f"Continuation model has no validation-domain labels for street={public.street}")
        holdings = tuple(ranges.ranges[public.current_player].weights)
        with torch.no_grad():
            result = self.learner.model(torch.tensor([self.features(public, ranges, h) for h in holdings], dtype=torch.float32))[:, 0]
        if not torch.isfinite(result).all():
            raise ValueError("Continuation model produced nonfinite values")
        initial = public.initial_stacks[public.current_player]
        if ((result < -initial) | (result > public.total_chips - initial)).any():
            raise ValueError("Learned continuation vector violates physical chip bounds")
        return dict(zip(holdings, result.tolist()))
