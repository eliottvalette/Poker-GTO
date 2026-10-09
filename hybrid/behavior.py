"""Observable action likelihoods, including explicitly bracketed off-tree raises."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from infoset import Observation
from actions import ACTION_IDS, SolverAction, legal_actions
from hybrid.beliefs import BeliefState
from hybrid.policy_source import PolicySource, distribution
from hybrid.ranges import Combo, combo
from poker_game_expresso import HandState


@dataclass(frozen=True)
class ActionLikelihood:
    probability: float
    method: str


@dataclass(frozen=True)
class BehaviorModel:
    """Explicitly distinguish strategic reference from population-specific estimation."""

    policy: PolicySource
    role: Literal["reference", "opponent_estimate"]
    population: str

    def __post_init__(self) -> None:
        if self.role not in ("reference", "opponent_estimate") or not self.population:
            raise ValueError("Behavior requires a stated reference/estimation role and population")

    @property
    def version(self) -> str:
        return f"{self.role}/{self.population}/{self.policy.version}"

    def probabilities(self, observation: Observation) -> tuple[float, ...]:
        return self.policy.probabilities(observation)


def likelihood(policy: PolicySource, public: HandState, hand: Combo,
               observed: SolverAction, *, interpolate: bool = False) -> ActionLikelihood:
    hand = combo(hand)
    if set(hand).intersection(public.board):
        return ActionLikelihood(0.0, "blocked")
    # Observation encoding does not inspect other private cards or the deck.
    view = public.clone()
    view.actor.cards = hand
    probabilities = distribution(policy, view)
    actions = legal_actions(view)
    for action in actions:
        if action.category == observed.category and action.amount_to == observed.amount_to:
            return ActionLikelihood(probabilities[ACTION_IDS.index(action.action_id)], "canonical-action")
    if observed.category != "RAISE" or observed.amount_to is None:
        raise ValueError(f"Unsupported observed action: {observed}")
    if not interpolate:
        raise ValueError("Off-tree raise requires explicit interpolate=True behavior assumption")
    raises = sorted((a.amount_to, probabilities[ACTION_IDS.index(a.action_id)])
                    for a in actions if a.category == "RAISE")
    for (low, lp), (high, hp) in zip(raises, raises[1:]):
        if low < observed.amount_to < high:
            fraction = (observed.amount_to - low) / (high - low)
            return ActionLikelihood((1 - fraction) * lp + fraction * hp,
                                    "linear-likelihood-kernel-between-canonical-raise-amounts")
    raise ValueError(f"Unsupported off-tree extrapolation: amount={observed.amount_to}, brackets={raises}")


def observe_action(public: HandState, beliefs: BeliefState, observed: SolverAction,
                   policy: PolicySource, *, interpolate: bool = False) -> tuple[HandState, BeliefState]:
    if tuple(public.board) != beliefs.public.board or set(public.players) != set(beliefs.public.ranges):
        raise ValueError("Observed action and belief public state mismatch")
    actor = public.current_player
    child = public.clone()
    observed.apply(child)
    updated = beliefs.update(actor, lambda h: likelihood(policy, public, h, observed,
                                                        interpolate=interpolate).probability,
                             model_version=policy.version + ("/explicit-linear-sizing-kernel" if interpolate else ""))
    if child.board != public.board:
        updated = updated.reveal(tuple(child.board))
    return child, updated
