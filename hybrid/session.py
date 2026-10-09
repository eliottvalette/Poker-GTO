"""Atomic full-hand public beliefs, including explicit behavior uncertainty."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import random
from typing import Mapping

from actions import SolverAction
from hybrid.behavior import likelihood
from hybrid.beliefs import BeliefState, IncompatibleObservation
from hybrid.decision_engine import HybridDecisionEngine
from hybrid.policy_source import PolicySource
from hybrid.ranges import JointRanges
from hybrid.state import ComputeBudget
from poker_game_expresso import HandState


@dataclass(frozen=True)
class Hypothesis:
    name: str
    probability: float
    beliefs: BeliefState
    policies: Mapping[int, PolicySource]


class HybridSession:
    """Mixture of compatible product distributions, not a product of marginals.

    Behavior labels are latent for the entire hand. Keeping one range factorization
    per hypothesis preserves the induced hand/profile correlations exactly when
    evidence is enumerated. Broad evidence uses explicit seeded Monte Carlo.
    """

    def __init__(
        self,
        state: HandState,
        hypotheses: tuple[Hypothesis, ...],
        *,
        evidence_samples: int = 512,
        exact_candidates: int = 100000,
        seed: int = 0,
        reference_policy: PolicySource | None = None,
    ) -> None:
        if not hypotheses or len({h.name for h in hypotheses}) != len(hypotheses):
            raise ValueError("Nonempty distinct named behavior hypotheses required")
        if any(
            not math.isfinite(h.probability) or h.probability < 0 for h in hypotheses
        ):
            raise ValueError("Behavior probabilities must be finite and nonnegative")
        total = sum(h.probability for h in hypotheses)
        if total <= 0 or evidence_samples < 2 or exact_candidates < 1:
            raise ValueError("Positive hypothesis mass and evidence budgets required")
        for hypothesis in hypotheses:
            if set(hypothesis.policies) != set(state.players) or set(
                hypothesis.beliefs.public.ranges
            ) != set(state.players):
                raise ValueError(
                    "Session hypothesis seats must match the hand, including folded seats"
                )
            if hypothesis.beliefs.public.board != tuple(state.board):
                raise ValueError("Session board and belief board disagree")
            hypothesis.beliefs.public.assert_compatible()
        self.state = state.clone()
        self.hypotheses = tuple(
            replace(h, probability=h.probability / total)
            for h in hypotheses
            if h.probability
        )
        self.evidence_samples, self.exact_candidates, self.seed = (
            evidence_samples,
            exact_candidates,
            seed,
        )
        self.events: list[dict] = []
        from hybrid.reference import ReferencePolicy

        self.reference_policy = reference_policy or ReferencePolicy()

    def _expectation(self, joint: JointRanges, callback) -> tuple[float, str]:
        if joint.candidate_count <= self.exact_candidates:
            return math.fsum(
                d.probability * callback(d.hands)
                for d in joint.enumerate(self.exact_candidates)
            ), "exact"
        rng = random.Random(self.seed + len(self.events))
        values = [callback(joint.sample(rng)) for _ in range(self.evidence_samples)]
        return math.fsum(values) / len(values), f"sampled-evidence/{len(values)}"

    def observe_action(
        self,
        action: SolverAction,
        *,
        likelihood_source: str,
        external_policy: PolicySource | None = None,
        interpolate: bool = False,
    ) -> None:
        if not likelihood_source:
            raise ValueError(
                "Every action needs an explicit likelihood source, including human actions"
            )
        if self.state.terminal:
            raise ValueError(
                "Cannot act after settlement; start a new session for the next hand"
            )
        actor = self.state.current_player
        child = self.state.clone()
        action.apply(child)
        updated, methods = [], []
        for h in self.hypotheses:
            policy = external_policy or h.policies[actor]
            table = {
                cards: likelihood(
                    policy, self.state, cards, action, interpolate=interpolate
                ).probability
                for cards in h.beliefs.public.ranges[actor].weights
            }
            if len(self.hypotheses) > 1:
                new_cards = set(child.board) - set(self.state.board)
                evidence, method = self._expectation(
                    h.beliefs.public,
                    lambda deal: table[deal[actor]]
                    if not new_cards.intersection(
                        c for cards in deal.values() for c in cards
                    )
                    else 0.0,
                )
            else:
                evidence, method = 1.0, "single-hypothesis/Bayesian-factors"
            if evidence <= 0:
                if method.startswith("sampled-evidence"):
                    raise IncompatibleObservation(
                        "Sampled evidence was zero; precision is insufficient to eliminate this hypothesis. Increase evidence_samples; session unchanged."
                    )
                methods.append(f"{h.name}: zero evidence ({method})")
                continue
            try:
                beliefs = h.beliefs.update(
                    actor, table.__getitem__, model_version=policy.version
                )
                if child.board != self.state.board:
                    beliefs = beliefs.reveal(tuple(child.board))
                beliefs.public.assert_compatible()
            except (IncompatibleObservation, ValueError):
                if len(self.hypotheses) == 1:
                    raise
                methods.append(f"{h.name}: incompatible observation")
                continue
            updated.append(
                replace(h, probability=h.probability * evidence, beliefs=beliefs)
            )
            methods.append(f"{h.name}: {method}")
        mass = sum(h.probability for h in updated)
        if mass <= 0:
            raise IncompatibleObservation(
                "All behavior hypotheses reject this action/reveal; session unchanged"
            )
        self.state = child
        self.hypotheses = tuple(
            replace(h, probability=h.probability / mass) for h in updated
        )
        self.events.append(
            {
                "actor": actor,
                "category": action.category,
                "amount_to": action.amount_to,
                "source": likelihood_source,
                "external_policy": external_policy.version if external_policy else None,
                "evidence": methods,
                "board": tuple(child.board),
            }
        )

    def observe_board(self, board: tuple[int, ...]) -> None:
        """Condition on an engine-observed board; never advance betting by guessing."""
        if tuple(self.state.board) != board:
            raise ValueError(
                "Board must match the canonical engine state; action transitions reveal cards automatically"
            )
        self.hypotheses = tuple(
            replace(h, beliefs=h.beliefs.reveal(board)) for h in self.hypotheses
        )

    def next_hand(
        self, state: HandState, hypotheses: tuple[Hypothesis, ...]
    ) -> HybridSession:
        """New private priors and explicitly routed policies after canonical settlement."""
        if not self.state.terminal:
            raise ValueError("Cannot replace an unsettled hand")
        if state.hand_number <= self.state.hand_number:
            raise ValueError(
                "Next hand must have a strictly increasing canonical hand number"
            )
        return HybridSession(
            state,
            hypotheses,
            evidence_samples=self.evidence_samples,
            exact_candidates=self.exact_candidates,
            seed=self.seed + 1,
            reference_policy=self.reference_policy,
        )

    def analyze(
        self, budget: ComputeBudget, *, mode: str = "reference", cancelled=lambda: False
    ):
        if mode not in ("reference", "exploitative"):
            raise ValueError(f"Unknown decision mode: {mode}")
        if mode == "reference":
            if len(self.hypotheses) != 1:
                raise ValueError(
                    "Reference analysis needs one explicit reference range hypothesis; exploitative mixtures stay separate"
                )
            h = self.hypotheses[0]
            return HybridDecisionEngine(
                dict.fromkeys(self.state.players, self.reference_policy)
            ).analyze(self.state, h.beliefs, budget, cancelled=cancelled)
        hero, cards = self.state.current_player, tuple(sorted(self.state.actor.cards))
        conditioned = []
        for h in self.hypotheses:
            if len(self.hypotheses) == 1:
                evidence = 1.0
            elif h.beliefs.public.candidate_count <= self.exact_candidates:
                evidence, _ = self._expectation(
                    h.beliefs.public, lambda deal: float(deal[hero] == cards)
                )
            else:
                rng = random.Random(self.seed)
                factors = h.beliefs.public.ranges
                valid_public = valid_private = 0
                for _ in range(self.evidence_samples):
                    deal = {
                        p: rng.choices(
                            tuple(r.weights), weights=tuple(r.weights.values())
                        )[0]
                        for p, r in factors.items()
                    }
                    flat = [c for hand in deal.values() for c in hand]
                    valid_public += len(set(flat)) == len(flat)
                    deal[hero] = cards
                    flat = [c for hand in deal.values() for c in hand]
                    valid_private += len(set(flat)) == len(flat)
                if not valid_public or not valid_private:
                    raise ValueError(
                        "Private behavior evidence precision insufficient; increase evidence_samples"
                    )
                evidence = (
                    factors[hero].probability(cards) * valid_private / valid_public
                )
            if evidence:
                conditioned.append(replace(h, probability=h.probability * evidence))
        total = sum(h.probability for h in conditioned)
        if not total:
            raise IncompatibleObservation(
                "Actual Hero holding has zero mixture support"
            )
        conditioned = [
            replace(h, probability=h.probability / total) for h in conditioned
        ]
        count = len(conditioned)
        if min(budget.samples, budget.max_nodes) < count:
            raise ValueError(
                "Budget cannot allocate at least one unit to each behavior hypothesis"
            )
        per = replace(
            budget, samples=budget.samples // count, max_nodes=budget.max_nodes // count
        )
        results = [
            (
                h,
                HybridDecisionEngine(h.policies, search_mode="response").analyze(
                    self.state, h.beliefs, per, cancelled=cancelled
                ),
            )
            for h in conditioned
        ]
        result = replace(results[0][1], assumptions=list(results[0][1].assumptions))
        ev = {
            a: sum(h.probability * r.estimated_ev_by_action[a] for h, r in results)
            for a in result.estimated_ev_by_action
        }
        best = max(ev.values())
        ties = [
            a for a, value in ev.items() if math.isclose(value, best, abs_tol=1e-10)
        ]
        result.estimated_ev_by_action = ev
        result.action_probabilities = {a: float(a in ties) / len(ties) for a in ev}
        result.search_nodes = sum(r.search_nodes for _, r in results)
        result.samples_used = sum(r.samples_used for _, r in results)
        result.computation_breakdown = {key: sum(r.computation_breakdown.get(key, 0) for _, r in results)
                                        for key in set().union(*(r.computation_breakdown for _, r in results))}
        result.warnings = list(dict.fromkeys(warning for _, r in results for warning in r.warnings))
        result.uncertainty_estimates = {
            "behavior_ev_range": {
                a: [
                    min(r.estimated_ev_by_action[a] for _, r in results),
                    max(r.estimated_ev_by_action[a] for _, r in results),
                ]
                for a in ev
            },
            "conditional_sampling": {
                h.name: r.uncertainty_estimates for h, r in results
            },
            "behavior_posterior": {h.name: h.probability for h, _ in results},
            "private_hands": "compatible posterior integrated within each behavior hypothesis",
            "continuation_error": "not bounded; supplied behavior governs unresolved continuations",
        }
        result.assumptions.append(
            "Exploitative mode: latent behavior mixture preserved jointly with hand ranges."
        )
        result.assumptions.append(
            "Behavior mixture is privately conditioned on the acting hand for EV only; public hypotheses are unchanged."
        )

        def marginals(hypotheses, private):
            accumulated = {p: {} for p in self.state.players}
            for h in hypotheses:
                joint = (
                    h.beliefs.private_view(hero, cards) if private else h.beliefs.public
                )
                if joint.candidate_count <= self.exact_candidates:
                    deals = [
                        (d.hands, d.probability)
                        for d in joint.enumerate(self.exact_candidates)
                    ]
                else:
                    rng = random.Random(self.seed)
                    deals = [
                        (joint.sample(rng), 1 / self.evidence_samples)
                        for _ in range(self.evidence_samples)
                    ]
                for deal, probability in deals:
                    for p, holding in deal.items():
                        accumulated[p][holding] = (
                            accumulated[p].get(holding, 0.0)
                            + h.probability * probability
                        )
            return accumulated

        private_marginals = marginals(conditioned, True)
        result.opponent_ranges = {
            p: weights for p, weights in private_marginals.items() if p != hero
        }
        result.own_public_range = marginals(self.hypotheses, False)[hero]
        result.assumptions = [
            item
            for item in result.assumptions
            if not item.startswith("Displayed ranges are factors;")
        ]
        result.assumptions.append(
            f"Displayed session ranges are compatible mixture marginals; broad displays use {self.evidence_samples} seeded deals per hypothesis. They are never reused as independent factors."
        )
        result.baseline_probabilities = {
            a: sum(h.probability * r.baseline_probabilities[a] for h, r in results)
            for a in result.baseline_probabilities
        }
        result.assumptions.extend(f"{h.name}: {h.probability:.8g}" for h, _ in results)
        result.model_versions_used = {
            h.name: {p: policy.version for p, policy in h.policies.items()}
            for h, _ in results
        }
        return result
