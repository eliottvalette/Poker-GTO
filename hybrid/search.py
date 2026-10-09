"""Paired, depth-limited action evaluation against explicit continuation policies.

This is a belief-conditioned response calculation, not safe re-solving. Hidden
worlds integrate the joint posterior; action branch likelihoods are multiplied
at each decision. Policies receive only their own observable information.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Callable, Mapping

from actions import ACTION_IDS, legal_actions
from hybrid.leaf_evaluation import rollout
from hybrid.policy_source import PolicySource, distribution
from hybrid.ranges import JointRanges
from hybrid.state import BudgetExceeded, ComputeBudget, WorkCounter, instantiate
from poker_game_expresso import HandState
from hybrid.actions import local_actions
from hybrid.beliefs import BeliefState, IncompatibleObservation
from hybrid.behavior import likelihood


@dataclass
class SearchResult:
    action_ev: dict[str, float]
    probabilities: dict[str, float]
    standard_errors: dict[str, float | None]
    paired_difference_errors: dict[str, float | None]
    work: WorkCounter
    method: str
    warnings: list[str]


def search(public: HandState, private: JointRanges, hero: int, policies: Mapping[int, PolicySource],
           budget: ComputeBudget, cancelled: Callable[[], bool] = lambda: False,
           root_raise_amounts: tuple[float, ...] = (), *,
           public_beliefs: BeliefState | None = None, learned_continuation=None,
           adaptive: bool = False) -> SearchResult:
    if public.terminal or public.current_player != hero or set(policies) != set(public.players):
        raise ValueError("Search requires an acting Hero and explicit policies for every seated player")
    if set(private.ranges) != set(public.players) or private.board != tuple(public.board):
        raise ValueError("Search belief and public state mismatch")
    if len(private.ranges[hero].weights) != 1:
        raise ValueError("Action evaluation requires a private Hero view, separate from public solving ranges")
    rng = random.Random(budget.seed)
    work = WorkCounter(budget, cancelled)
    actions = local_actions(public, root_raise_amounts)
    records: list[dict[str, float]] = []
    warnings = ["Action EV assumes the supplied continuation behavior; this is not an equilibrium solve.",
                "Sampling errors exclude uncertainty about opponent ranges and behavior."]
    if learned_continuation is not None and public_beliefs is None:
        raise ValueError("Learned continuations require unconditioned public beliefs")

    def advance_beliefs(before, after, action, beliefs):
        if beliefs is None:
            return None
        def action_likelihood(holding):
            work.visit()
            return likelihood(policies[before.current_player], before, holding, action).probability
        updated = beliefs.update(before.current_player, action_likelihood,
                                 model_version=policies[before.current_player].version)
        return updated.reveal(tuple(after.board)) if after.board != before.board else updated

    def evaluate(state: HandState, depth: int, branch_rng: random.Random, beliefs: BeliefState | None) -> float:
        work.visit()
        if state.terminal:
            work.terminal_leaves += 1
            return state.utility(hero)
        if depth >= budget.max_depth:
            if learned_continuation is not None and state.current_player == hero and beliefs is not None:
                work.learned_leaves += 1
                return learned_continuation.value(state, beliefs.public, tuple(sorted(state.players[hero].cards)))
            return rollout(state, hero, policies, branch_rng, work)
        probabilities = distribution(policies[state.current_player], state)
        value = 0.0
        # All legal actions stay available; exact zero policy mass is a stated
        # continuation assumption, not permanent action pruning in a solver.
        for action in legal_actions(state):
            probability = probabilities[ACTION_IDS.index(action.action_id)]
            if probability == 0:
                continue
            child = state.clone()
            action.apply(child)
            next_beliefs = advance_beliefs(state, child, action, beliefs) if learned_continuation is not None else None
            value += probability * evaluate(child, depth + 1, branch_rng, next_beliefs)
        return value

    pilot_count = min(8, budget.samples//4) if adaptive and budget.samples >= 16 else 0
    pilot_records: list[dict[str, float]] = []
    focused: set[str] = set()
    for sample_index in range(budget.samples):
        if pilot_count and sample_index == pilot_count:
            pilot_records, records = records, []
            pilot_means = {a.action_id: math.fsum(r[a.action_id] for r in pilot_records)/pilot_count for a in actions}
            best_pilot = max(pilot_means.values())
            for a, mean in pilot_means.items():
                se = math.sqrt(sum((r[a]-mean)**2 for r in pilot_records)/(pilot_count-1)/pilot_count)
                if mean + 2*se >= best_pilot:
                    focused.add(a)
            focused.update(sorted(pilot_means, key=pilot_means.get, reverse=True)[:2])
        hands = private.sample(rng)
        world = instantiate(public, hands, rng)
        continuation_seed = rng.randrange(2**63)
        row = {}
        try:
            for action in actions:
                # Allocation depends only on the independent pilot. Every action
                # receives fresh evaluation; none is permanently pruned.
                if pilot_count and sample_index >= pilot_count and action.action_id not in focused and (sample_index-pilot_count) % 4:
                    continue
                child = world.clone()
                action.apply(child)
                next_beliefs = None
                if learned_continuation is not None:
                    try:
                        next_beliefs = advance_beliefs(world, child, action, public_beliefs)
                    except IncompatibleObservation:
                        warning = f"Root action {action.action_id} has zero reference likelihood; baseline rollouts used for this intervention."
                        if warning not in warnings:
                            warnings.append(warning)
                row[action.action_id] = evaluate(child, 1, random.Random(continuation_seed), next_beliefs)
        except BudgetExceeded:
            if budget.strict or not records:
                raise
            warnings.append("Budget exhausted: incomplete paired action round discarded; stopping may depend on path cost.")
            break
        records.append(row)
        work.samples += 1

    if not records:
        raise BudgetExceeded("Budget exhausted before independent evaluation after adaptive pilot")
    values_by_action = {a.action_id: [row[a.action_id] for row in records if a.action_id in row] for a in actions}
    if any(not values for values in values_by_action.values()):
        raise BudgetExceeded("Adaptive evaluation did not obtain a fresh estimate for every legal action")
    means = {a: math.fsum(values)/len(values) for a, values in values_by_action.items()}

    def error(values: list[float]) -> float | None:
        if len(values) < 2:
            return None
        mean = math.fsum(values) / len(values)
        return math.sqrt(math.fsum((v - mean)**2 for v in values) / (len(values) - 1) / len(values))

    standard_errors = {a: error(values) for a, values in values_by_action.items()}
    reference = actions[0].action_id
    paired = {a: error([r[a] - r[reference] for r in records if a in r and reference in r]) for a in means}
    if pilot_count:
        warnings.append(f"Independent adaptive pilot: {pilot_count} worlds; evaluation counts per action: "
                        f"{ {a: len(v) for a, v in values_by_action.items()} }. Pilot excluded from EV estimates.")
    best = max(means.values())
    ties = [a for a, v in means.items() if math.isclose(v, best, abs_tol=1e-10, rel_tol=0)]
    strategy = {a: float(a in ties) / len(ties) for a in means}
    leaders = sorted(means, key=means.get, reverse=True)[:2]
    if len(leaders) == 2:
        left, right = leaders
        gap_error = error([r[left]-r[right] for r in records if left in r and right in r])
        if gap_error is None or means[left]-means[right] <= 1.96*gap_error:
            warnings.append("Monte Carlo precision is insufficient to distinguish the two highest estimated action values at an approximate 95% paired threshold.")
    method = ("sampled-chance/depth-limited-learned-and-rollout-leaves" if work.learned_leaves
              else "sampled-chance/depth-limited-policy-rollout")
    return SearchResult(means, strategy, standard_errors, paired, work, method, warnings)
