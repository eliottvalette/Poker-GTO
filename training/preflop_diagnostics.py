"""Fixed-context policy probes and paired rollouts, not equilibrium references."""
from __future__ import annotations

from itertools import combinations
import math
import random
from typing import Protocol

from actions import ACTION_IDS, legal_actions
from cfr_solver import validate_strategy
from hybrid.state import instantiate
from infoset import Observation, observe
from poker_game_expresso import HandState
from training.poker_evaluation import EvaluationBudget, play, statistics


class Policy(Protocol):
    def probabilities(self, observation: Observation) -> tuple[float, ...]: ...


def opening_matrix(policy: Policy, count: int, stack_bb: float) -> dict:
    """Query every exact combo at one unopened root, matching Overview aggregation."""
    root = HandState.start({p: stack_bb for p in range(count)}, 0, random.Random(42))
    classes: dict[str, list[tuple[float, ...]]] = {}
    reversal_error = 0.0
    ranks = '23456789TJQKA'
    for a, b in combinations(range(52), 2):
        root.actor.cards = (a, b)
        observation = observe(root)
        values = policy.probabilities(observation)
        validate_strategy(values, observation.legal_mask)
        root.actor.cards = (b, a)
        reversed_values = policy.probabilities(observe(root))
        reversal_error = max(reversal_error, *(abs(x-y) for x, y in zip(values, reversed_values)))
        lo, hi = a // 4, b // 4
        label = ranks[hi] + ranks[lo] + ('' if lo == hi else 's' if a % 4 == b % 4 else 'o')
        classes.setdefault(label, []).append(values)
    return {'player_count': count, 'stack_bb': stack_bb, 'position': root.actor.position,
            'card_reversal_max_error': reversal_error,
            'classes': {label: {'combos': len(rows), 'probabilities': {
                action: math.fsum(row[i] for row in rows) / len(rows)
                for i, action in enumerate(ACTION_IDS) if observation.legal_mask[i]}}
                for label, rows in classes.items()}}


def paired_opening_values(policy: Policy, count: int, stack_bb: float,
                          holding: tuple[int, int], samples: int = 256, seed: int = 19001) -> dict:
    """Force each root action; all later seats follow the same fixed policy.

    Uniform compatible opponent deals and common random numbers across actions.
    Physical hand chip deltas include posted blinds. These values are neither
    counterfactual cumulative regrets nor optimal-continuation action values.
    """
    if type(samples) is not int or not 2 <= samples <= 4096:
        raise ValueError('Opening diagnostic requires 2..4096 samples')
    if len(set(holding)) != 2 or any(type(c) is not int or not 0 <= c < 52 for c in holding):
        raise ValueError('Holding requires two distinct card IDs in 0..51')
    root = HandState.start({p: stack_bb for p in range(count)}, 0, random.Random(seed))
    hero = root.current_player
    actions = legal_actions(root)
    outcomes: dict[str, list[float]] = {action.action_id: [] for action in actions}
    probabilities = None
    for sample in range(samples):
        rng = random.Random(seed + sample * 1009)
        deck = [c for c in range(52) if c not in holding]
        rng.shuffle(deck)
        hands = {hero: holding}
        for seat in root.players:
            if seat != hero:
                hands[seat] = (deck.pop(), deck.pop())
        world = instantiate(root, hands, rng)
        if probabilities is None:
            probabilities = policy.probabilities(observe(world))
            validate_strategy(probabilities, observe(world).legal_mask)
        for action in actions:
            branch = world.clone()
            action.apply(branch)
            value, _ = play(branch, hero, policy, policy, seed + sample * 1009,
                            EvaluationBudget(), lambda: None)
            outcomes[action.action_id].append(value)
    mixture = [math.fsum(probabilities[ACTION_IDS.index(action)] * values[i]
                         for action, values in outcomes.items()) for i in range(samples)]
    return {'samples': samples, 'seed': seed, 'units': 'BB per hand',
            'assumption': 'Uniform private deals; fixed published-policy continuations at every seat',
            'actions': {action: {'probability': probabilities[ACTION_IDS.index(action)],
                                 'ev': statistics(values),
                                 'paired_gain_vs_mixture': statistics([v-m for v, m in zip(values, mixture)])}
                        for action, values in outcomes.items()},
            'mixture_ev': statistics(mixture)}
