"""Baseline continuations end at real canonical settlements, never equity-as-EV."""
from __future__ import annotations
import random
from typing import Mapping

from actions import ACTION_IDS, legal_actions
from cfr_solver import sample_index
from hybrid.policy_source import PolicySource, distribution
from hybrid.state import WorkCounter
from poker_game_expresso import HandState


def rollout(state: HandState, hero: int, policies: Mapping[int, PolicySource],
            rng: random.Random, work: WorkCounter) -> float:
    state = state.clone()
    work.rollout_leaves += 1
    while not state.terminal:
        work.visit()
        probabilities = distribution(policies[state.current_player], state)
        selected = ACTION_IDS[sample_index(probabilities, rng)]
        next(a for a in legal_actions(state) if a.action_id == selected).apply(state)
    work.terminal_leaves += 1
    return state.utility(hero)
