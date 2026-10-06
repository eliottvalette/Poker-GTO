"""Arithmetic derived exclusively from the raw observable numeric/card contract."""
from __future__ import annotations
from actions import ACTION_IDS
from features.cards import CARD_FEATURE_NAMES, card_features

DERIVED_NAMES = ("pot_odds", "call_pot", "hero_stack_pot", "spr_1", "spr_2") + tuple(
    f"target_pot_{a}" for a in ACTION_IDS) + CARD_FEATURE_NAMES


def derived_features(cards: tuple[int, ...], numeric: tuple[float, ...], names: tuple[str, ...]) -> tuple[float, ...]:
    state = dict(zip(names, numeric))
    pot, call = state["pot"], min(state["to_call"], state["stack_0"])
    if pot <= 0:
        raise ValueError(f"Live hand must have a positive pot: pot={pot}")
    return (call / (pot + call), call / pot, state["stack_0"] / pot,
            state["effective_1"] / pot, state["effective_2"] / pot,
            *(state[f"target_{a}"] / pot for a in ACTION_IDS),
            *card_features(cards[:2], tuple(c for c in cards[2:] if c != 52)))
