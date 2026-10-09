"""Optional local root sizes retain physical raise amounts and engine legality."""
from __future__ import annotations
import math
from actions import SolverAction, legal_actions
from poker_game_expresso import EPS, HandState


def local_actions(state: HandState, amounts: tuple[float, ...] = ()) -> tuple[SolverAction, ...]:
    actions = list(legal_actions(state))
    for amount in amounts:
        if not math.isfinite(amount):
            raise ValueError(f"Nonfinite local raise amount: {amount}")
        action = SolverAction(f"RAISE_TO_{amount:.12g}", "RAISE", amount)
        probe = state.clone()
        action.apply(probe)
        if not any(a.amount_to is not None and abs(a.amount_to - amount) <= EPS for a in actions):
            actions.append(action)
    return tuple(actions)
