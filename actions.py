"""Canonical discrete abstraction; engine accepts arbitrary legal raise-to sizes."""
from dataclasses import dataclass
from poker_game_expresso import EPS, HandState

ACTION_SCHEMA_VERSION = 1

ACTION_IDS = ("FOLD", "CHECK", "CALL", "RAISE_2.0X", "RAISE_2.5X", "RAISE_3.0X",
              "RAISE_4.0X", "BET_RAISE_33P", "BET_RAISE_50P", "BET_RAISE_75P",
              "BET_RAISE_100P", "BET_RAISE_150P", "ALL_IN")


@dataclass(frozen=True)
class SolverAction:
    action_id: str
    category: str
    amount_to: float | None = None

    def apply(self, hand: HandState) -> None:
        hand.act(self.category, self.amount_to)


def legal_actions(hand: HandState) -> tuple[SolverAction, ...]:
    if hand.terminal:
        return ()
    call = hand.to_call()
    actions = ([SolverAction("FOLD", "FOLD"), SolverAction("CALL", "CALL")]
               if call > EPS else [SolverAction("CHECK", "CHECK")])
    if not hand.can_raise():
        return tuple(actions)
    maximum = hand.actor.stack + hand.actor.street_bet
    if hand.street == "PREFLOP":
        # Open in multiples of BB; re-raise in multiples of the current raise-to.
        base = max(hand.blinds.big, hand.highest)
        targets = [(f"RAISE_{x:.1f}X", x * base) for x in (2.0, 2.5, 3.0, 4.0)]
    else:
        # Raise increment is a fraction of the pot AFTER matching the current bet.
        targets = [(f"BET_RAISE_{pct}P", hand.highest + pct / 100 * (hand.pot + call))
                   for pct in (33, 50, 75, 100, 150)]
    seen: list[float] = []
    for action_id, target in targets:
        target = max(target, hand.min_raise_to)
        if target >= maximum - EPS or any(abs(target - t) < EPS for t in seen):
            continue
        actions.append(SolverAction(action_id, "RAISE", target))
        seen.append(target)
    actions.append(SolverAction("ALL_IN", "RAISE", maximum))
    return tuple(actions)


def apply_action(hand: HandState, action_id: str) -> None:
    actions = {a.action_id: a for a in legal_actions(hand)}
    if action_id not in actions:
        raise ValueError(f"Illegal canonical action {action_id!r}; expected {list(actions)}")
    actions[action_id].apply(hand)
