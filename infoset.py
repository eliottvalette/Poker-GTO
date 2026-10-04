"""Exact observations and lossless tabular keys; no opponent private information."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import json
import math
from actions import ACTION_IDS, legal_actions
from poker_game_expresso import HandState, STREETS
from tournament import TournamentState

STATE_VERSION = 3
POSITIONS = ("BTN", "SB", "BB")
EVENTS = ("BLIND", "FOLD", "CHECK", "CALL", "RAISE", "CARD", "STACK")
NUMERIC_NAMES = tuple(f"{feature}_{i}" for feature in ("stack", "street_bet", "contribution", "folded", "effective") for i in range(3)) + (
    "pot", "to_call", "highest", "last_full_raise", "hero_bet", "small_blind", "big_blind",
    "player_count", "hero_position", "button", "hand_number", "initial_0", "initial_1", "initial_2",
) + tuple(f"target_{a}" for a in ACTION_IDS) + ("blind_level_index", "chip_unit_big_blind")
HISTORY_WIDTH = 12


@dataclass(frozen=True)
class Observation:
    version: int
    hero: int
    objective: str
    cards: tuple[int, ...]  # Two private cards and five board slots; 52 means unknown.
    street: int
    numeric: tuple[float, ...]
    legal_mask: tuple[bool, ...]
    history: tuple[tuple[float, ...], ...]
    recall: str  # Lossless perfect-recall source for tabular keys and diagnostics.

    def __post_init__(self) -> None:
        if (self.version != STATE_VERSION or self.objective != "hand_chip_delta"
                or self.street not in range(4) or len(self.cards) != 7
                or any(c not in range(52) for c in self.cards[:2])
                or any(c not in range(53) for c in self.cards[2:])
                or len(set(c for c in self.cards if c != 52)) != sum(c != 52 for c in self.cards)
                or len(self.numeric) != len(NUMERIC_NAMES) or len(self.legal_mask) != len(ACTION_IDS)
                or not any(self.legal_mask) or not self.history
                or any(not math.isfinite(v) for v in self.numeric)
                or any(len(e) != HISTORY_WIDTH or any(not math.isfinite(v) for v in e) for e in self.history)):
            raise ValueError(f"Invalid structured observation: version={self.version}, objective={self.objective}, cards={self.cards}")
        if not isinstance(json.loads(self.recall), list):
            raise ValueError("Observation recall must encode a list of hand observations")

    def key(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), allow_nan=False)


def observe(state: HandState | TournamentState) -> Observation:
    hand = state if isinstance(state, HandState) else state.hand
    if hand is None or hand.terminal:
        raise ValueError("Observation requires a live decision")
    scale = 25 * hand.blinds.big
    hero = hand.actor
    seats = [hero.player_id] + [i for i in hand.players if i != hero.player_id]
    seat_index = {i: n for n, i in enumerate(seats)}
    players = [hand.players[i] for i in seats]
    actions = {a.action_id: a for a in legal_actions(hand)}
    numeric: list[float] = []
    for feature in ("stack", "street_bet", "contribution", "folded", "effective"):
        values = []
        for p in players:
            value = min(hero.stack, p.stack) if feature == "effective" else getattr(p, feature)
            values.append(float(value) if feature == "folded" else float(value) / scale)
        numeric.extend(values + [0.0] * (3 - len(values)))
    hand_number = hand.hand_number
    numeric.extend([hand.pot / scale, hand.to_call() / scale, hand.highest / scale,
                    hand.last_full_raise / scale, hero.street_bet / scale,
                    hand.blinds.small / scale, hand.blinds.big / scale,
                    len(players) / 3, POSITIONS.index(hero.position) / 2,
                    seat_index[hand.button] / 2, hand_number / 25])
    numeric.extend([hand.initial_stacks[i] / scale for i in seats] + [0.0] * (3 - len(seats)))
    numeric.extend((actions[a].amount_to or 0.0) / scale if a in actions else 0.0 for a in ACTION_IDS)
    numeric.extend((float(hand.blind_level_index), hand.blinds.big))
    if len(numeric) != len(NUMERIC_NAMES):
        raise ValueError(f"Numeric schema mismatch: {len(numeric)} != {len(NUMERIC_NAMES)}")
    history: list[tuple[float, ...]] = []
    identity = seats
    # Per-hand recall retains every public event and exact hero cards. Earlier
    # hands are simulator records, not descendants or history of this solver game.
    recall = [{"hand": hand_number, "button": hand.button, "initial": hand.initial_stacks,
               "hole": hero.cards, "board": hand.board, "shown_cards": {}, "final_stacks": None,
               "events": [asdict(event) for event in hand.history]}]
    hand_token = 1 / 25
    for player_id, amount in hand.initial_stacks.items():
        history.append((hand_token, 0, identity.index(player_id) / 2,
                        POSITIONS.index(hand.players[player_id].position) / 2, EVENTS.index("STACK") / 6,
                        amount / scale, 0, 0, 0, 0, 0, 0))
    for card in hero.cards:
        history.append((hand_token, 0, 0, POSITIONS.index(hero.position) / 2,
                        EVENTS.index("CARD") / 6, 0, 0, 0, 0, 0, card / 51, 1))
    revealed = 0
    for event in hand.history:
        street = STREETS.index(event.street)
        required = (0, 3, 4, 5)[street]
        for card in hand.board[revealed:required]:
            history.append((hand_token, street / 3, 0, 0, EVENTS.index("CARD") / 6,
                            0, 0, 0, 0, 0, card / 51, 1))
        revealed = required
        history.append((hand_token, street / 3, identity.index(event.player_id) / 2,
                        POSITIONS.index(event.position) / 2, EVENTS.index(event.action) / 6,
                        event.amount_to / scale, event.amount_added / scale, event.pot_before / scale,
                        event.pot_after / scale, event.highest_before / scale, 0, 0))
    for index in range(revealed, len(hand.board)):
        street = 1 if index < 3 else index - 1
        history.append((hand_token, street / 3, 0, 0, EVENTS.index("CARD") / 6,
                        0, 0, 0, 0, 0, hand.board[index] / 51, 1))
    return Observation(STATE_VERSION, hero.player_id,
                       "hand_chip_delta",
                       (*hero.cards, *hand.board, *([52] * (5 - len(hand.board)))),
                       STREETS.index(hand.street), tuple(numeric),
                       tuple(a in actions for a in ACTION_IDS), tuple(history),
                       json.dumps(recall, sort_keys=True, separators=(",", ":"), allow_nan=False))
