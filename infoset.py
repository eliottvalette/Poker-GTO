"""Exact observations and lossless tabular keys; no opponent private information."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import json
import math
from actions import ACTION_IDS, legal_actions
from poker_game_expresso import HandState, STREETS
from tournament import TournamentState

STATE_VERSION = 2
POSITIONS = ("BTN", "SB", "BB")
EVENTS = ("BLIND", "FOLD", "CHECK", "CALL", "RAISE", "CARD", "STACK")
NUMERIC_NAMES = tuple(f"{feature}_{i}" for feature in ("stack", "street_bet", "contribution", "folded", "effective") for i in range(3)) + (
    "pot", "to_call", "highest", "last_full_raise", "hero_bet", "small_blind", "big_blind",
    "player_count", "hero_position", "button", "hand_number", "initial_0", "initial_1", "initial_2",
) + tuple(f"target_{a}" for a in ACTION_IDS)
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
        if (self.version != STATE_VERSION or self.objective not in ("tournament_winner", "hand_chip_delta")
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
            values.append(float(value) if feature == "folded" else float(value) / 25)
        numeric.extend(values + [0.0] * (3 - len(values)))
    hand_number = 1 if isinstance(state, HandState) else state.hand_number
    numeric.extend([hand.pot / 25, hand.to_call() / 25, hand.highest / 25,
                    hand.last_full_raise / 25, hero.street_bet / 25,
                    hand.blinds.small / 25, hand.blinds.big / 25,
                    len(players) / 3, POSITIONS.index(hero.position) / 2,
                    seat_index[hand.button] / 2, hand_number / 25])
    numeric.extend([hand.initial_stacks[i] / 25 for i in seats] + [0.0] * (3 - len(seats)))
    numeric.extend((actions[a].amount_to or 0.0) / 25 if a in actions else 0.0 for a in ACTION_IDS)
    if len(numeric) != len(NUMERIC_NAMES):
        raise ValueError(f"Numeric schema mismatch: {len(numeric)} != {len(NUMERIC_NAMES)}")
    hands = [hand] if isinstance(state, HandState) else state.completed + [hand]
    history: list[tuple[float, ...]] = []
    recall = []
    # Identity mapping includes earlier eliminated players. IDs are independent of roles.
    identity = list(hand.players) if isinstance(state, HandState) else list(state.original_players)
    identity.remove(hero.player_id)
    identity.insert(0, hero.player_id)
    for n, previous in enumerate(hands, 1):
        own = previous.players.get(hero.player_id)
        private = () if own is None else own.cards
        shown = {i: list(p.cards) for i, p in previous.players.items() if previous.showdown and not p.folded}
        final = {i: p.stack for i, p in previous.players.items()} if previous.terminal else None
        recall.append({"hand": n, "button": previous.button, "initial": previous.initial_stacks,
                       "hole": private, "board": previous.board, "shown_cards": shown, "final_stacks": final,
                       "events": [asdict(e) for e in previous.history]})
        for i, amount in previous.initial_stacks.items():
            history.append((n / 25, 0, identity.index(i) / 2,
                            POSITIONS.index(previous.players[i].position) / 2, EVENTS.index("STACK") / 6,
                            amount / 25, 0, 0, 0, 0, 0, 0))
        for card in private:
            history.append((n / 25, 0, 0, POSITIONS.index(own.position) / 2, EVENTS.index("CARD") / 6, 0, 0, 0, 0, 0, card / 51, 1))
        revealed = 0
        for e in previous.history:
            street = STREETS.index(e.street)
            required = (0, 3, 4, 5)[street]
            for card in previous.board[revealed:required]:
                history.append((n / 25, street / 3, 0, 0, EVENTS.index("CARD") / 6, 0, 0, 0, 0, 0, card / 51, 1))
            revealed = required
            history.append((n / 25, street / 3, identity.index(e.player_id) / 2,
                            POSITIONS.index(e.position) / 2, EVENTS.index(e.action) / 6,
                            e.amount_to / 25, e.amount_added / 25, e.pot_before / 25,
                            e.pot_after / 25, e.highest_before / 25, 0, 0))
        for idx in range(revealed, len(previous.board)):
            street = 1 if idx < 3 else idx - 1
            history.append((n / 25, street / 3, 0, 0, EVENTS.index("CARD") / 6, 0, 0, 0, 0, 0, previous.board[idx] / 51, 1))
        if previous.terminal:
            for i, cards in shown.items():
                if i != hero.player_id:
                    for card in cards:
                        history.append((n / 25, 1, identity.index(i) / 2,
                                        POSITIONS.index(previous.players[i].position) / 2,
                                        EVENTS.index("CARD") / 6, 0, 0, 0, 0, 0, card / 51, 1))
            for i, p in previous.players.items():
                history.append((n / 25, STREETS.index(previous.street) / 3, identity.index(i) / 2,
                                POSITIONS.index(p.position) / 2, EVENTS.index("STACK") / 6,
                                p.stack / 25, 0, 0, 0, 0, 0, 0))
    return Observation(STATE_VERSION, hero.player_id,
                       "hand_chip_delta" if isinstance(state, HandState) else "tournament_winner",
                       (*hero.cards, *hand.board, *([52] * (5 - len(hand.board)))),
                       STREETS.index(hand.street), tuple(numeric),
                       tuple(a in actions for a in ACTION_IDS), tuple(history),
                       json.dumps(recall, sort_keys=True, separators=(",", ":"), allow_nan=False))
