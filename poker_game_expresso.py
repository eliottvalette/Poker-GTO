"""No-limit Hold'em rules in fixed physical chip units equal to the initial BB."""
from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field
from typing import Mapping

from utils import rank7

EPS = 1e-9
STREETS = ("PREFLOP", "FLOP", "TURN", "RIVER")


@dataclass(frozen=True)
class BlindLevel:
    small: float = 0.5
    big: float = 1.0

    def __post_init__(self) -> None:
        if not (math.isfinite(self.small) and math.isfinite(self.big)
                and 0 < self.small < self.big):
            raise ValueError(f"Invalid blinds: {self}; expected 0 < SB < BB")


@dataclass
class HandPlayer:
    player_id: int
    stack: float
    position: str
    cards: tuple[int, int]
    street_bet: float = 0.0
    contribution: float = 0.0
    folded: bool = False
    acted_at: float | None = None


@dataclass(frozen=True)
class ActionEvent:
    street: str
    player_id: int
    position: str
    action: str
    amount_to: float
    amount_added: float
    pot_before: float
    pot_after: float
    highest_before: float


@dataclass
class HandState:
    players: dict[int, HandPlayer]
    button: int
    blinds: BlindLevel
    deck: list[int]
    total_chips: float
    initial_stacks: dict[int, float]
    board: list[int] = field(default_factory=list)
    street: str = "PREFLOP"
    pot: float = 0.0
    highest: float = 1.0
    last_full_raise: float = 1.0
    pending: set[int] = field(default_factory=set)
    current_player: int | None = None
    history: list[ActionEvent] = field(default_factory=list)
    terminal: bool = False
    showdown: bool = False
    awards: dict[int, float] = field(default_factory=dict)
    hand_number: int = 1
    blind_level_index: int = 0

    @classmethod
    def start(cls, stacks: Mapping[int, float], button: int, rng: random.Random,
              blinds: BlindLevel = BlindLevel(), deck: list[int] | None = None, *,
              hand_number: int = 1, blind_level_index: int = 0) -> HandState:
        if len(stacks) not in (2, 3) or button not in stacks:
            raise ValueError(f"Expected 2 or 3 active seats and live button, got {stacks}, {button}")
        if any(not math.isfinite(s) or s <= 0 for s in stacks.values()):
            raise ValueError(f"Active stacks must be finite and positive: {stacks}")
        cards = list(range(52)) if deck is None else list(deck)
        if len(cards) != 52 or set(cards) != set(range(52)):
            raise ValueError("Deck must be a permutation of card IDs 0..51")
        if deck is None:
            rng.shuffle(cards)
        seats = list(stacks)
        offset = seats.index(button)
        order = seats[offset:] + seats[:offset]
        roles = ("SB", "BB") if len(order) == 2 else ("BTN", "SB", "BB")
        players = {i: HandPlayer(i, float(stacks[i]), roles[order.index(i)],
                                (cards.pop(), cards.pop())) for i in seats}
        hand = cls(players, button, blinds, cards, sum(stacks.values()), dict(stacks),
                   highest=blinds.big, last_full_raise=blinds.big,
                   hand_number=hand_number, blind_level_index=blind_level_index)
        for role, amount in (("SB", blinds.small), ("BB", blinds.big)):
            p = next(p for p in players.values() if p.position == role)
            before = hand.pot
            hand._pay(p, min(p.stack, amount))
            hand.history.append(ActionEvent("PREFLOP", p.player_id, role, "BLIND",
                                            p.street_bet, p.street_bet, before, hand.pot, 0.0))
        hand.pending = {i for i, p in players.items() if p.stack > EPS}
        first = button
        hand._progress(hand.previous(first))
        hand.assert_invariants()
        return hand

    def clone(self) -> HandState:
        return copy.deepcopy(self)

    def previous(self, seat: int) -> int:
        seats = list(self.players)
        return seats[(seats.index(seat) - 1) % len(seats)]

    def next(self, seat: int) -> int:
        seats = list(self.players)
        return seats[(seats.index(seat) + 1) % len(seats)]

    @property
    def actor(self) -> HandPlayer:
        if self.terminal or self.current_player is None:
            raise ValueError("Terminal hand has no acting player")
        return self.players[self.current_player]

    def to_call(self, p: HandPlayer | None = None) -> float:
        p = self.actor if p is None else p
        opponents = [q for q in self.players.values() if q.player_id != p.player_id and not q.folded]
        highest = self.highest
        if not any(q.stack > EPS for q in opponents):
            # A lone actionable player matches actual all-in wagers, not an
            # unposted portion of a short blind. There is no dry side pot.
            highest = min(highest, max((q.street_bet for q in opponents), default=0.0))
        return max(0.0, highest - p.street_bet)

    @property
    def min_raise_to(self) -> float:
        return self.highest + self.last_full_raise

    def can_raise(self) -> bool:
        p = self.actor
        opponents = [q for q in self.players.values() if q.player_id != p.player_id
                     and not q.folded and q.stack > EPS]
        reopened = p.acted_at is None or self.highest - p.acted_at >= self.last_full_raise - EPS
        return bool(opponents) and reopened and p.stack + p.street_bet > self.highest + EPS

    def _pay(self, p: HandPlayer, amount: float) -> None:
        if not math.isfinite(amount) or amount < -EPS or amount > p.stack + EPS:
            raise ValueError(f"Invalid payment {amount}; player {p.player_id} stack={p.stack}")
        p.stack -= amount
        p.street_bet += amount
        p.contribution += amount
        self.pot += amount

    def act(self, category: str, amount_to: float | None = None) -> None:
        p = self.actor
        call = self.to_call(p)
        highest_before, before = self.highest, self.pot
        if category != "RAISE" and amount_to is not None:
            raise ValueError(f"{category} must not specify amount_to={amount_to}")
        if category == "FOLD":
            if call <= EPS:
                raise ValueError("FOLD is excluded when CHECK is available")
            p.folded = True
        elif category == "CHECK":
            if call > EPS:
                raise ValueError(f"Cannot check; to_call={call}")
        elif category == "CALL":
            if call <= EPS:
                raise ValueError(f"Cannot call; to_call={call}")
            self._pay(p, min(call, p.stack))
        elif category == "RAISE":
            if amount_to is None or not math.isfinite(amount_to) or not self.can_raise():
                raise ValueError(f"Invalid raise {amount_to}; raising rights={self.can_raise()}")
            maximum = p.stack + p.street_bet
            if amount_to <= self.highest + EPS or amount_to > maximum + EPS:
                raise ValueError(f"Raise-to {amount_to} must be in ({self.highest}, {maximum}]")
            full = amount_to >= self.min_raise_to - EPS
            if not full and not math.isclose(amount_to, maximum, abs_tol=EPS):
                raise ValueError(f"Raise-to {amount_to} below minimum {self.min_raise_to}; only all-in allowed")
            self._pay(p, amount_to - p.street_bet)
            self.highest = amount_to
            if full:
                self.last_full_raise = amount_to - highest_before
                self.pending = {i for i, q in self.players.items()
                                if i != p.player_id and not q.folded and q.stack > EPS}
            else:
                self.pending.update(i for i, q in self.players.items() if i != p.player_id
                                    and not q.folded and q.stack > EPS
                                    and q.street_bet < self.highest - EPS)
        else:
            raise ValueError(f"Unknown engine category {category!r}; expected FOLD/CHECK/CALL/RAISE")
        p.acted_at = self.highest
        self.pending.discard(p.player_id)
        self.history.append(ActionEvent(self.street, p.player_id, p.position, category,
                                        p.street_bet, self.pot - before, before, self.pot, highest_before))
        self._progress(p.player_id)
        self.assert_invariants()

    def _progress(self, after: int) -> None:
        live = [p for p in self.players.values() if not p.folded]
        if len(live) == 1:
            self._settle()
            return
        able = [p for p in live if p.stack > EPS]
        if len(able) <= 1:
            # A lone player only decides if still facing a bet. No dry side-pot bets.
            self.pending = {p.player_id for p in able if self.to_call(p) > EPS}
        if self.pending:
            seat = after
            for _ in self.players:
                seat = self.next(seat)
                if seat in self.pending:
                    self.current_player = seat
                    return
            raise ValueError(f"Pending seats not in hand: {self.pending}")
        if len(able) <= 1 or self.street == "RIVER":
            if len(live) > 1:
                self.board.extend(self.deck.pop() for _ in range(5 - len(self.board)))
            self._settle()
            return
        self.street = STREETS[STREETS.index(self.street) + 1]
        self.board.extend(self.deck.pop() for _ in range(3 if self.street == "FLOP" else 1))
        self.highest = 0.0
        self.last_full_raise = self.blinds.big
        for p in self.players.values():
            p.street_bet = 0.0
            p.acted_at = None
        self.pending = {p.player_id for p in able}
        self._progress(self.button)

    def _settle(self) -> None:
        live = [p for p in self.players.values() if not p.folded]
        self.showdown = len(live) > 1
        self.awards = {i: 0.0 for i in self.players}
        if len(live) == 1:
            self.awards[live[0].player_id] = self.pot
        else:
            if len(self.board) != 5:
                raise ValueError(f"Showdown requires five board cards: {self.board}")
            ranks = {p.player_id: rank7((*p.cards, *self.board)) for p in live}
            previous = 0.0
            for level in sorted({p.contribution for p in self.players.values() if p.contribution > EPS}):
                contributors = [p for p in self.players.values() if p.contribution >= level - EPS]
                amount = (level - previous) * len(contributors)
                if len(contributors) == 1:
                    winners = contributors  # Return uncalled chips, without showdown eligibility.
                else:
                    eligible = [p for p in contributors if not p.folded]
                    if not eligible:
                        raise ValueError(f"No eligible player for side pot at {level}: {contributors}")
                    best = max(ranks[p.player_id] for p in eligible)
                    winners = [p for p in eligible if ranks[p.player_id] == best]
                for p in winners:
                    self.awards[p.player_id] += amount / len(winners)
                previous = level
        if not math.isclose(sum(self.awards.values()), self.pot, abs_tol=EPS):
            raise ValueError(f"Pot distribution mismatch: pot={self.pot}, awards={self.awards}")
        for i, amount in self.awards.items():
            self.players[i].stack += amount
        self.pot = 0.0
        self.terminal = True
        self.pending.clear()
        self.current_player = None

    def utility(self, player: int) -> float:
        """Settled chip EV: final minus initial stack, in fixed initial-BB units."""
        if not self.terminal:
            raise ValueError("Hand utility requested before settlement")
        return self.players[player].stack - self.initial_stacks[player]

    def assert_invariants(self) -> None:
        if (type(self.hand_number) is not int or self.hand_number < 1
                or type(self.blind_level_index) is not int or self.blind_level_index < 0):
            raise ValueError(f"Invalid hand/blind context: hand_number={self.hand_number!r}, blind_level_index={self.blind_level_index!r}")
        if self.street not in STREETS or (not self.terminal and len(self.board) != (0, 3, 4, 5)[STREETS.index(self.street)]):
            raise ValueError(f"Invalid street/board contract: {self.street}, board={self.board}")
        if self.button not in self.players or len(self.players) not in (2, 3):
            raise ValueError(f"Invalid live seats/button: {list(self.players)}, {self.button}")
        if any(i not in self.players or self.players[i].folded or self.players[i].stack <= EPS for i in self.pending):
            raise ValueError(f"Invalid pending actors: {self.pending}")
        if any(p.street_bet > p.contribution + EPS for p in self.players.values()):
            raise ValueError("Street contributions exceed total hand contributions")
        values = [self.pot, self.highest, self.last_full_raise]
        values.extend(v for p in self.players.values() for v in (p.stack, p.street_bet, p.contribution))
        if any(not math.isfinite(v) or v < -EPS for v in values):
            raise ValueError(f"Nonfinite/negative chip state: {values}")
        chips = sum(p.stack for p in self.players.values()) + self.pot
        if not math.isclose(chips, self.total_chips, abs_tol=EPS):
            raise ValueError(f"Chip conservation failed: {chips} != {self.total_chips}")
        if self.terminal:
            utility_sum = sum(self.players[i].stack - self.initial_stacks[i] for i in self.players)
            if not math.isclose(utility_sum, 0.0, abs_tol=EPS):
                raise ValueError(f"Settled hand utilities must be zero-sum: sum={utility_sum}, initial_stacks={self.initial_stacks}")
        if not self.terminal and not math.isclose(sum(p.contribution for p in self.players.values()), self.pot, abs_tol=EPS):
            raise ValueError("Pot must equal total hand contributions")
        cards = [c for p in self.players.values() for c in p.cards] + self.board + self.deck
        if len(cards) != 52 or set(cards) != set(range(52)):
            raise ValueError(f"Duplicate or missing cards: {cards}")
        if not self.terminal and self.current_player not in self.pending:
            raise ValueError(f"Actor {self.current_player} not pending: {self.pending}")
