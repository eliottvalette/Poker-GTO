"""Persistent fixed-blind 3-max Expresso tournament lifecycle."""
from __future__ import annotations
import copy
import math
import random
from dataclasses import dataclass, field
from poker_game_expresso import BlindLevel, EPS, HandState
from actions import apply_action


@dataclass
class TournamentState:
    stacks: dict[int, float] = field(default_factory=lambda: {0: 25.0, 1: 25.0, 2: 25.0})
    button: int = 0
    blinds: BlindLevel = field(default_factory=BlindLevel)
    rng: random.Random = field(default_factory=random.Random)
    hand_number: int = 0
    hand: HandState | None = None
    completed: list[HandState] = field(default_factory=list)
    total_chips: float = field(init=False)
    original_players: tuple[int, ...] = field(init=False)

    def __post_init__(self) -> None:
        self.stacks = dict(self.stacks)
        if len(self.stacks) not in (2, 3) or any(not math.isfinite(v) or v < 0 for v in self.stacks.values()):
            raise ValueError(f"Expected 2/3 finite nonnegative tournament stacks: {self.stacks}")
        if not self.active or self.button not in self.active:
            raise ValueError(f"Button {self.button} must be active: {self.active}")
        self.total_chips = sum(self.stacks.values())
        self.original_players = tuple(self.stacks)

    @property
    def active(self) -> tuple[int, ...]:
        stacks = ({i: p.stack for i, p in self.hand.players.items()}
                  if self.hand is not None and self.hand.terminal else self.stacks)
        return tuple(i for i, s in stacks.items() if s > EPS)

    @property
    def terminal(self) -> bool:
        return len(self.active) == 1

    @property
    def winner(self) -> int | None:
        return self.active[0] if self.terminal else None

    def clone(self) -> TournamentState:
        return copy.deepcopy(self)

    def start_hand(self, deck: list[int] | None = None) -> HandState:
        if self.hand is not None:
            if not self.hand.terminal:
                raise ValueError("Cannot start another hand before settlement")
            for i, p in self.hand.players.items():
                self.stacks[i] = p.stack
            self.completed.append(self.hand)
            seats = list(self.stacks)
            active = tuple(i for i in seats if self.stacks[i] > EPS)
            if len(self.hand.players) == 3 and len(active) == 2:
                # TDA 36-C: entering heads-up must not charge the previous BB twice.
                previous_bb = next(i for i, p in self.hand.players.items() if p.position == "BB")
                offset = seats.index(previous_bb)
                next_bb = next(seats[(offset + n) % len(seats)] for n in range(1, len(seats) + 1)
                               if seats[(offset + n) % len(seats)] in active)
                self.button = next(i for i in active if i != next_bb)
            else:
                offset = seats.index(self.button)
                self.button = next(seats[(offset + n) % len(seats)] for n in range(1, len(seats) + 1)
                                   if seats[(offset + n) % len(seats)] in active)
            self.hand = None
        if self.terminal:
            raise ValueError(f"Tournament already won by player {self.winner}")
        self.hand_number += 1
        self.hand = HandState.start({i: self.stacks[i] for i in self.active}, self.button,
                                    self.rng, self.blinds, deck)
        self.assert_invariants()
        return self.hand

    @property
    def current_player(self) -> int | None:
        return None if self.hand is None else self.hand.current_player

    def act(self, action_id: str) -> None:
        if self.hand is None:
            raise ValueError("Tournament has no current hand; call start_hand")
        apply_action(self.hand, action_id)
        self.assert_invariants()

    def utility(self, player: int) -> float:
        if player not in self.original_players or not self.terminal:
            raise ValueError(f"Winner utility requires terminal tournament and valid player: {player}")
        return float(player == self.winner) - 1.0 / len(self.original_players)

    def assert_invariants(self) -> None:
        if self.hand is None:
            actual = sum(self.stacks.values())
        else:
            self.hand.assert_invariants()
            actual = sum(p.stack for p in self.hand.players.values()) + self.hand.pot
        if not math.isclose(actual, self.total_chips, abs_tol=EPS):
            raise ValueError(f"Tournament chip conservation: {actual} != {self.total_chips}")
