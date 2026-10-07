"""Compact neural state; lossless diagnostic recall stays in infoset.Observation."""
from __future__ import annotations
from dataclasses import dataclass
import math
import struct
from actions import ACTION_IDS
from infoset import EVENTS, HISTORY_WIDTH, NUMERIC_NAMES, POSITIONS, STATE_VERSION, Observation
from features import FEATURE_SCHEMA_VERSION
from features.cards import canonical_suits
from features.deterministic import DERIVED_NAMES, derived_features

NEURAL_NUMERIC_NAMES = NUMERIC_NAMES + DERIVED_NAMES


@dataclass(frozen=True, slots=True)
class NeuralObservation:
    version: int
    hero: int
    objective: str
    cards: tuple[int, ...]
    street: int
    numeric_data: bytes
    legal_mask: tuple[bool, ...]
    history_data: bytes
    feature_version: int = FEATURE_SCHEMA_VERSION

    @property
    def numeric(self) -> tuple[float, ...]:
        return struct.unpack(f"<{len(self.numeric_data) // 8}d", self.numeric_data)

    @property
    def history(self) -> tuple[tuple[float, ...], ...]:
        values = struct.unpack(f"<{len(self.history_data) // 8}d", self.history_data)
        return tuple(values[i:i + HISTORY_WIDTH] for i in range(0, len(values), HISTORY_WIDTH))

    def __post_init__(self) -> None:
        if (self.version != STATE_VERSION or self.feature_version != FEATURE_SCHEMA_VERSION
                or self.objective != "hand_chip_delta" or type(self.hero) is not int or self.hero < 0
                or self.street not in range(4) or len(self.cards) != 7
                or any(c not in range(52) for c in self.cards[:2]) or any(c not in range(53) for c in self.cards[2:])
                or len(set(c for c in self.cards if c != 52)) != sum(c != 52 for c in self.cards)
                or len(self.legal_mask) != len(ACTION_IDS) or not any(self.legal_mask)
                or any(type(m) is not bool for m in self.legal_mask)
                or len(self.numeric_data) != 8 * len(NEURAL_NUMERIC_NAMES)
                or not self.history_data or len(self.history_data) % (8 * HISTORY_WIDTH)
                or any(not math.isfinite(v) for v in self.numeric)
                or any(not math.isfinite(v) for row in self.history for v in row)):
            raise ValueError(f"Invalid compact neural schema: state={self.version}, feature={self.feature_version}")


def canonical_player_fields(obs: Observation) -> tuple[tuple[float, ...], tuple[tuple[float, ...], ...]]:
    """Remove arbitrary ring starting seat; order hero then opponents clockwise."""
    count = round(obs.numeric[NUMERIC_NAMES.index("player_count")] * 3)
    positions = (0, 1, 2) if count == 3 else (1, 2)
    stacks = obs.history[:count]
    if count not in (2, 3) or len(stacks) != count or any(row[4] != EVENTS.index("STACK") / 6 for row in stacks):
        raise ValueError(f"Missing initial stack context for canonical seats: player_count={count}")
    by_position = {round(row[3] * 2): round(row[2] * 2) for row in stacks}
    hero_position = round(obs.numeric[NUMERIC_NAMES.index("hero_position")] * 2)
    if set(by_position) != set(positions) or set(by_position.values()) != set(range(count)) or by_position.get(hero_position) != 0:
        raise ValueError(f"Invalid observable position mapping: {by_position}, hero_position={hero_position}")
    offset = positions.index(hero_position)
    order = [by_position[positions[(offset + index) % count]] for index in range(count)]
    mapping = {old: new for new, old in enumerate(order)}
    numeric = list(obs.numeric)
    for name in ("stack", "street_bet", "contribution", "folded", "effective", "initial"):
        indices = [NUMERIC_NAMES.index(f"{name}_{index}") for index in range(count)]
        for new, old in enumerate(order):
            numeric[indices[new]] = obs.numeric[indices[old]]
    button = NUMERIC_NAMES.index("button")
    numeric[button] = mapping[round(obs.numeric[button] * 2)] / 2
    history = []
    for row in obs.history:
        event = list(row)
        event[2] = mapping[round(row[2] * 2)] / 2
        history.append(tuple(event))
    # Stack tokens describe simultaneous initial state, not betting chronology.
    history[:count] = sorted(history[:count], key=lambda row: row[2])
    return tuple(numeric), tuple(history)


def neural_observation(obs: Observation | NeuralObservation) -> NeuralObservation:
    if isinstance(obs, NeuralObservation):
        return obs
    raw_numeric, raw_history = canonical_player_fields(obs)
    cards, mapping = canonical_suits(obs.cards)
    history = []
    for row in raw_history:
        event = list(row)
        if event[-1] == 1:
            card = round(event[10] * 51)
            event[10] = (card // 4 * 4 + mapping[card % 4]) / 51
        history.extend(event)
    numeric = (*raw_numeric, *derived_features(obs.cards, raw_numeric, NUMERIC_NAMES))
    return NeuralObservation(obs.version, obs.hero, obs.objective, cards, obs.street,
                             struct.pack(f"<{len(numeric)}d", *numeric), obs.legal_mask,
                             struct.pack(f"<{len(history)}d", *history))
