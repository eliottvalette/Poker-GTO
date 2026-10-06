"""Compact neural state; lossless diagnostic recall stays in infoset.Observation."""
from __future__ import annotations
from dataclasses import dataclass
import math
import struct
from actions import ACTION_IDS
from infoset import HISTORY_WIDTH, NUMERIC_NAMES, STATE_VERSION, Observation
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


def neural_observation(obs: Observation | NeuralObservation) -> NeuralObservation:
    if isinstance(obs, NeuralObservation):
        return obs
    cards, mapping = canonical_suits(obs.cards)
    history = []
    for row in obs.history:
        event = list(row)
        if event[-1] == 1:
            card = round(event[10] * 51)
            event[10] = (card // 4 * 4 + mapping[card % 4]) / 51
        history.extend(event)
    numeric = (*obs.numeric, *derived_features(obs.cards, obs.numeric, NUMERIC_NAMES))
    return NeuralObservation(obs.version, obs.hero, obs.objective, cards, obs.street,
                             struct.pack(f"<{len(numeric)}d", *numeric), obs.legal_mask,
                             struct.pack(f"<{len(history)}d", *history))
