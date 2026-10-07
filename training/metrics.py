"""Streaming coverage counters; no diagnostic metadata bloats replay entries."""
from __future__ import annotations
from collections import Counter
from features.neural import NEURAL_NUMERIC_NAMES, NeuralObservation
from infoset import Observation
from infoset import POSITIONS, EVENTS
from actions import ACTION_IDS


def opening_hand_class(obs: Observation | NeuralObservation) -> str | None:
    """Classify a first SB (HU) or BTN (3-max) preflop decision."""
    if obs.street != 0 or any(round(row[4] * 6) not in (EVENTS.index("STACK"), EVENTS.index("CARD"), EVENTS.index("BLIND")) for row in obs.history):
        return None
    player_count = round(obs.numeric[NEURAL_NUMERIC_NAMES.index("player_count")] * 3)
    position = round(obs.numeric[NEURAL_NUMERIC_NAMES.index("hero_position")] * 2)
    if position != (POSITIONS.index("SB") if player_count == 2 else POSITIONS.index("BTN")):
        return None
    high, low = sorted((card // 4 for card in obs.cards[:2]), reverse=True)
    ranks = "23456789TJQKA"
    return ranks[high] + ranks[low] + ("" if high == low else "s" if obs.cards[0] % 4 == obs.cards[1] % 4 else "o")


class Coverage:
    def __init__(self) -> None:
        self.counts: dict[str, Counter] = {}

    def record(self, obs: Observation | NeuralObservation, source: str) -> None:
        holding = opening_hand_class(obs)
        if holding is not None:
            self.counts.setdefault(f"{source}_opening_hand_class", Counter())[holding] += 1
        values = dict(zip(NEURAL_NUMERIC_NAMES, obs.numeric))
        pot = values["pot"]
        effective = [values[f"effective_{i}"] * 25 for i in range(1, round(values["player_count"] * 3))]
        stack = values["stack_0"] * 25
        minimum = min(effective)
        bin_name = "shallow" if minimum < 8 else "medium" if minimum < 20 else "deep"
        stack_values = [values[f"initial_{i}"] for i in range(round(values["player_count"] * 3))]
        labels = {"source": source, "players": round(values["player_count"] * 3), "street": obs.street,
                  "position": POSITIONS[round(values["hero_position"] * 2)], "blind_level": int(values["blind_level_index"]),
                  "effective_stack_bin": bin_name, "stack_ratio_class": "asymmetric" if max(stack_values) / min(stack_values) >= 4 else "balanced",
                  "hero_stack_current_bb": round(stack, 1), "effective_stack_current_bb": round(minimum, 1),
                  "pot_current_bb": round(pot * 25, 1), "spr": round(minimum / (pot * 25), 1),
                  "call_pot": round(values["to_call"] / pot, 2), "legal_actions": sum(obs.legal_mask),
                  "history_length": len(obs.history)}
        for name, value in labels.items():
            self.counts.setdefault(name, Counter())[str(value)] += 1
        for name, legal in zip(ACTION_IDS, obs.legal_mask):
            if legal:
                self.counts.setdefault("legal_action_frequencies", Counter())[name] += 1
                self.counts.setdefault("action_target_current_bb", Counter())[f"{name}:{values[f'target_{name}'] * 25:.2f}"] += 1

    def merge(self, raw: dict) -> None:
        for name, values in raw.items():
            self.counts.setdefault(name, Counter()).update(values)

    def as_dict(self) -> dict:
        return {name: dict(sorted(values.items())) for name, values in self.counts.items()}
