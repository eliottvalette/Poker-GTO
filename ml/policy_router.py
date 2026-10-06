"""Route explicit average policies by seated player count; never invent a model."""
from __future__ import annotations
from pathlib import Path
from typing import Mapping
from infoset import NUMERIC_NAMES, Observation
from features.neural import NeuralObservation
from ml.deep_cfr import NeuralAveragePolicy


class AveragePolicyRouter:
    def __init__(self, artifacts: Mapping[int, str | Path]) -> None:
        if any(count not in (2, 3) for count in artifacts):
            raise ValueError(f"Policy routes must use player counts 2 or 3: {tuple(artifacts)}")
        self.policies = {}
        for count, path in artifacts.items():
            policy = NeuralAveragePolicy(path)
            if policy.supported_player_counts != (count,):
                raise ValueError(f"Route {count} requires a dedicated model; {path} supports {policy.supported_player_counts}")
            self.policies[count] = policy

    def query(self, observation: Observation | NeuralObservation) -> tuple[float, ...]:
        count = round(observation.numeric[NUMERIC_NAMES.index('player_count')] * 3)
        if count not in self.policies:
            raise KeyError(f"Average policy unavailable for {count} players; loaded routes={tuple(self.policies)}")
        return self.policies[count].query(observation)
