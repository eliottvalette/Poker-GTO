"""Policies consume safe observations, never a dealt world or future deck."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Protocol

from actions import ACTION_IDS
from cfr_solver import validate_strategy
from infoset import Observation, observe
from poker_game_expresso import HandState


class PolicySource(Protocol):
    version: str

    def probabilities(self, observation: Observation) -> tuple[float, ...]: ...


def distribution(policy: PolicySource, state: HandState) -> tuple[float, ...]:
    obs = observe(state)
    result = policy.probabilities(obs)
    validate_strategy(result, obs.legal_mask)
    return result


@dataclass(frozen=True)
class UniformLegalPolicy:
    version: str = "uniform-legal-v1"

    def probabilities(self, observation: Observation) -> tuple[float, ...]:
        return tuple(float(m) / sum(observation.legal_mask) for m in observation.legal_mask)


@dataclass(frozen=True)
class ScriptedPolicy:
    callback: Callable[[Observation], tuple[float, ...]]
    version: str

    def probabilities(self, observation: Observation) -> tuple[float, ...]:
        result = self.callback(observation)
        validate_strategy(result, observation.legal_mask)
        return result


@dataclass(frozen=True)
class TabularPolicy:
    table: Mapping[str, tuple[float, ...]]
    version: str = "explicit-table-v1"

    def probabilities(self, observation: Observation) -> tuple[float, ...]:
        if observation.key() not in self.table:
            raise KeyError(f"Policy table lacks information set: {observation.key()}")
        result = self.table[observation.key()]
        validate_strategy(result, observation.legal_mask)
        return result


class LocallySolvedPolicy(TabularPolicy):
    """An explicit table produced by a local solver, with identical lookup semantics."""


@dataclass(frozen=True)
class NeuralAveragePolicy:
    """Adapter for a validated existing router/artifact; reliability is not implied."""

    query: Callable[[Observation], tuple[float, ...]]
    version: str

    def probabilities(self, observation: Observation) -> tuple[float, ...]:
        result = self.query(observation)
        validate_strategy(result, observation.legal_mask)
        return result


def category_policy(weights: Mapping[str, float], version: str) -> ScriptedPolicy:
    """Explicit behavior population without hidden-card hand-strength rules."""
    def choose(obs: Observation) -> tuple[float, ...]:
        values = [weights.get(a if a in ("FOLD", "CHECK", "CALL", "ALL_IN") else "RAISE", 0.0)
                  if m else 0.0 for a, m in zip(ACTION_IDS, obs.legal_mask)]
        total = sum(values)
        if total <= 0:
            raise ValueError(f"Script {version} assigns zero mass to all legal actions")
        return tuple(v / total for v in values)
    return ScriptedPolicy(choose, version)
