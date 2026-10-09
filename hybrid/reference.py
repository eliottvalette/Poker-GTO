"""Auditable behavior populations and validated finite-game reference tables."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import json
import math
from pathlib import Path

from actions import ACTION_IDS
from features.cards import canonical_private_cards
from hybrid.policy_source import PolicySource, UniformLegalPolicy
from infoset import NUMERIC_NAMES, Observation
from utils import rank7

PROFILES = {
    "conservative": (0.0, 0.25),
    "tight_passive": (0.12, -0.8),
    "loose_passive": (-0.12, -0.8),
    "tight_aggressive": (0.12, 0.8),
    "loose_aggressive": (-0.12, 0.8),
    "push_fold": (0.04, 1.4),
}


@dataclass(frozen=True)
class MixturePolicy:
    """Explicit public-history-dependent synthetic ensemble, not a calibrated human."""

    components: tuple[PolicySource, ...]
    weights: tuple[float, ...]
    adaptive: bool = False
    version: str = "synthetic-mixture-v1"

    def __post_init__(self) -> None:
        if (
            len(self.components) != len(self.weights)
            or not self.weights
            or any(not math.isfinite(w) or w < 0 for w in self.weights)
            or sum(self.weights) <= 0
        ):
            raise ValueError(
                "Mixture requires matching policies and positive finite weight mass"
            )

    def probabilities(self, observation: Observation) -> tuple[float, ...]:
        from infoset import EVENTS

        weights = list(self.weights)
        if self.adaptive:
            raises = sum(
                round(event[4] * 6) == EVENTS.index("RAISE")
                for event in observation.history
            )
            weights[-1] *= 1 + min(raises, 10)
        total = sum(weights)
        probabilities = [
            policy.probabilities(observation) for policy in self.components
        ]
        return tuple(
            sum(w * p[i] for w, p in zip(weights, probabilities)) / total
            for i in range(len(ACTION_IDS))
        )


@lru_cache(maxsize=32768)
def showdown_strength(cards: tuple[int, ...], count: int, samples: int = 24) -> float:
    """Deterministic uniform-deal equity proxy, explicitly not betting action EV.

    Shared integer PRNG permits browser parity. This is a scripted population
    ingredient, not an equilibrium claim or a learned hand-strength policy.
    """
    canonical, _ = canonical_private_cards(cards)
    known = tuple(c for c in canonical if c < 52)
    seed = 2166136261
    for c in (*canonical, count):
        seed = ((seed ^ c) * 16777619) & 0xFFFFFFFF
    total = 0.0
    for _ in range(samples):
        deck = [c for c in range(52) if c not in known]
        drawn = []
        for _ in range(2 * (count - 1) + 7 - len(known)):
            seed = (1664525 * seed + 1013904223) & 0xFFFFFFFF
            index = seed % len(deck)
            drawn.append(deck.pop(index))
        board = (*known[2:], *drawn[2 * (count - 1) :])
        ranks = [rank7((*known[:2], *board))]
        ranks.extend(
            rank7((*drawn[2 * i : 2 * i + 2], *board)) for i in range(count - 1)
        )
        best = max(ranks)
        total += 1 / ranks.count(best) if ranks[0] == best else 0.0
    return total / samples


@dataclass(frozen=True)
class ProfilePolicy:
    """Explicit synthetic profile; parameters are assumptions, not human calibration."""

    profile: str = "conservative"

    def __post_init__(self) -> None:
        if self.profile not in PROFILES and self.profile != "uniform":
            raise ValueError(f"Unsupported behavior profile: {self.profile}")

    @property
    def version(self) -> str:
        return f"observable-equity-profile-v1/{self.profile}/24-deals"

    def probabilities(self, observation: Observation) -> tuple[float, ...]:
        if self.profile == "uniform":
            return UniformLegalPolicy().probabilities(observation)
        values = dict(zip(NUMERIC_NAMES, observation.numeric))
        count = round(values["player_count"] * 3)
        strength = showdown_strength(observation.cards, count)
        tightness, aggression = PROFILES[self.profile]
        pot, call = values["pot"], min(values["to_call"], values["stack_0"])
        odds = call / (pot + call) if call else 0.0
        advantage = strength - odds - tightness
        scores = []
        for action, legal in zip(ACTION_IDS, observation.legal_mask):
            if not legal:
                scores.append(0.0)
                continue
            if action == "FOLD":
                logit = -5 * advantage
            elif action in ("CHECK", "CALL"):
                logit = 3 * advantage - aggression
            else:
                added = max(0.0, values[f"target_{action}"] - values["hero_bet"])
                risk = added / (pot + added) if added else 0.0
                logit = 6 * (strength - 0.5 - tightness) + aggression - 2 * risk
            if self.profile == "push_fold" and action not in ("FOLD", "ALL_IN"):
                logit -= 5
            scores.append(math.exp(max(-20.0, min(20.0, logit))))
        total = sum(scores)
        return tuple(v / total for v in scores)


@dataclass
class ReferencePolicy:
    """Exact validated table first; explicit computed conservative population elsewhere."""

    table: dict[str, tuple[float, ...]] = field(default_factory=dict)
    validation: dict = field(default_factory=dict)
    fallback: PolicySource = field(default_factory=ProfilePolicy)
    version: str = "computed-reference-v1"
    table_hits: int = 0
    fallback_hits: int = 0

    def __post_init__(self) -> None:
        if self.table and not self.validation.get("accepted"):
            raise ValueError(
                "Nonempty computed reference tables require explicit validation evidence"
            )

    def probabilities(self, observation: Observation) -> tuple[float, ...]:
        key = observation.key()
        if key in self.table:
            self.table_hits += 1
            return self.table[key]
        self.fallback_hits += 1
        return self.fallback.probabilities(observation)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "schema": 1,
                    "version": self.version,
                    "validation": self.validation,
                    "table": self.table,
                },
                allow_nan=False,
            )
            + "\n"
        )

    @classmethod
    def load(cls, path: Path) -> ReferencePolicy:
        from cfr_solver import validate_strategy

        raw = json.loads(path.read_text())
        if raw["schema"] != 1 or not raw["validation"].get("accepted"):
            raise ValueError(
                "Reference table requires a supported schema and measured acceptance"
            )
        table = {key: tuple(value) for key, value in raw["table"].items()}
        for key, value in table.items():
            validate_strategy(value, tuple(json.loads(key)["legal_mask"]))
        return cls(table, raw["validation"], version=raw["version"])


def solve_reference(
    root,
    ranges,
    *,
    iterations: int = 200,
    max_nodes: int = 1000000,
    maximum_response_gain: float = 0.05,
) -> ReferencePolicy:
    from hybrid.river_solver import RiverSolver
    from hybrid.state import ComputeBudget, instantiate
    from evaluation import subgame_best_response

    solved = RiverSolver().solve(
        root, ranges, ComputeBudget(iterations=iterations, max_nodes=max_nodes)
    )
    worlds = [(d.probability, instantiate(root, d.hands)) for d in ranges.enumerate()]
    responses = {
        p: subgame_best_response(worlds, solved.policy.probabilities, p)
        for p in root.players
    }
    gain = sum(row["best_response_gain_bb"] for row in responses.values())
    accepted = gain <= maximum_response_gain
    if not accepted:
        raise ValueError(
            f"Computed reference rejected: response gain {gain} > {maximum_response_gain}"
        )
    return ReferencePolicy(
        dict(solved.policy.table),
        {
            "accepted": True,
            "response_gain_sum": gain,
            "responses": responses,
            "iterations": iterations,
            "nodes": solved.work.nodes,
            "scope": "this exact finite public-range river game; no safe re-solving guarantee",
        },
    )
