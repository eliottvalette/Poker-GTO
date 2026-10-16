"""Versioned root-only, range-conditioned inputs for optional solver students."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, permutations
import math
import json

import torch
from torch import nn

from actions import ACTION_IDS
from cfr_solver import regret_matching, validate_strategy
from features.cards import canonical_private_cards
from infoset import Observation, NUMERIC_NAMES
from ml.model import StateEncoder, STREETS, encode_batch
from hybrid.policy_source import PolicySource

INPUT_SCHEMA = "exact-public-factors-root-v1"
COMBOS = tuple(combinations(range(52), 2))
COMBO_INDEX = {hand: index for index, hand in enumerate(COMBOS)}
RIVER_CONTINUATION = "terminal-river-v1"


def continuation_id(contract: dict) -> str:
    return (RIVER_CONTINUATION if contract["root"]["street"] == "RIVER"
            and not contract["forest"].get("depth_limited_river", False) else contract["continuation"])


def range_vector(obs: Observation, contract: dict) -> torch.Tensor:
    """Canonical factors, ordered actor then clockwise; no hidden dealt cards.

    Resolve suit symmetries using the complete sparse factors, including suits
    absent from the visible cards. This preserves exact suit-blocker structure.
    """
    if contract.get("schema") != 1:
        raise ValueError("Unsupported solver game contract")
    root = contract["root"]
    players = {int(p): value for p, value in root["players"].items()}
    count = len(players)
    if count not in (2, 3) or obs.hero not in players:
        raise ValueError("Observation and range player counts differ")
    if round(obs.numeric[NUMERIC_NAMES.index("player_count")] * 3) != count:
        raise ValueError("Observation and range player counts differ")
    if STREETS[obs.street] != root["street"] or tuple(c for c in obs.cards[2:] if c != 52) != tuple(root["board"]):
        raise ValueError("Student query must use its public root board and street")
    recall = json.loads(obs.recall)
    if (len(recall) != 1 or obs.hero != root["current_player"]
            or recall[0]["events"] != root["history"]
            or recall[0]["initial"] != root["initial_stacks"]
            or recall[0]["button"] != root["button"]):
        raise ValueError("Student query is not the declared public decision root")
    positions = ("BTN", "SB", "BB") if count == 3 else ("SB", "BB")
    by_position = {v["position"]: p for p, v in players.items()}
    if set(by_position) != set(positions):
        raise ValueError("Invalid root position mapping")
    start = positions.index(players[obs.hero]["position"])
    seats = [by_position[positions[(start + i) % count]] for i in range(count)]
    factors = []
    if {int(p) for p in contract["ranges"]} != set(players):
        raise ValueError("Root range seats differ")
    for seat in seats:
        rows = contract["ranges"][str(seat)]
        factor = {}
        for hand, weight in rows:
            hand = tuple(sorted(hand))
            if hand not in COMBO_INDEX or hand in factor or not math.isfinite(weight) or weight < 0:
                raise ValueError("Invalid exact range factor")
            if set(hand).intersection(root["board"]):
                raise ValueError("Public board blocks a range combination")
            factor[hand] = weight
        if not math.isclose(sum(factor.values()), 1., abs_tol=1e-10):
            raise ValueError("Public range factor must be normalized")
        factors.append(factor)
    if factors[0].get(tuple(sorted(obs.cards[:2])), 0) <= 0:
        raise ValueError("Candidate holding is outside its public range")
    target, _ = canonical_private_cards(obs.cards)
    candidates = []
    for perm in permutations(range(4)):
        def card(c: int) -> int:
            return 52 if c == 52 else c // 4 * 4 + perm[c % 4]
        visible = (*sorted(card(c) for c in obs.cards[:2]), *(card(c) for c in obs.cards[2:]))
        if visible != target:
            continue
        sparse = tuple(tuple(sorted((COMBO_INDEX[tuple(sorted(card(c) for c in h))], w)
                                    for h, w in factor.items() if w > 0)) for factor in factors)
        candidates.append(sparse)
    if not candidates:
        raise ValueError("No joint card/range suit canonicalization")
    result = torch.zeros(3, len(COMBOS))
    for seat, factor in enumerate(min(candidates)):
        for index, weight in factor:
            result[seat, index] = weight
            if weight > 0 and result[seat, index] == 0:
                raise ValueError("Positive range weight underflows float32")
    return result.flatten()


def encode_roots(observations: list[Observation], contracts: list[dict], vocabulary: tuple[str, ...]) -> dict[str, torch.Tensor]:
    if len(observations) != len(contracts) or not observations:
        raise ValueError("Every root observation requires its solving contract")
    vectors = []
    for obs, contract in zip(observations, contracts):
        identity = continuation_id(contract)
        if identity not in vocabulary:
            raise ValueError(f"Unsupported continuation profile: {identity}")
        profile = [float(identity == value) for value in vocabulary]
        depth = (0 if obs.street == 3 and not contract["forest"].get("depth_limited_river", False)
                 else contract["forest"]["depth"])
        vectors.append(torch.cat((range_vector(obs, contract), torch.tensor([*profile, float(depth)]))))
    return {**encode_batch(observations), "solving_context": torch.stack(vectors)}


class RootStudent(nn.Module):
    """Independent state/range encoder; only valid at a declared solving root."""

    def __init__(self, kind: str, context_width: int, *, conditioned: bool, residual: bool):
        super().__init__()
        self.kind, self.conditioned = kind, conditioned
        self.encoder = StateEncoder()
        self.context = nn.Sequential(nn.Linear(context_width, 64), nn.SiLU())
        self.normalization = nn.LayerNorm(192)
        self.head = nn.Sequential(nn.Linear(192, 64), nn.SiLU() if residual else nn.ReLU(), nn.Linear(64, len(ACTION_IDS)))
        self.direct = nn.Linear(192, len(ACTION_IDS)) if residual else None

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        context = batch["solving_context"]
        if not self.conditioned:
            context = torch.zeros_like(context)
        features = self.normalization(torch.cat((self.encoder(batch), self.context(context)), dim=1))
        output = self.head(features)
        if self.direct is not None:
            output = output + self.direct(features)
        if not torch.isfinite(output).all():
            raise ValueError("Nonfinite root student prediction")
        return output if self.kind == "advantage" else output.masked_fill(~batch["mask"], -torch.inf).softmax(1)


class RootStudentBank(nn.Module):
    def __init__(self, kind: str, vocabulary: tuple[str, ...], *, player_count: int, conditioned: bool, residual: bool):
        super().__init__()
        self.kind, self.vocabulary, self.player_count = kind, vocabulary, player_count
        self.specialists = nn.ModuleDict({s: RootStudent(kind, 3 * len(COMBOS) + len(vocabulary) + 1,
                                                       conditioned=conditioned, residual=residual) for s in STREETS})
        self.register_buffer("trained", torch.zeros(4, dtype=torch.bool))

    def predict(self, observation: Observation, contract: dict) -> tuple[float, ...]:
        if len(contract["root"]["players"]) != self.player_count:
            raise ValueError("Root student player-count track mismatch")
        if not self.trained[observation.street]:
            raise ValueError(f"Untrained solver specialist: {self.kind}/{STREETS[observation.street]}")
        with torch.no_grad():
            output = self.specialists[STREETS[observation.street]](encode_roots([observation], [contract], self.vocabulary))[0].tolist()
        if self.kind == "advantage":
            return tuple(output)
        total = sum(output)
        probabilities = tuple(value / total for value in output)
        validate_strategy(probabilities, observation.legal_mask)
        return probabilities

    def probabilities(self, observation: Observation, contract: dict) -> tuple[float, ...]:
        values = self.predict(observation, contract)
        return regret_matching(values, observation.legal_mask) if self.kind == "advantage" else values


@dataclass
class RootPriorPolicy:
    """Explicit root NN prior and declared baseline at every downstream infoset."""
    bank: RootStudentBank
    contract: dict
    downstream: PolicySource
    version: str = "experimental-root-prior-with-explicit-downstream-v1"

    def probabilities(self, observation: Observation) -> tuple[float, ...]:
        root = self.contract["root"]
        recall = json.loads(observation.recall)
        if observation.hero == root["current_player"] and recall[0]["events"] == root["history"]:
            return self.bank.probabilities(observation, self.contract)
        return self.downstream.probabilities(observation)
