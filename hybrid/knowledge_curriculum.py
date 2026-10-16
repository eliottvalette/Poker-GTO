"""Lazy public-root families with immutable split ownership and bounded manifests."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random

from actions import legal_actions
from hybrid.knowledge import public_root
from hybrid.policy_source import PolicySource, UniformLegalPolicy, category_policy
from hybrid.ranges import HandRange, JointRanges, combo
from poker_game_expresso import HandState
from training.checkpoint import atomic_bytes


@dataclass(frozen=True)
class Family:
    name: str
    split: str
    stack: float
    asymmetric: bool
    texture: str
    raises: int
    width: int
    profile: str


FAMILIES = (
    Family("shallow-dry-open", "train", 5, False, "dry", 0, 8, "uniform"),
    Family("medium-two-tone-open", "train", 15, False, "two-tone", 1, 32, "passive"),
    Family("medium-paired-asymmetric", "validation", 18, True, "paired", 1, 48, "passive"),
    Family("deep-monotone-threebet", "test", 35, True, "monotone", 2, 128, "aggressive"),
    Family("shallow-paired-open", "test", 7, True, "paired", 0, 16, "uniform"),
)


@dataclass(frozen=True)
class Scenario:
    identity: str
    family: str
    split: str
    root: HandState
    ranges: JointRanges
    policy: PolicySource


def board_texture(board: list[int]) -> str:
    flop = board[:3]
    if len(flop) != 3:
        raise ValueError("Texture requires a flop")
    if len({c // 4 for c in flop}) < 3:
        return "paired"
    suits = len({c % 4 for c in flop})
    return {1: "monotone", 2: "two-tone", 3: "dry"}[suits]


def scenario(count: int, street: str, family: Family, index: int) -> Scenario:
    if count not in (2, 3) or street not in ("PREFLOP", "FLOP", "TURN", "RIVER") or type(index) is not int or index < 0:
        raise ValueError("Invalid curriculum coordinates")
    if family not in FAMILIES:
        raise ValueError("Family must belong to the declared split registry")
    identity = f"{family.name}/{count}/{street}/{index}"
    # Seed strings avoid integer-offset collisions as the dataset expands.
    rng = random.Random(identity)
    for attempt in range(512):
        stack = family.stack * rng.uniform(.9, 1.1)
        stacks = {p: stack * (1 + .3 * p if family.asymmetric else 1) for p in range(count)}
        root = HandState.start(stacks, index % count, rng)
        while root.street != street:
            root.act("CALL" if root.to_call() else "CHECK")
        if street == "PREFLOP" or board_texture(root.board) == family.texture:
            break
    else:
        raise ValueError(f"Texture generation exhausted its attempt budget: {identity}")
    for _ in range(family.raises):
        raises = [a for a in legal_actions(root) if a.category == "RAISE" and a.action_id != "ALL_IN"]
        if not raises:
            raise ValueError(f"Required betting context unavailable: {identity}")
        raises[0].apply(root)
    available = [c for c in range(52) if c not in root.board]
    factors = {}
    for p, player in root.players.items():
        hands = {combo(player.cards)}
        while len(hands) < family.width:
            hands.add(combo(tuple(rng.sample(available, 2))))
        # Varied exact-combination probabilities, without hand-strength action rules.
        factors[p] = HandRange({hand: rng.uniform(.01, 1) ** (1 + index % 3) for hand in sorted(hands)})
    policy = (UniformLegalPolicy() if family.profile == "uniform" else
              category_policy({"FOLD": .1, "CHECK": 1, "CALL": 2, "RAISE": .2, "ALL_IN": .1}, "passive-continuation-v1")
              if family.profile == "passive" else
              category_policy({"FOLD": .2, "CHECK": .3, "CALL": .5, "RAISE": 2, "ALL_IN": .5}, "aggressive-continuation-v1"))
    return Scenario(identity, family.name, family.split, root, JointRanges(factors, tuple(root.board)), policy)


def curriculum(roots_per_family_street: int):
    if type(roots_per_family_street) is not int or roots_per_family_street < 1:
        raise ValueError("Positive root count required")
    for family in FAMILIES:
        for count in (2, 3):
            for street in ("PREFLOP", "FLOP", "TURN", "RIVER"):
                for index in range(roots_per_family_street):
                    yield scenario(count, street, family, index)


def coordinates(family: Family, count: int, street: str, index: int) -> dict:
    if family not in FAMILIES or count not in (2, 3) or street not in ("PREFLOP", "FLOP", "TURN", "RIVER") or type(index) is not int or index < 0:
        raise ValueError("Invalid curriculum coordinates")
    return {"id": f"{family.name}/{count}/{street}/{index}", "family": family.name, "split": family.split,
            "player_count": count, "street": street, "index": index, "range_width": family.width,
            "profile": family.profile, "texture": family.texture, "base_stack_bb": family.stack,
            "raises": family.raises, "asymmetric": family.asymmetric}


def manifest(path: Path, roots_per_family_street: int) -> dict:
    """Plan thousands of roots without solving them or materializing their ranges."""
    if type(roots_per_family_street) is not int or roots_per_family_street < 1:
        raise ValueError("Positive root count required")
    rows = [coordinates(f, count, street, index)
            for f in FAMILIES for count in (2, 3) for street in ("PREFLOP", "FLOP", "TURN", "RIVER")
            for index in range(roots_per_family_street)]
    result = {"schema": "heldout-family-curriculum-v1", "roots": rows, "count": len(rows),
              "split_counts": dict(Counter(r["split"] for r in rows)), "generation": "not executed"}
    atomic_bytes(path, json.dumps(result, separators=(",", ":")).encode())
    return result


def fingerprint(item: Scenario) -> str:
    return hashlib.sha256(json.dumps(public_root(item.root), sort_keys=True).encode()).hexdigest()
