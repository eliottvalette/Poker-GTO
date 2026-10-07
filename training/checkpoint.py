"""Atomic, checksummed primitive/tensor checkpoints; strict schema, no pickle objects."""
from __future__ import annotations
from dataclasses import asdict
import hashlib
import io
import os
from pathlib import Path
import tempfile
import torch
from actions import ACTION_IDS, ACTION_SCHEMA_VERSION
from features import FEATURE_SCHEMA_VERSION
from features.neural import NeuralObservation
from infoset import STATE_VERSION
from ml.memory import MEMORY_VERSION, ReservoirMemory, TrainingSample
from ml.model import MODEL_ARCHITECTURE
from poker_game_expresso import ActionEvent, BlindLevel, HandPlayer, HandState
from tournament import TournamentState
from training.root_sampler import ROOT_SAMPLER_VERSION

CHECKPOINT_VERSION = 3
CONTRACT = {"checkpoint": CHECKPOINT_VERSION, "advantage_layout": "shared_per_player_count", "state": STATE_VERSION, "feature": FEATURE_SCHEMA_VERSION,
            "memory": MEMORY_VERSION, "action_schema": ACTION_SCHEMA_VERSION, "evaluation_schema": 2, "actions": list(ACTION_IDS), "model": MODEL_ARCHITECTURE,
            "objective": "hand_chip_delta", "root_sampler": ROOT_SAMPLER_VERSION, "traversal_modes": ["external_sampling", "outcome_sampling"],
            "external_strategy_collectors": {"hu": "opponent_nodes", "3max": "uniform_importance"}}


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "wb") as file:
            file.write(payload)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_checkpoint(path: Path, raw: dict) -> None:
    inner = io.BytesIO()
    torch.save(raw, inner)
    payload = inner.getvalue()
    outer = io.BytesIO()
    torch.save({"sha256": hashlib.sha256(payload).hexdigest(), "payload": payload}, outer)
    atomic_bytes(path, outer.getvalue())


def read_checkpoint(path: Path) -> dict:
    try:
        envelope = torch.load(path, map_location="cpu", weights_only=True)
        if set(envelope) != {"sha256", "payload"} or hashlib.sha256(envelope["payload"]).hexdigest() != envelope["sha256"]:
            raise ValueError("Checksum mismatch")
        raw = torch.load(io.BytesIO(envelope["payload"]), map_location="cpu", weights_only=True)
        if raw.get("contract") != CONTRACT:
            raise ValueError(f"Incompatible checkpoint schema: {raw.get('contract')}")
        return raw
    except Exception as error:
        raise ValueError(f"Invalid training checkpoint at {path}: {error}") from error


def pack_memory(memory: ReservoirMemory) -> dict:
    return {"capacity": memory.capacity, "kind": memory.kind, "objective": memory.objective,
            "byte_budget": memory.byte_budget, "seen": memory.seen, "rng": memory.rng.getstate(),
            "traversal_mode": memory.traversal_mode, "samples": [asdict(s) for s in memory.samples]}


def unpack_memory(raw: dict) -> ReservoirMemory:
    memory = ReservoirMemory(raw["capacity"], 0, raw["kind"], raw["objective"], raw["byte_budget"], raw["traversal_mode"])
    if type(raw["seen"]) is not int or raw["seen"] < 0 or len(raw["samples"]) != min(raw["seen"], memory.capacity):
        raise ValueError("Invalid checkpoint reservoir counts")
    for row in raw["samples"]:
        row = dict(row)
        row["state"] = NeuralObservation(**row["state"])
        memory.add(TrainingSample(**row))
    memory.seen = raw["seen"]
    memory.rng.setstate(raw["rng"])
    return memory


def pack_tournament(tournament: TournamentState | None) -> dict | None:
    if tournament is None:
        return None
    return {"stacks": tournament.stacks, "button": tournament.button, "hand_number": tournament.hand_number,
            "rng": tournament.rng.getstate(), "hand": asdict(tournament.hand) if tournament.hand else None}


def unpack_tournament(raw: dict | None) -> TournamentState | None:
    if raw is None:
        return None
    tournament = TournamentState(raw["stacks"], raw["button"], hand_number=raw["hand_number"])
    tournament.rng.setstate(raw["rng"])
    if raw["hand"] is not None:
        hand = dict(raw["hand"])
        hand["players"] = {i: HandPlayer(**p) for i, p in hand["players"].items()}
        hand["blinds"] = BlindLevel(**hand["blinds"])
        hand["history"] = [ActionEvent(**e) for e in hand["history"]]
        tournament.hand = HandState(**hand)
        tournament.hand.assert_invariants()
    tournament.assert_invariants()
    return tournament
