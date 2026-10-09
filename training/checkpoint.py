"""Atomic, checksummed primitive/tensor checkpoints; strict schema, no pickle objects."""
from __future__ import annotations
from dataclasses import asdict, fields
import hashlib
import io
import json
import zipfile
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
from training.runtime_storage import metric_count

CHECKPOINT_VERSION = 4
_SAMPLE_FIELDS = tuple(f.name for f in fields(TrainingSample) if f.name != 'state')
_OBSERVATION_FIELDS = tuple(f.name for f in fields(NeuralObservation))
CONTRACT = {"checkpoint": CHECKPOINT_VERSION, "advantage_layout": "shared_per_player_count", "state": STATE_VERSION, "feature": FEATURE_SCHEMA_VERSION,
            "memory": MEMORY_VERSION, "action_schema": ACTION_SCHEMA_VERSION, "evaluation_schema": 2, "actions": list(ACTION_IDS), "model": MODEL_ARCHITECTURE,
            "objective": "hand_chip_delta", "root_sampler": ROOT_SAMPLER_VERSION, "traversal_modes": ["external_sampling", "outcome_sampling"],
            "external_strategy_collectors": {"hu": "opponent_nodes", "3max": "partial_enumeration"}}


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
    """Stream a checksummed envelope through disk instead of multiple RAM buffers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent, prefix=".checkpoint-") as temporary:
        payload = Path(temporary) / "payload.pt"
        torch.save(raw, payload)
        checksum = hashlib.sha256()
        with payload.open("rb") as file:
            for block in iter(lambda: file.read(1024 * 1024), b""):
                checksum.update(block)
        envelope = Path(temporary) / "envelope.pt"
        with zipfile.ZipFile(envelope, "w", compression=zipfile.ZIP_STORED) as archive:
            archive.write(payload, "payload.pt")
            archive.writestr("checkpoint.json", json.dumps({"format": "streamed-checkpoint-v1", "sha256": checksum.hexdigest()}))
        with envelope.open("rb") as file:
            os.fsync(file.fileno())
        os.replace(envelope, path)


def read_checkpoint(path: Path) -> dict:
    try:
        with zipfile.ZipFile(path) as archive:
            streamed = "checkpoint.json" in archive.namelist()
            if streamed:
                if set(archive.namelist()) != {"checkpoint.json", "payload.pt"}:
                    raise ValueError("Unexpected checkpoint envelope members")
                manifest = json.loads(archive.read("checkpoint.json"))
                if set(manifest) != {"format", "sha256"} or manifest["format"] != "streamed-checkpoint-v1":
                    raise ValueError("Invalid streamed checkpoint manifest")
                # Disk-backed staging also bounds the reader's serialized-byte overhead.
                with tempfile.TemporaryFile() as payload, archive.open("payload.pt") as source:
                    checksum = hashlib.sha256()
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        checksum.update(block)
                        payload.write(block)
                    if checksum.hexdigest() != manifest["sha256"]:
                        raise ValueError("Checksum mismatch")
                    payload.seek(0)
                    raw = torch.load(payload, map_location="cpu", weights_only=True)
        if not streamed:
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
    from ml.stratified_memory import ProtectedReplay
    if isinstance(memory, ProtectedReplay):
        return {"allocation": "opening-protected-v1", "strata": {
            name: pack_memory(m) for name, m in memory.memories.items()}}
    return {"capacity": memory.capacity, "kind": memory.kind, "objective": memory.objective,
            "byte_budget": memory.byte_budget, "seen": memory.seen, "rng": memory.rng.getstate(),
            "traversal_mode": memory.traversal_mode, "samples": [_pack_sample(s) for s in memory.samples]}


def _pack_sample(sample: TrainingSample) -> dict:
    """Same primitive schema as asdict, without recursively copying immutable scalars."""
    result = {name: getattr(sample, name) for name in _SAMPLE_FIELDS}
    state = {name: getattr(sample.state, name) for name in _OBSERVATION_FIELDS}
    # Valid compact observations contain flat sequences; keep mutable inputs isolated.
    for name in ('cards', 'legal_mask'):
        if isinstance(state[name], list):
            state[name] = state[name].copy()
    for name in ('numeric_data', 'history_data'):
        if not isinstance(state[name], bytes):
            import copy
            state[name] = copy.deepcopy(state[name])
    if isinstance(result['target'], list):
        result['target'] = result['target'].copy()
    result['state'] = state
    return result


def unpack_memory(raw: dict) -> ReservoirMemory:
    if "allocation" in raw:
        from ml.stratified_memory import ProtectedReplay
        if raw["allocation"] != "opening-protected-v1" or set(raw["strata"]) != {"opening", "other"}:
            raise ValueError("Unsupported replay allocation schema")
        result = ProtectedReplay.__new__(ProtectedReplay)
        result.memories = {name: unpack_memory(m) for name, m in raw["strata"].items()}
        if any(not isinstance(m, ReservoirMemory) for m in result.memories.values()):
            raise ValueError("Nested replay allocations are unsupported")
        if (len({m.kind for m in result.memories.values()}) != 1
                or any(m.traversal_mode not in (None, "external_sampling") for m in result.memories.values())):
            raise ValueError("Incompatible protected replay strata")
        return result
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


def _checkpoint_average_payload(source: Path, track_name: str) -> dict:
    """Extract inference state without reconstructing replay, samplers or a runner.

    The current envelope still requires full primitive deserialization. Replay
    player counts are checked directly; training-resume validation remains owned
    by TrainingRunner.load_checkpoint.
    """
    import struct
    from training.config import config_hash, validate_config
    from ml.deep_cfr import average_policy_payload
    from features.neural import NEURAL_NUMERIC_NAMES

    counts = {"hu": 2, "3max": 3}
    if track_name not in counts:
        raise ValueError(f"Unknown policy track: {track_name}")
    raw = read_checkpoint(source)
    config = validate_config(raw["config"])
    if config_hash(config) != raw["config_hash"]:
        raise ValueError(f"Checkpoint configuration hash mismatch at {source}")
    iteration = raw["iteration"]
    if (set(raw["tracks"]) != {track_name} or type(iteration) is not int or iteration < 1
            or len(raw["metrics"]) != metric_count(raw)
            or {name for name in counts if config[name]["enabled"]} != {track_name}):
        raise ValueError(f"No trained dedicated {track_name} policy in {source}")
    track = raw["tracks"][track_name]
    count = counts[track_name]
    if (track["version"] != iteration or len(track["metrics"]) != metric_count(raw)
            or track["seed"] != config["seed"] + count
            or track["traversal_mode"] != config["traversal_mode"]
            or track["average_weights"] is None):
        raise ValueError(f"Inconsistent policy version/metrics/mode/weights at {source}")
    memory = track["strategy_memory"]
    if "allocation" in memory:
        if memory["allocation"] != "opening-protected-v1" or set(memory["strata"]) != {"opening", "other"}:
            raise ValueError(f"Invalid strategy allocation at {source}")
        memories = list(memory["strata"].values())
    else:
        memories = [memory]
    samples = 0
    offset = NEURAL_NUMERIC_NAMES.index("player_count") * 8
    for memory in memories:
        if (memory["kind"] != "strategy" or memory["objective"] != "hand_chip_delta"
                or memory["traversal_mode"] != track["traversal_mode"]
                or len(memory["samples"]) != min(memory["seen"], memory["capacity"])):
            raise ValueError(f"Invalid strategy replay metadata at {source}")
        for row in memory["samples"]:
            state = row["state"]
            if (state["version"] != STATE_VERSION or state["feature_version"] != FEATURE_SCHEMA_VERSION
                    or state["objective"] != "hand_chip_delta"
                    or len(state["numeric_data"]) != 8 * len(NEURAL_NUMERIC_NAMES)
                    or round(struct.unpack_from("<d", state["numeric_data"], offset)[0] * 3) != count
                    or row["player"] not in range(count) or not 1 <= row["iteration"] <= iteration):
                raise ValueError(f"Foreign/future strategy sample at {source}")
            samples += 1
    if not samples:
        raise ValueError(f"No strategy coverage at {source}")
    payload = average_policy_payload(weights=track["average_weights"], iteration=iteration,
        objective="hand_chip_delta", player_count=count, traversal_mode=track["traversal_mode"],
        metrics=track["metrics"], supported_player_counts=[count])
    return payload


def export_checkpoint_average(source: Path, destination: Path, track_name: str) -> int:
    """Release the deserialized replay before saving and validating inference state."""
    from ml.deep_cfr import NeuralAveragePolicy
    payload = _checkpoint_average_payload(source, track_name)
    torch.save(payload, destination)
    with torch.random.fork_rng(devices=[]):
        NeuralAveragePolicy(destination)
    return payload["iteration"]
