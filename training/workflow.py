"""Independent timed sessions and reviewed, atomic policy/ONNX publication."""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
from typing import Iterator

from infoset import observe
from ml.deep_cfr import NeuralAveragePolicy
from ml.export_onnx import export_average_policy
from training.checkpoint import atomic_bytes
from training.config import load_config
from training.runner import TrainingRunner

REPOSITORY = Path(__file__).resolve().parents[1]
TRACKS = {"3max": 3, "hu": 2}
POLICY_ROOT = REPOSITORY / "policy"
UI_POLICY_ROOT = REPOSITORY / "ui" / "public" / "policy"
CATALOG = UI_POLICY_ROOT / "index.json"
CATALOG_VERSION = 1


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def config_for(track: str) -> dict:
    if track not in TRACKS:
        raise ValueError(f"Unknown training track: {track}; expected {tuple(TRACKS)}")
    return load_config(REPOSITORY / "configs" / f"train_{track}.json")


def candidate_checkpoint(track: str) -> Path:
    config = config_for(track)
    directory = Path(config["output_dir"])
    if not directory.is_absolute():
        directory = REPOSITORY / directory
    return directory / "checkpoint.pt"


def catalog_snapshot() -> tuple[dict, str | None]:
    if not CATALOG.exists():
        return {"version": CATALOG_VERSION, "active": {}, "exports": {}}, None
    payload = CATALOG.read_bytes()
    raw = json.loads(payload)
    if not isinstance(raw, dict) or set(raw) != {"version", "active", "exports"} or raw["version"] != CATALOG_VERSION:
        raise ValueError(f"Invalid policy catalog schema at {CATALOG}")
    for group in ("active", "exports"):
        if not isinstance(raw[group], dict) or set(raw[group]) - set(TRACKS):
            raise ValueError(f"Invalid catalog tracks at {CATALOG}: {group}")
        for track, entry in raw[group].items():
            fields = {"release_id", "iteration"} if group == "active" else {"bundle_id", "checkpoint_sha256", "iteration"}
            if not isinstance(entry, dict) or set(entry) != fields or type(entry["iteration"]) is not int or entry["iteration"] < 1:
                raise ValueError(f"Invalid catalog entry at {CATALOG}: {group}/{track}")
            for key in fields - {"iteration"}:
                if not isinstance(entry[key], str) or not re.fullmatch('[a-f0-9]{64}', entry[key]):
                    raise ValueError(f"Invalid artifact identifier at {CATALOG}: {group}/{track}/{key}")
    return raw, hashlib.sha256(payload).hexdigest()


def read_catalog() -> dict:
    return catalog_snapshot()[0]


def validate_bundle(directory: Path) -> dict:
    manifest = json.loads((directory / "bundle.json").read_text())
    if not isinstance(manifest, dict) or manifest.get("version") != 1 or not isinstance(manifest.get("files"), dict):
        raise ValueError(f"Invalid bundle manifest at {directory}")
    expected = set(manifest["files"]) | {"bundle.json"}
    if {path.name for path in directory.iterdir()} != expected:
        raise ValueError(f"Incomplete or unexpected bundle files at {directory}")
    for name, checksum in manifest["files"].items():
        if Path(name).name != name or digest(directory / name) != checksum:
            raise ValueError(f"Bundle checksum mismatch at {directory / name}")
    return manifest


def active_checkpoint(track: str) -> Path | None:
    config_for(track)
    entry = read_catalog()["active"].get(track)
    if entry is None:
        return None
    directory = POLICY_ROOT / "releases" / track / entry["release_id"]
    manifest = validate_bundle(directory)
    if manifest.get("track") != track or manifest.get("iteration") != entry["iteration"]:
        raise ValueError(f"Active policy/catalog mismatch at {directory}")
    path = directory / "checkpoint.pt"
    if digest(path) != entry["release_id"]:
        raise ValueError(f"Active checkpoint identifier mismatch at {path}")
    return path


@contextmanager
def exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as file:
        try:
            fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another operation holds {path}") from error
        try:
            yield
        finally:
            fcntl.flock(file, fcntl.LOCK_UN)


def train_track(track: str, seconds: float) -> dict:
    config = config_for(track)
    checkpoint = candidate_checkpoint(track)
    with exclusive_lock(POLICY_ROOT / ".compute.lock"), exclusive_lock(checkpoint.parent / ".training.lock"):
        if checkpoint.exists():
            source = checkpoint
            origin = "training checkpoint"
        else:
            source = active_checkpoint(track)
            origin = "active policy checkpoint" if source else "fresh initialization"
        runner = TrainingRunner.load_checkpoint(source, config) if source else TrainingRunner(config)
        if set(runner.solvers) != {track}:
            raise ValueError(f"Session {track} requires a dedicated checkpoint: {source}")
        print(f"[{track}] {origin}; iteration={runner.iteration}; duration={seconds / 60:g} minutes", flush=True)
        print(f"[{track}] checkpoint={checkpoint}; workers={config['workers']}; traversals/iteration={TRACKS[track] * config[track]['traversals_per_player']}", flush=True)
        result = runner.run_for(seconds, checkpoint)
        print(f"[{track}] saved iteration {runner.iteration} at {checkpoint}", flush=True)
        return result


def seal_bundle(directory: Path, metadata: dict) -> None:
    files = {path.name: digest(path) for path in directory.iterdir() if path.is_file()}
    payload = {"version": 1, **metadata, "files": files}
    (directory / "bundle.json").write_text(json.dumps(payload, indent=2) + "\n")
    validate_bundle(directory)


@dataclass
class PreparedMigration:
    action: str
    tracks: tuple[str, ...]
    bundles: list[tuple[Path, Path]]
    catalog: dict
    prior_catalog_sha256: str | None
    summary: list[dict]

    def close(self) -> None:
        for temporary, _ in self.bundles:
            if temporary.exists():
                shutil.rmtree(temporary)

    def publish(self) -> None:
        with exclusive_lock(POLICY_ROOT / ".publication.lock"):
            current = digest(CATALOG) if CATALOG.exists() else None
            if current != self.prior_catalog_sha256:
                raise RuntimeError(f"Policy catalog changed since preparation: {CATALOG}; prepare again")
            for temporary, destination in self.bundles:
                expected = validate_bundle(temporary)
                if destination.exists():
                    if validate_bundle(destination) != expected:
                        raise ValueError(f"Existing immutable release differs: {destination}")
                else:
                    temporary.rename(destination)
            # One pointer controls active checkpoints and public ONNX models.
            atomic_bytes(CATALOG, (json.dumps(self.catalog, indent=2) + "\n").encode())
        self.close()


def prepare_migration(action: str, tracks: tuple[str, ...]) -> PreparedMigration:
    if action not in ("activate", "export", "both") or not tracks or len(set(tracks)) != len(tracks) or set(tracks) - set(TRACKS):
        raise ValueError(f"Invalid migration request: action={action}, tracks={tracks}")
    if action in ("export", "both") and set(tracks) != set(TRACKS):
        raise ValueError("UI export requires both 3max and HU candidate checkpoints")
    catalog, previous = catalog_snapshot()
    prepared = PreparedMigration(action, tracks, [], catalog, previous, [])
    snapshots: dict[str, Path] = {}
    try:
        for track in tracks:
            source = candidate_checkpoint(track)
            if not source.is_file():
                raise FileNotFoundError(f"Candidate checkpoint missing: {source}; run the corresponding training script first")
            parent = POLICY_ROOT / "releases" / track
            parent.mkdir(parents=True, exist_ok=True)
            temporary = Path(tempfile.mkdtemp(prefix=".prepared-", dir=parent))
            # Pin the checkpoint inode before reading models or computing hashes.
            shutil.copyfile(source, temporary / "checkpoint.pt")
            checkpoint_id = digest(temporary / "checkpoint.pt")
            destination = parent / checkpoint_id
            prepared.bundles.append((temporary, destination))
            runner = TrainingRunner.load_checkpoint(temporary / "checkpoint.pt", config_for(track))
            if set(runner.solvers) != {track} or runner.iteration < 1:
                raise ValueError(f"No trained dedicated {track} policy in {source}")
            average = temporary / f"average_{track}.pt"
            runner.solvers[track].export_average(average)
            policy = NeuralAveragePolicy(average)
            if policy.supported_player_counts != (TRACKS[track],):
                raise ValueError(f"Policy player-count mismatch: {average}")
            seal_bundle(temporary, {"kind": "active_policy", "track": track, "iteration": runner.iteration,
                                    "checkpoint_sha256": checkpoint_id})
            snapshots[track] = temporary
            prepared.summary.append({"track": track, "iteration": runner.iteration, "source": str(source),
                                     "checkpoint_sha256": checkpoint_id, "release": str(destination)})
            if action in ("activate", "both"):
                catalog["active"][track] = {"release_id": checkpoint_id, "iteration": runner.iteration}
        if action in ("export", "both"):
            sources = {row["track"]: row["checkpoint_sha256"] for row in prepared.summary}
            bundle_id = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
            parent = UI_POLICY_ROOT / "releases"
            parent.mkdir(parents=True, exist_ok=True)
            temporary = Path(tempfile.mkdtemp(prefix=".prepared-", dir=parent))
            destination = parent / bundle_id
            prepared.bundles.append((temporary, destination))
            for track, snapshot in snapshots.items():
                runner = TrainingRunner.load_checkpoint(snapshot / "checkpoint.pt", config_for(track))
                export_average_policy(snapshot / f"average_{track}.pt", temporary / f"average_{track}.onnx",
                                      temporary / f"average_{track}.json", observe(runner.probes[track][0][1]))
                catalog["exports"][track] = {"bundle_id": bundle_id, "checkpoint_sha256": sources[track],
                                             "iteration": runner.iteration}
            seal_bundle(temporary, {"kind": "onnx_export", "sources": sources})
        return prepared
    except BaseException:
        prepared.close()
        raise
