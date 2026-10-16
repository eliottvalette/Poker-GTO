"""Bounded resumable generation; uncertainty stays alongside every response label."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import fcntl
import math
import json
from pathlib import Path
import random
import time

from hybrid.adaptive_labels import LabelBudget, generate
from hybrid.knowledge_curriculum import FAMILIES, coordinates, scenario
from training.checkpoint import atomic_bytes


def _build(manifest_path: Path, output: Path, budget: LabelBudget, *, max_new_labels: int = 8,
          max_cpu_seconds: float = 120., max_storage_bytes: int = 256 * 1024 * 1024) -> dict:
    if (type(max_new_labels) is not int or max_new_labels < 1 or not math.isfinite(max_cpu_seconds)
            or max_cpu_seconds <= 0 or type(max_storage_bytes) is not int or max_storage_bytes < 1):
        raise ValueError("Positive label, CPU and storage limits required")
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    if manifest["schema"] != "heldout-family-curriculum-v1":
        raise ValueError("Unsupported curriculum manifest")
    if manifest["count"] != len(manifest["roots"]) or len({row["id"] for row in manifest["roots"]}) != manifest["count"]:
        raise ValueError("Manifest count or identities are inconsistent")
    from importlib.metadata import version
    import platform
    repository = Path(__file__).resolve().parents[1]
    dependencies = ("hybrid/adaptive_labels.py", "hybrid/knowledge_curriculum.py", "hybrid/label_dataset.py",
                    "hybrid/leaf_evaluation.py", "hybrid/policy_source.py", "hybrid/ranges.py", "hybrid/state.py",
                    "hybrid/knowledge.py", "poker_game_expresso.py", "actions.py", "infoset.py", "utils.py",
                    "tournament.py", "cfr_solver.py", "features/range_features.py")
    digest = hashlib.sha256()
    for name in dependencies:
        digest.update(name.encode())
        digest.update((repository / name).read_bytes())
    source_hash = digest.hexdigest()
    contract = {"schema": "adaptive-dataset-v1", "manifest_sha256": hashlib.sha256(raw).hexdigest(),
                "budget": json.loads(json.dumps(asdict(budget))), "generator_sha256": source_hash, "python": platform.python_version(), "treys": version("treys")}
    output.mkdir(parents=True, exist_ok=True)
    contract_path = output / "contract.json"
    if contract_path.exists():
        if json.loads(contract_path.read_text()) != contract:
            raise ValueError("Dataset contract changed; use a distinct explicit experiment")
    elif any(p.name != ".writer.lock" for p in output.iterdir()):
        raise ValueError("Existing dataset lacks its contract")
    else:
        atomic_bytes(contract_path, json.dumps(contract).encode())
    started = time.process_time()
    written, skipped, results = 0, 0, []
    expected_files = {hashlib.sha256(row["id"].encode()).hexdigest() + ".json" for row in manifest["roots"]} | {"contract.json", ".writer.lock"}
    if any(p.name not in expected_files for p in output.iterdir()):
        raise ValueError("Unexpected artifact in dataset directory")
    size = sum(p.stat().st_size for p in output.glob("*.json"))
    if size > max_storage_bytes:
        raise ValueError("Existing dataset exceeds storage limit")
    for row in manifest["roots"]:
        family = next((f for f in FAMILIES if f.name == row["family"]), None)
        if family is None or family.split != row["split"]:
            raise ValueError("Manifest violates family split ownership")
        if row != coordinates(family, row["player_count"], row["street"], row["index"]):
            raise ValueError("Manifest differs from its declared family coordinates")
        path = output / (hashlib.sha256(row["id"].encode()).hexdigest() + ".json")
        if path.exists():
            saved = json.loads(path.read_text())
            if saved["scenario"] != row or saved["label"]["budget"] != contract["budget"]:
                raise ValueError("Saved label provenance differs from its manifest")
            skipped += 1
            continue
        if written >= max_new_labels or time.process_time() - started >= max_cpu_seconds:
            break
        item = scenario(row["player_count"], row["street"], family, row["index"])
        deal = item.ranges.sample(random.Random(item.identity + "/holding"))
        label = generate(item.root, item.ranges, deal[item.root.current_player],
                         {p: item.policy for p in item.root.players}, budget)
        payload = {"scenario": row, "label": label}
        encoded = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()
        if size + len(encoded) > max_storage_bytes:
            raise ValueError("Dataset storage limit reached before writing the next label")
        atomic_bytes(path, encoded)
        size += len(encoded)
        written += 1
        status = label["estimate"]["status"] if label["estimate"] else "no_scheduled_estimate"
        results.append({"id": row["id"], "status": status, **label["work"]})
        print(f'{row["id"]}: {status}, {label["work"]["completed_worlds"]} worlds', flush=True)
    return {"new_labels": written, "existing_labels": skipped, "planned": manifest["count"],
            "complete": skipped + written == manifest["count"], "cpu_seconds": time.process_time() - started,
            "storage_bytes": size, "labels": results,
            "cpu_limit_scope": "checked between labels; each label also has explicit node and world limits"}


def build(manifest_path: Path, output: Path, budget: LabelBudget, *, max_new_labels: int = 8,
          max_cpu_seconds: float = 120., max_storage_bytes: int = 256 * 1024 * 1024) -> dict:
    """One writer per dataset; complete labels are reused only under identical provenance."""
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".writer.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Dataset already has an active writer: {output}") from error
        return _build(manifest_path, output, budget, max_new_labels=max_new_labels,
                      max_cpu_seconds=max_cpu_seconds, max_storage_bytes=max_storage_bytes)
