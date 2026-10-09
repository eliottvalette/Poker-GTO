"""Bounded operational diagnostics; learning replay and optimizer semantics are untouched."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import gzip
import json
import os
from pathlib import Path
import tempfile


@dataclass(frozen=True)
class RuntimeStorage:
    metrics_keep: int = 1
    checkpoints_keep: int = 3
    journal_keep: int = 256
    journal_bytes: int = 256 * 1024**2

    def __post_init__(self) -> None:
        if any(type(value) is not int or value < 1 for value in asdict(self).values()):
            raise ValueError("Storage retention limits must be positive integers")


def metric_count(raw: dict) -> int:
    """Legacy snapshots contain every metric; bounded snapshots declare retention."""
    policy = raw.get('runtime_storage')
    return raw['iteration'] if policy is None else min(raw['iteration'], RuntimeStorage(**policy).metrics_keep)


def archive_metric(directory: Path, row: dict, policy: RuntimeStorage) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"iteration_{row['iteration']:06d}.json.gz"
    descriptor, temporary = tempfile.mkstemp(dir=directory, prefix='.metric-')
    try:
        with os.fdopen(descriptor, 'wb') as file:
            with gzip.GzipFile(fileobj=file, mode='wb', mtime=0) as compressed:
                compressed.write(json.dumps(row, allow_nan=False, separators=(',', ':')).encode())
            file.flush()
            os.fsync(file.fileno())
        if Path(temporary).stat().st_size > policy.journal_bytes:
            raise ValueError(f"One metric exceeds the journal quota: {destination}")
        os.replace(temporary, destination)
        paths = sorted(directory.glob('iteration_*.json.gz'))
        size = sum(path.stat().st_size for path in paths)
        while len(paths) > policy.journal_keep or size > policy.journal_bytes:
            oldest = paths.pop(0)
            size -= oldest.stat().st_size
            oldest.unlink()
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def rotate_checkpoints(directory: Path, keep: int) -> None:
    """Only prune completed numbered checkpoints after a new save succeeds."""
    paths = sorted((path for path in directory.glob('iteration_*.pt')
                    if path.stem.removeprefix('iteration_').isdigit()),
                   key=lambda path: int(path.stem.removeprefix('iteration_')))
    for path in paths[:-keep]:
        path.unlink()
