"""Verify a stopped v3-to-v4 migration before activating the new trainer."""
from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch

from training.checkpoint import atomic_bytes, read_checkpoint
from training.runner import TrainingRunner


def digest(value: Any) -> str:
    result = hashlib.sha256()

    def visit(item: Any) -> None:
        result.update(type(item).__name__.encode() + b':')
        if isinstance(item, torch.Tensor):
            result.update(str((item.dtype, tuple(item.shape))).encode())
            result.update(item.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=repr):
                visit(key)
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            for child in item:
                visit(child)
        elif isinstance(item, (set, frozenset)):
            for child_digest in sorted(digest(child) for child in item):
                result.update(child_digest.encode())
        elif isinstance(item, bytes):
            result.update(item)
        elif item is None or isinstance(item, (str, int, float, bool)):
            result.update(repr(item).encode())
        else:
            raise TypeError(f'Unsupported checkpoint primitive: {type(item)}')
        result.update(b'\0')

    visit(value)
    return result.hexdigest()


def verify(directory: Path, track: str) -> dict:
    target = directory / 'selfplay-v4'
    contract = json.loads((target / 'selfplay.json').read_text())
    source = Path(contract['migration']['source'])
    raw = read_checkpoint(source)
    expected = {key: digest(value) for key, value in raw.items() if key not in ('config', 'config_hash', 'metadata')}
    del raw
    gc.collect()
    raw = read_checkpoint(target / 'checkpoint.pt')
    actual = {key: digest(value) for key, value in raw.items() if key not in ('config', 'config_hash', 'metadata')}
    if actual != expected:
        raise ValueError(f'Migration changed learning state: {track}')
    iteration = raw['iteration']
    del raw
    gc.collect()
    restored = TrainingRunner.load_checkpoint(target / 'checkpoint.pt')
    if restored.iteration != iteration or set(restored.solvers) != {track}:
        raise ValueError('Migrated checkpoint cannot restore its dedicated training state')
    report = {'track': track, 'iteration': iteration, 'run_id': contract['run_id'],
              'source': str(source), 'preserved_field_sha256': actual,
              'learning_state_equal': True, 'full_runner_reload': True,
              'minimum_unchanged_iteration': contract['migration']['fixed_hyperparameters_until_iteration']}
    atomic_bytes(target / 'migration-verification.json', (json.dumps(report, indent=2) + '\n').encode())
    return report


if __name__ == '__main__':
    torch.set_num_threads(1)
    print(json.dumps(verify(Path(os.environ['GTO_DATA_DIR']), os.environ['GTO_TRACK'])), flush=True)
