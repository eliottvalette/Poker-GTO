"""Preserve explicitly selected atomic checkpoints without touching the trainer."""
from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
import zipfile


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + '.partial')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def validate_envelope(path: Path) -> dict:
    """Validate streamed bytes without importing PyTorch or reconstructing replay."""
    with zipfile.ZipFile(path) as archive:
        if set(archive.namelist()) != {'payload.pt', 'checkpoint.json'}:
            raise ValueError(f'Unsupported checkpoint envelope: {path}')
        manifest = json.loads(archive.read('checkpoint.json'))
        if manifest['format'] != 'streamed-checkpoint-v1':
            raise ValueError(f'Unsupported checkpoint format: {path}')
        payload = hashlib.sha256()
        with archive.open('payload.pt') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                payload.update(block)
        if payload.hexdigest() != manifest['sha256']:
            raise ValueError(f'Checkpoint payload checksum mismatch: {path}')
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(block)
    return {'sha256': checksum.hexdigest(), 'payload_sha256': payload.hexdigest(),
            'bytes': path.stat().st_size}


def archive_once(source: Path, destination: Path, iterations: tuple[int, ...],
                 run_id: str) -> dict:
    contract = json.loads((source / 'selfplay.json').read_text())
    if contract['run_id'] != run_id or contract['track'] != 'hu':
        raise ValueError('Archive source run identity or track changed')
    pointer = json.loads((source / 'resume.json').read_text())
    status = {'run_id': run_id, 'observed_iteration': pointer['iteration'],
              'checked_at': datetime.now(timezone.utc).isoformat(), 'milestones': {}}
    for iteration in iterations:
        name = f'iteration_{iteration:06d}.pt'
        target = destination / name
        record = target.with_suffix('.json')
        if record.exists():
            saved = json.loads(record.read_text())
            if (saved['run_id'] != run_id or saved['iteration'] != iteration
                    or not target.is_file() or target.stat().st_size != saved['bytes']):
                raise ValueError(f'Invalid archived milestone: {target}')
            status['milestones'][str(iteration)] = 'saved'
            continue
        pending = target.with_suffix('.pt.partial')
        if not pending.exists() and not target.exists():
            try:
                # Numbered checkpoints are published by atomic replacement. A hard
                # link survives trainer unlink/rotation without copying its payload.
                os.link(source / 'checkpoints' / name, pending)
            except FileNotFoundError:
                if pointer['iteration'] >= iteration + 10:
                    raise FileNotFoundError(f'Milestone {iteration} missed; no archive or source checkpoint')
                status['milestones'][str(iteration)] = 'pending'
                continue
        path = pending if pending.exists() else target
        evidence = validate_envelope(path)
        if path != target:
            pending.replace(target)
        write_json(record, {**evidence, 'iteration': iteration, 'run_id': run_id,
                            'source': str(source / 'checkpoints' / name),
                            'saved_at': datetime.now(timezone.utc).isoformat(),
                            'validation': 'Envelope checksum; iteration from immutable numbered filename'})
        status['milestones'][str(iteration)] = 'saved'
        print(f'Archived HU {iteration}: {evidence["sha256"]}', flush=True)
    status['complete'] = all(value == 'saved' for value in status['milestones'].values())
    write_json(destination / 'status.json', status)
    return status


def watch(config_path: Path) -> None:
    config = json.loads(config_path.read_text())
    source, destination = Path(config['source']), Path(config['destination'])
    iterations = tuple(config['iterations'])
    if (not iterations or tuple(sorted(set(iterations))) != iterations
            or any(type(i) is not int or i <= 0 or i % 5 for i in iterations)):
        raise ValueError('Milestones must be explicit increasing checkpoint multiples of five')
    if not destination.is_dir() or source.stat().st_dev != destination.stat().st_dev:
        raise ValueError('Existing source/archive directories must share a filesystem')
    deadline = datetime.fromisoformat(config['deadline_utc'])
    with (destination / '.archive.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            try:
                if datetime.now(timezone.utc) >= deadline:
                    raise TimeoutError('Archive deadline reached before all milestones were saved')
                result = archive_once(source, destination, iterations, config['run_id'])
                if result['complete']:
                    return
            except Exception as error:
                write_json(destination / 'error.json', {
                    'error': f'{type(error).__name__}: {error}',
                    'at': datetime.now(timezone.utc).isoformat()})
                raise
            time.sleep(30)


if __name__ == '__main__':
    watch(Path(os.environ['GTO_ARCHIVE_CONFIG']))
