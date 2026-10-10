"""Bounded hourly evaluation delivery using the existing public Storage bucket."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import fcntl
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile

from ml.onnx_policy import OnnxPolicy
from training.poker_evaluation import EvaluationBudget, SUITE, evaluate_policy
from training.publication import SHA, TRACKS, StorageError, SupabaseStorage, encoded, validate_pointer

HISTORY_LIMIT = 720
HISTORY_BYTES = 8 * 1024**2
TRACE_BYTES = 32 * 1024**2


def read_json(storage: SupabaseStorage, key: str) -> dict | None:
    try:
        value = json.loads(storage.get(key))
    except StorageError as error:
        if error.status == 404:
            return None
        raise
    if not isinstance(value, dict):
        raise ValueError(f'Invalid JSON object: {key}')
    return value


def read_history(storage: SupabaseStorage, key: str) -> dict | None:
    try:
        payload = storage.get(key)
    except StorageError as error:
        if error.status == 404:
            return None
        raise
    with gzip.GzipFile(fileobj=io.BytesIO(payload)) as stream:
        raw = stream.read(HISTORY_BYTES + 1)
    if len(raw) > HISTORY_BYTES:
        raise ValueError('Decompressed evaluation history exceeds quota')
    return json.loads(raw)


def publish_history(storage: SupabaseStorage, track: str, history: dict, *, write: bool) -> None:
    data = gzip.compress(encoded(history), mtime=0) if write else storage.get(f'evaluation/{track}/history.json.gz')
    if write:
        storage.put(f'evaluation/{track}/history.json.gz', data, mutable=True)
        if read_history(storage, f'evaluation/{track}/history.json.gz') != history:
            raise ValueError('Evaluation history read-back mismatch')
    index = {'version': 1, 'track': track, 'sha256': hashlib.sha256(data).hexdigest(),
             'updated_at': history['updated_at']}
    key = f'evaluation/{track}/index.json'
    if read_json(storage, key) != index:
        storage.put(key, encoded(index), mutable=True)


def validate_history(value: dict, track: str) -> dict:
    if (value.get('version') != 1 or value.get('track') != track
            or not isinstance(value.get('runs'), list) or len(value['runs']) > HISTORY_LIMIT):
        raise ValueError('Invalid evaluation history')
    ids = set()
    for run in value['runs']:
        if (not isinstance(run, dict) or not SHA.fullmatch(run.get('release_id', ''))
                or run['release_id'] in ids or run.get('status') not in ('complete', 'budget_limited', 'failed')
                or not isinstance(run.get('iteration'), int) or run['iteration'] < 1):
            raise ValueError('Invalid or duplicate evaluation record')
        date = datetime.fromisoformat(run['evaluated_at'])
        if date.tzinfo is None:
            raise ValueError('Evaluation timestamps must include a timezone')
        ids.add(run['release_id'])
    return value


def retain_history(history: dict, now: datetime) -> dict:
    cutoff = now - timedelta(days=30)
    runs = sorted((r for r in history['runs'] if datetime.fromisoformat(r['evaluated_at']) >= cutoff),
                  key=lambda r: r['evaluated_at'])[-HISTORY_LIMIT:]
    result = {**history, 'runs': runs, 'updated_at': now.isoformat()}
    while len(encoded(result)) > HISTORY_BYTES and result['runs']:
        result['runs'].pop(0)
    if len(encoded(result)) > HISTORY_BYTES:
        raise ValueError('Evaluation metadata exceeds history quota')
    return result


def retain_traces(directory: Path, now: datetime) -> None:
    cutoff = now.timestamp() - 48 * 3600
    paths = sorted(directory.glob('*.json.gz'), key=lambda p: p.stat().st_mtime)
    size = sum(p.stat().st_size for p in paths)
    for path in paths:
        stat = path.stat()
        if stat.st_mtime < cutoff or size > TRACE_BYTES:
            size -= stat.st_size
            path.unlink()


def write_trace(path: Path, value: dict) -> None:
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix='.trace-')
    try:
        with os.fdopen(descriptor, 'wb') as target:
            target.write(gzip.compress(encoded(value), mtime=0))
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def evaluate_published(storage: SupabaseStorage, track: str, directory: Path,
                       budget: EvaluationBudget = EvaluationBudget()) -> dict | None:
    """One writer per track; never modifies the policy publication pointer."""
    if track not in TRACKS:
        raise ValueError('Unknown evaluation track')
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _evaluate(storage, track, directory, budget)


def _evaluate(storage: SupabaseStorage, track: str, directory: Path, budget: EvaluationBudget) -> dict | None:
    now = datetime.now(timezone.utc)
    retain_traces(directory, now)
    key = f'evaluation/{track}/history.json.gz'
    history = read_history(storage, key) or {'version': 1, 'track': track, 'runs': []}
    validate_history(history, track)
    compact = retain_history(history, now)
    pointer = read_json(storage, f'{track}/current.json')
    if pointer is None:
        print(f'[{track}] evaluation waiting for a published policy', flush=True)
        return None
    validate_pointer(pointer, track)
    # Retention continues even when training stops producing models.
    if any(r['release_id'] == pointer['release_id'] and r['suite'] == SUITE for r in compact['runs']):
        if compact['runs'] != history['runs']:
            publish_history(storage, track, compact, write=True)
        else:
            # Repairs an interrupted index write without rerunning poker evaluation.
            publish_history(storage, track, history, write=False)
        return None
    prefix = f'{track}/releases/{pointer["release_id"]}/average_{track}'
    model = storage.get(prefix + '.onnx')
    manifest = json.loads(storage.get(prefix + '.json'))
    if manifest.get('iteration') != pointer['iteration']:
        raise ValueError('Evaluation publication iteration mismatch')
    candidate = OnnxPolicy(model, manifest, TRACKS[track])
    reference_key = f'evaluation/{track}/{SUITE}/reference.json'
    reference = read_json(storage, reference_key)
    if reference is None:
        reference = {'release_id': pointer['release_id'], 'model_sha256': manifest['model_sha256'],
                     'iteration': pointer['iteration'], 'suite': SUITE}
        ref_prefix = f'evaluation/{track}/{SUITE}/reference'
        storage.ensure(ref_prefix + '.onnx', model)
        storage.ensure(ref_prefix + '.manifest.json', encoded(manifest))
        storage.ensure(reference_key, encoded(reference))
    if reference.get('suite') != SUITE or not SHA.fullmatch(reference.get('model_sha256', '')):
        raise ValueError('Invalid pinned evaluation reference')
    ref_prefix = f'evaluation/{track}/{SUITE}/reference'
    ref_model = storage.get(ref_prefix + '.onnx')
    if hashlib.sha256(ref_model).hexdigest() != reference['model_sha256']:
        raise ValueError('Pinned reference checksum mismatch')
    baseline = OnnxPolicy(ref_model, json.loads(storage.get(ref_prefix + '.manifest.json')), TRACKS[track])
    report, traces = evaluate_policy(candidate, baseline, TRACKS[track], budget)
    report.update(release_id=pointer['release_id'], model_sha256=manifest['model_sha256'],
                  iteration=pointer['iteration'], published_at=pointer['published_at'],
                  evaluated_at=datetime.now(timezone.utc).isoformat(), reference=reference,
                  is_reference=manifest['model_sha256'] == reference['model_sha256'])
    source = Path(__file__).resolve().parents[1] / 'deployment-source.json'
    report['source_commit'] = json.loads(source.read_text())['metadata']['source_git_commit'] if source.exists() else None
    write_trace(directory / f'{pointer["release_id"]}.json.gz', {'report': report, 'groups': traces})
    retain_traces(directory, datetime.now(timezone.utc))
    # Replace the same release when a versioned suite is deliberately changed.
    compact['runs'] = [r for r in compact['runs'] if r['release_id'] != pointer['release_id']] + [report]
    compact = retain_history(compact, datetime.now(timezone.utc))
    publish_history(storage, track, compact, write=True)
    print(f'[{track}] evaluated {pointer["iteration"]}: {report["status"]}, '
          f'{report["completed_groups"]} groups, {report["cpu_seconds"]:.1f}s CPU', flush=True)
    return report
