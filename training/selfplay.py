"""Continuous full-hand Deep CFR service with explicit fresh-run ownership."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Callable

import torch

from training.checkpoint import atomic_bytes
from training.config import load_config
from training.publication import SupabaseStorage, encoded
from training.runner import TrainingRunner
from training.runtime_storage import RuntimeStorage
from training.vps import run_continuous

ENGINE = 'continuous_selfplay_v1'


def configuration(track: str, directory: Path) -> dict:
    if track not in ('hu', '3max'):
        raise ValueError(f'Unknown self-play track: {track}')
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / 'configs' / f'train_{track}_selfplay.json')
    config['output_dir'] = str(directory.resolve())
    return config


def initialize(directory: Path, track: str, *, source_directory: Path | None = None) -> dict:
    """Initialize only an empty run; reject different learning contracts on resume."""
    config = configuration(track, directory)
    contract = {'engine': ENGINE, 'track': track, 'config': config}
    contract_path = directory / 'selfplay.json'
    if contract_path.exists():
        saved = json.loads(contract_path.read_text())
        if {k: saved[k] for k in contract} != contract:
            raise ValueError('Self-play configuration changed; explicit run migration required')
        if not (directory / 'resume.json').is_file():
            raise FileNotFoundError(f'Missing self-play resume pointer: {directory}')
        return saved
    if directory.exists() and any(directory.iterdir()):
        raise ValueError(f'Nonempty directory without self-play contract: {directory}')
    torch.set_num_threads(1)
    if source_directory is None:
        runner = TrainingRunner(config)
    else:
        import uuid
        from training.runner import source_metadata
        pointer = json.loads((source_directory / 'resume.json').read_text())
        relative = Path(pointer['path'])
        if relative.is_absolute() or '..' in relative.parts or not (
            relative == Path('checkpoint.pt') or relative.parent == Path('checkpoints')
            and relative.name.startswith('iteration_') and relative.suffix == '.pt'
        ):
            raise ValueError('Invalid source checkpoint pointer for fitting-budget migration')
        source = source_directory / relative
        runner = TrainingRunner.load_checkpoint(source)
        if runner.iteration != pointer['iteration'] or set(runner.solvers) != {track}:
            raise ValueError('Source iteration or track mismatch for fitting-budget migration')
        before = runner.config
        expected = {**before, 'output_dir': config['output_dir'],
                    'advantage_epochs': config['advantage_epochs'],
                    'average_epochs': config['average_epochs'],
                    'fit_schedule': {**before['fit_schedule'],
                        'advantage_max_updates': config['fit_schedule']['advantage_max_updates'],
                        'average_max_updates': config['fit_schedule']['average_max_updates']}}
        if config != expected:
            raise ValueError('Fitting-budget migration cannot change replay, sampling, features or other learning settings')
        runner.config = config
        runner.metadata = {**runner.metadata, **source_metadata(), 'run_id': uuid.uuid4().hex,
                           'network_fit': dict(config['fit_schedule']),
                           'fitting_budget_migration': {'source': str(source),
                               'source_run_id': runner.metadata['run_id'],
                               'source_iteration': runner.iteration,
                               'before': before['fit_schedule'], 'after': config['fit_schedule']}}
    runner.enable_bounded_storage(RuntimeStorage(checkpoints_keep=2))
    runner.save_checkpoint(directory / 'checkpoint.pt')
    contract.update(run_id=runner.metadata['run_id'],
                    created_at=datetime.now(timezone.utc).isoformat())
    atomic_bytes(contract_path, encoded(contract))
    return contract


def run(directory: Path, track: str, ready: Callable[[], None]) -> None:
    initialize(directory, track)
    run_continuous(track, directory, workers=1, first_export_seconds=60, export_seconds=300,
                   storage=RuntimeStorage(checkpoints_keep=2), ready=ready)


def publish_status(storage: SupabaseStorage, directory: Path, track: str) -> None:
    contract = json.loads((directory / 'selfplay.json').read_text())
    if contract['engine'] != ENGINE or contract['track'] != track:
        raise ValueError('Self-play publication contract mismatch')
    path = directory / 'metrics.jsonl'
    if not path.exists():
        print(f'{track}: waiting for the first complete self-play iteration', flush=True)
        return
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if len(rows) != 1 or set(rows[0]['tracks']) != {track}:
        raise ValueError('Expected one bounded metric for the dedicated self-play track')
    row = rows[0]
    metric = row['tracks'][track]
    status = {'engine': ENGINE, 'track': track, 'run_id': contract['run_id'],
              'iteration': row['iteration'], 'iteration_limit': None,
              'updated_at': datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
              'published_at': datetime.now(timezone.utc).isoformat(),
              'iteration_seconds': row['wall_seconds'], 'traversals': metric['traversals'],
              'nodes': metric['nodes'], 'replay': metric['replay'],
              'advantage': metric['advantage'], 'average_policy': metric['average_policy'],
              'quality': 'experimental', 'objective': 'hand_chip_delta',
              'learning_objective': 'decision_balanced_v1', 'publication_gate': 'opening_health_v1'}
    selected = directory / 'exports/current.json'
    if selected.exists():
        from training.workflow import validate_bundle
        name = json.loads(selected.read_text())['directory']
        import re
        if not re.fullmatch(r'iteration_[0-9]+', name):
            raise ValueError('Invalid self-play export selection')
        seal = validate_bundle(directory / 'exports' / name)
        status['opening_health'] = seal.get('opening_health', {}).get(track)
    storage.put(f'selfplay/{track}/status.json', encoded(status), mutable=True)
