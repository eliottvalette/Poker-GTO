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
RUN_DIRECTORY = 'selfplay-v4'


def configuration(track: str, directory: Path) -> dict:
    if track not in ('hu', '3max'):
        raise ValueError(f'Unknown self-play track: {track}')
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / 'configs' / f'train_{track}_selfplay.json')
    config['output_dir'] = str(directory.resolve())
    return config


def migrate_v4(directory: Path, track: str, *, source_directory: Path) -> dict:
    """Preserve complete v3 state and change only the declared v4 learning budget."""
    import copy
    import hashlib
    import tempfile
    import uuid
    from training.checkpoint import read_checkpoint, write_checkpoint
    from training.config import config_hash
    from training.runner import source_metadata

    directory, source_directory = directory.resolve(), source_directory.resolve()
    if directory.name != RUN_DIRECTORY or source_directory.name != 'selfplay-v3':
        raise ValueError('Budget migration requires explicit selfplay-v3 -> selfplay-v4 directories')
    if directory.exists():
        if not (directory / 'selfplay.json').is_file():
            raise FileExistsError(f'Unowned v4 destination: {directory}')
        return initialize(directory, track)
    config = configuration(track, directory)
    source_contract = json.loads((source_directory / 'selfplay.json').read_text())
    pointer = json.loads((source_directory / 'resume.json').read_text())
    relative = Path(pointer['path'])
    if relative.is_absolute() or '..' in relative.parts or not (
        relative == Path('checkpoint.pt') or relative.parent == Path('checkpoints')
        and relative.name.startswith('iteration_') and relative.suffix == '.pt'
    ):
        raise ValueError('Invalid v3 resume pointer')
    source = source_directory / relative
    torch.set_num_threads(1)
    # Full runner validation checks replay, RNG, sampler and model contracts.
    restored = TrainingRunner.load_checkpoint(source)
    if restored.iteration != pointer['iteration'] or restored.iteration < 1 or set(restored.solvers) != {track}:
        raise ValueError('v3 checkpoint iteration or dedicated track mismatch')
    del restored
    raw = read_checkpoint(source)
    before = raw['config']
    if (raw['iteration'] != pointer['iteration'] or source_contract['engine'] != ENGINE
            or source_contract['track'] != track or source_contract['config'] != before
            or source_contract['run_id'] != raw['metadata']['run_id']):
        raise ValueError('v3 source contract/checkpoint mismatch; stop training before migrating')
    if (before[track]['traversals_per_player'] != 64
            or before['fit_schedule']['advantage_max_updates'] != 2048
            or before['fit_schedule']['average_max_updates'] != 2048
            or before['fit_schedule']['average_every'] != 1
            or before['root_policy'] != 'current_advantage'
            or before[track]['advantage_capacity'] != 50000
            or before['strategy_capacity'] != 50000 or 'street_networks' in before):
        raise ValueError('v4 migration requires the declared shared-network v3 production budget')
    expected = copy.deepcopy(before)
    expected['output_dir'] = str(directory)
    expected[track]['traversals_per_player'] = 256
    expected['fit_schedule'].update(advantage_max_updates=1024, average_max_updates=1024, average_every=3)
    if config != expected:
        raise ValueError('v4 migration cannot change settings outside traversals and fitting cadence/caps')
    checksum = hashlib.sha256()
    with source.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(block)
    raw['config'], raw['config_hash'] = config, config_hash(config)
    migration = {'source': str(source), 'source_sha256': checksum.hexdigest(),
                 'source_run_id': raw['metadata']['run_id'], 'source_iteration': raw['iteration'],
                 'before': {'traversals_per_player': 64, 'fit_schedule': before['fit_schedule']},
                 'after': {'traversals_per_player': 256, 'fit_schedule': config['fit_schedule']},
                 'fixed_hyperparameters_until_iteration': raw['iteration'] + 500,
                 'exceptions': 'Technical incidents only; continue running beyond the observation horizon'}
    raw['metadata'] = {**raw['metadata'], **source_metadata(), 'run_id': uuid.uuid4().hex,
                       'network_fit': dict(config['fit_schedule']), 'traversal_budget_migration': migration}
    contract = {'engine': ENGINE, 'track': track, 'config': config,
                'run_id': raw['metadata']['run_id'], 'created_at': datetime.now(timezone.utc).isoformat(),
                'migration': migration}
    # All learning state, including both global RNGs and every track field, is
    # serialized unchanged. Publish the complete destination as one directory.
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=directory.parent, prefix='.selfplay-v4-') as temporary:
        staged = Path(temporary) / RUN_DIRECTORY
        staged.mkdir()
        write_checkpoint(staged / 'checkpoint.pt', raw)
        atomic_bytes(staged / 'resume.json', encoded({'path': 'checkpoint.pt', 'iteration': raw['iteration']}))
        atomic_bytes(staged / 'selfplay.json', encoded(contract))
        staged.rename(directory)
    return contract


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
