"""Importable continuous CPU training with bounded storage and live-model exports."""
from __future__ import annotations
from dataclasses import asdict
from contextlib import nullcontext
from typing import Callable, ContextManager
import json
import math
from pathlib import Path
import shutil
import tempfile
import time
import torch
from infoset import observe
from ml.export_onnx import export_average_policy
from training.checkpoint import atomic_bytes
from training.runner import TrainingRunner
from training.runtime_storage import RuntimeStorage


def prepare_runner(track: str, checkpoint: Path, output_dir: Path, *, workers: int = 1,
                   storage: RuntimeStorage = RuntimeStorage(),
                   measure: Callable[[str], ContextManager] | None = None) -> TrainingRunner:
    """Explicit relocation/execution changes; all learning parameters come from the checkpoint."""
    if track not in ('hu', '3max') or workers not in (1, 2):
        raise ValueError('VPS track must be hu/3max with one or two workers')
    if not checkpoint.is_file():
        raise FileNotFoundError(f'Explicit source checkpoint required: {checkpoint}')
    torch.set_num_threads(1)
    with measure("load_checkpoint") if measure else nullcontext():
        runner = TrainingRunner.load_checkpoint(checkpoint)
    if set(runner.solvers) != {track}:
        raise ValueError(f'Expected dedicated {track} checkpoint: {checkpoint}')
    before = {k: runner.config[k] for k in ('workers', 'trainer_threads', 'output_dir')}
    runner.config = {**runner.config, 'workers': workers, 'trainer_threads': 1, 'output_dir': str(output_dir.resolve())}
    runner.metadata.setdefault('vps_execution_changes', []).append({
        'iteration': runner.iteration, 'source': str(checkpoint.resolve()), 'before': before,
        'after': {k: runner.config[k] for k in before}, 'storage': asdict(storage)})
    with measure("bound_and_archive_diagnostics") if measure else nullcontext():
        runner.enable_bounded_storage(storage)
    return runner


def export_live(runner: TrainingRunner, *, keep: int = 2) -> Path:
    """Export the in-memory average models, validate ONNX, then atomically select the bundle."""
    if type(keep) is not int or keep < 1 or runner.iteration < 1:
        raise ValueError('Live export requires a trained iteration and positive retention')
    from training.workflow import seal_bundle, validate_bundle
    directory = runner.output_dir / 'exports'
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f'iteration_{runner.iteration:06d}'
    metadata = {'kind': 'live_training_export', 'iteration': runner.iteration, 'tracks': sorted(runner.solvers)}
    if destination.exists():
        manifest = validate_bundle(destination)
        if any(manifest.get(key) != value for key, value in metadata.items()):
            raise ValueError(f'Existing export conflicts: {destination}')
    else:
        temporary = Path(tempfile.mkdtemp(dir=directory, prefix='.export-'))
        try:
            for track, solver in runner.solvers.items():
                average = temporary / f'average_{track}.pt'
                solver.export_average(average)
                export_average_policy(average, temporary / f'average_{track}.onnx',
                                      temporary / f'average_{track}.json', observe(runner.probes[track][0][1]))
            seal_bundle(temporary, metadata)
            temporary.rename(destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    atomic_bytes(directory / 'current.json', (json.dumps({'directory': destination.name, **metadata})+'\n').encode())
    candidates = sorted(p for p in directory.glob('iteration_*') if p.is_dir() and p.name.removeprefix('iteration_').isdigit())
    for old in candidates[:-keep]:
        if old != destination:
            shutil.rmtree(old)
    return destination


def run_continuous(track: str, output_dir: Path, *, source_checkpoint: Path | None = None,
                   workers: int = 1, export_seconds: float = 3600,
                   total_seconds: float | None = None, storage: RuntimeStorage = RuntimeStorage()) -> None:
    """Run until interrupted; optional total_seconds bounds diagnostics. No network publication."""
    from training.workflow import exclusive_lock
    if not math.isfinite(export_seconds) or export_seconds <= 0:
        raise ValueError('export_seconds must be positive and finite')
    if total_seconds is not None and (not math.isfinite(total_seconds) or total_seconds <= 0):
        raise ValueError('total_seconds must be positive and finite')
    checkpoint = output_dir / 'checkpoint.pt'
    with exclusive_lock(output_dir / '.training.lock'):
        pointer = output_dir / 'resume.json'
        if pointer.exists():
            selected = json.loads(pointer.read_text())
            relative = Path(selected['path'])
            if (relative.is_absolute() or '..' in relative.parts
                    or not (relative == Path('checkpoint.pt') or
                            (relative.parent == Path('checkpoints') and relative.name.startswith('iteration_') and relative.suffix == '.pt'))):
                raise ValueError(f'Invalid resume pointer: {pointer}')
            source = output_dir / relative
        else:
            source = checkpoint if checkpoint.exists() else source_checkpoint
        if source is None:
            raise FileNotFoundError(f'No checkpoint at {checkpoint}; provide source_checkpoint for initial import')
        runner = prepare_runner(track, source, output_dir, workers=workers, storage=storage)
        if pointer.exists() and runner.iteration != selected['iteration']:
            raise ValueError(f'Resume pointer iteration does not match checkpoint: {pointer}')
        started = time.monotonic()
        while total_seconds is None or time.monotonic()-started < total_seconds:
            seconds = export_seconds if total_seconds is None else min(export_seconds, total_seconds-(time.monotonic()-started))
            if seconds <= 0:
                break
            result = runner.run_for(seconds, checkpoint)
            bundle = export_live(runner)
            print(f'[{track}] validated live export: {bundle}', flush=True)
            if result['stop_reason'] == 'interrupted':
                break


def prepare_vps_checkpoint(track: str, source: Path, output_dir: Path) -> Path:
    """One-time local import for a small VPS; keep source weights/replay immutable."""
    if output_dir.exists():
        raise FileExistsError(f'Choose an unused VPS import directory: {output_dir}')
    runner = prepare_runner(track, source, output_dir, workers=1)
    destination = output_dir / 'checkpoint.pt'
    runner.save_checkpoint(destination)
    return destination
