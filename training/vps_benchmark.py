"""Bounded, isolated CPU trials with process-tree RSS sampling at each lifecycle phase."""
from __future__ import annotations
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import threading
import time
import torch
from training.runtime_storage import RuntimeStorage
from training.vps import prepare_runner, export_live


class PhaseMemory:
    def __init__(self) -> None:
        import psutil
        self.process = psutil.Process()
        self.rows: list[dict] = []

    @contextmanager
    def phase(self, name: str):
        import psutil
        stop = threading.Event()
        peak = {'parent_rss_bytes': 0, 'tree_rss_bytes': 0, 'processes': 0}
        def measure():
            try:
                parent = self.process.memory_info().rss
                children = self.process.children(recursive=True)
                total = parent
                for child in children:
                    try:
                        total += child.memory_info().rss
                    except psutil.NoSuchProcess:
                        pass
                peak['parent_rss_bytes'] = max(peak['parent_rss_bytes'], parent)
                peak['tree_rss_bytes'] = max(peak['tree_rss_bytes'], total)
                peak['processes'] = max(peak['processes'], 1+len(children))
            except psutil.NoSuchProcess:
                pass
        def watch():
            while not stop.wait(.02):
                measure()
        measure()
        thread = threading.Thread(target=watch, daemon=True)
        started = time.perf_counter()
        thread.start()
        try:
            yield
        finally:
            measure(); stop.set(); thread.join()
            row = {'phase': name, 'seconds': time.perf_counter()-started, **peak,
                   'process_lifetime_high_water_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if platform.system() == 'Darwin' else 1024)}
            self.rows.append(row)
            print(json.dumps(row), flush=True)


def model_digest(runner) -> str:
    result = hashlib.sha256()
    for solver in runner.solvers.values():
        for model in (solver.advantage_model, solver.average_model):
            for name, tensor in model.state_dict().items():
                result.update(name.encode()); result.update(tensor.cpu().numpy().tobytes())
    return result.hexdigest()


def trial(track: str, source: Path, output: Path, workers: int, *, iterations: int = 1) -> dict:
    if output.exists():
        raise FileExistsError(f'Diagnostic output must be new: {output}')
    output.mkdir(parents=True)
    torch.set_num_threads(1)
    probe = PhaseMemory()
    runner = prepare_runner(track, source, output, workers=workers, storage=RuntimeStorage(), measure=probe.phase)
    initial = runner.iteration
    from scripts.parallel_cfr import traversal_workers
    for i in range(iterations):
        with probe.phase(f'iteration_{i+1}_including_worker_start_stop'):
            with traversal_workers(workers) as executor:
                metric = runner.run_iteration(executor=executor)
    with probe.phase('save'):
        runner.save_checkpoint(output/'checkpoint.pt')
    with probe.phase('live_export'):
        export_live(runner)
    result = {'track': track, 'source': str(source.resolve()), 'workers': workers,
              'trainer_threads': 1, 'platform': platform.platform(), 'initial_iteration': initial,
              'final_iteration': runner.iteration, 'phases': probe.rows,
              'model_sha256': model_digest(runner), 'checkpoint_bytes': (output/'checkpoint.pt').stat().st_size,
              'nodes': metric['tracks'][track]['nodes'], 'traversals': metric['tracks'][track]['traversals'],
              'capacities': {'advantage': runner.config[track]['advantage_capacity'], 'strategy': runner.config['strategy_capacity']},
              'rss_note': '20ms sampled sum of process RSS; shared pages may be counted more than once. Lifetime high-water is parent-only.'}
    (output/'report.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def compare_workers(track: str, source: Path, output: Path, *, timeout_seconds: float = 300) -> list[dict]:
    """Fresh process per configuration; call from a guarded Python entrypoint."""
    import multiprocessing
    results = []
    for workers in (1, 2):
        directory = output / f'{track}-{workers}-workers'
        process = multiprocessing.get_context('spawn').Process(target=trial, args=(track, source, directory, workers))
        process.start()
        process.join(timeout_seconds)
        if process.is_alive():
            # Interrupt the worker so Python can unwind its executor and diagnostics.
            import signal
            os.kill(process.pid, signal.SIGINT)
            process.join(30)
            if process.is_alive():
                process.kill(); process.join()
            raise TimeoutError(f'{track}/{workers} exceeded {timeout_seconds}s; inspect {directory}')
        if process.exitcode:
            raise RuntimeError(f'{track}/{workers} failed with exit {process.exitcode}; inspect {directory}')
        results.append(json.loads((directory/'report.json').read_text()))
    if results[0]['model_sha256'] != results[1]['model_sha256']:
        raise AssertionError('Worker configuration changed trained model weights')
    (output/f'{track}-comparison.json').write_text(json.dumps(results, indent=2)+'\n')
    return results
