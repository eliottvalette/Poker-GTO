"""Bounded real-checkpoint performance measurements; no policy publication."""
from __future__ import annotations
import copy
import cProfile
from dataclasses import asdict
import io
import json
from pathlib import Path
import random
import statistics
import time
from typing import Callable, TypeVar
import torch
from infoset import NUMERIC_NAMES, Observation
from ml.deep_cfr import generate_samples
from ml.model import AdvantageNetwork
from ml.train import fit
from scripts.parallel_cfr import collect_samples, traversal_workers
from training.checkpoint import atomic_bytes
from training.evaluation import model_policy
from training.runner import TrainingRunner

_Result = TypeVar('_Result')


def _timed(operation: Callable[[], _Result]) -> tuple[float, _Result]:
    start = time.perf_counter()
    result = operation()
    return time.perf_counter() - start, result


def benchmark_checkpoint(checkpoint: Path, track: str, task_ids: tuple[int, ...],
                         max_nodes: int, workers: int, repeats: int, output: Path) -> dict:
    """Replay selected actual tasks and fit actual replay, without advancing training."""
    if track not in ('3max', 'hu') or not task_ids or repeats < 1 or workers < 1 or max_nodes < 1:
        raise ValueError('Explicit valid checkpoint benchmark configuration required')
    print(f'Benchmark checkpoint={checkpoint}; track={track}; tasks={task_ids}; workers={workers}', flush=True)
    old_threads, python_rng = torch.get_num_threads(), random.getstate()
    try:
        with torch.random.fork_rng(devices=[]):
            torch.set_num_threads(1)
            runner = TrainingRunner.load_checkpoint(checkpoint)
            if track not in runner.solvers:
                raise ValueError(f'Track {track} absent from checkpoint {checkpoint}')
            solver = runner.solvers[track]
            sampler = copy.deepcopy(runner.samplers[track])
            def rollout(obs: Observation) -> tuple[float, ...]:
                count = round(obs.numeric[NUMERIC_NAMES.index('player_count')] * 3)
                name = '3max' if count == 3 else 'hu'
                if name not in runner.solvers or runner.solvers[name].average_model is None:
                    raise ValueError(f'Missing {count}-player rollout model in {checkpoint}')
                return model_policy(runner.solvers[name].average_model, obs)
            sampler.policy = None if runner.iteration == 0 else rollout
            config = runner.config
            tasks, _ = solver.traversal_tasks(sampler.sample, config[track]['traversals_per_player'],
                max_nodes, config['max_depth'], config['sample_byte_budget'], config['traversal_mode'],
                config.get('outcome_epsilon', 0.6))
            if any(type(index) is not int or index < 0 or index >= len(tasks) for index in task_ids):
                raise ValueError(f'Task IDs must be within 0..{len(tasks)-1}: {task_ids}')
            selected = [tasks[index] for index in task_ids]
            snapshot = solver.snapshot()
            generation = []
            results = None
            for trial in range(repeats):
                print(f'Serial generation {trial+1}/{repeats}', flush=True)
                elapsed, current = _timed(lambda: [generate_samples(snapshot, task) for task in selected])
                generation.append(elapsed)
                if results is not None and [(r.nodes, r.value, r.advantages, r.strategies) for r in results] != [(r.nodes, r.value, r.advantages, r.strategies) for r in current]:
                    raise ValueError('Repeated generation produced different samples')
                results = current
            profiler = cProfile.Profile()
            profiler.runcall(generate_samples, snapshot, selected[max(range(len(results)), key=lambda i: results[i].nodes)])
            stats = io.BytesIO()
            # pstats files are developer diagnostics, never model artifacts.
            output.mkdir(parents=True, exist_ok=True)
            profiler.dump_stats(str(output / 'generation.prof'))
            parallel = []
            with traversal_workers(workers) as executor:
                for trial in range(repeats):
                    print(f'Persistent worker batch {trial+1}/{repeats}', flush=True)
                    elapsed, current = _timed(lambda: collect_samples(snapshot, selected, workers, executor=executor))
                    parallel.append(elapsed)
                    if [(r.nodes, r.value, r.advantages, r.strategies) for r in results] != [(r.nodes, r.value, r.advantages, r.strategies) for r in current]:
                        raise ValueError('Parallel generation differs from serial samples')
            old_copy_seconds, _ = _timed(lambda: copy.deepcopy(solver))
            fork_seconds, _ = _timed(solver.fork)
            print('Fit actual retained advantage replay', flush=True)
            torch.manual_seed(123)
            model = AdvantageNetwork()
            fit_seconds, fit_metrics = _timed(lambda: fit(model, solver.advantage_memory.samples, 2, 64, 123))
            report = {'checkpoint':str(checkpoint), 'iteration':runner.iteration, 'track':track,
                'task_ids':list(task_ids), 'workers':workers, 'traversal_mode':config['traversal_mode'],
                'generation_seconds':generation, 'generation_median':statistics.median(generation),
                'persistent_worker_seconds':parallel, 'deepcopy_solver_seconds':old_copy_seconds,
                'fork_solver_seconds':fork_seconds, 'fit_seconds':fit_seconds, 'fit_metrics':fit_metrics,
                'total_nodes':sum(r.nodes for r in results), 'cpu_threads':1,
                'scope':'diagnostic generation and fit only; no training checkpoint or policy modified'}
            torch.save({'samples':[{'task':r.task_id, 'nodes':r.nodes, 'value':r.value,
                'advantages':[asdict(s) for s in r.advantages], 'strategies':[asdict(s) for s in r.strategies]} for r in results],
                'trained_weights':model.state_dict()}, stats)
            atomic_bytes(output / 'results.pt', stats.getvalue())
            atomic_bytes(output / 'report.json', (json.dumps(report,indent=2)+'\n').encode())
            return report
    finally:
        random.setstate(python_rng)
        torch.set_num_threads(old_threads)
