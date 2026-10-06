"""Bounded CPU cost profiling; no pilot training and no production optimizations."""
from __future__ import annotations
import copy
import cProfile
from dataclasses import replace
import io
import json
from pathlib import Path
import pickle
import pstats
import random
import statistics
import time
from typing import Callable

import torch
from actions import ACTION_IDS
from features.neural import neural_observation
from infoset import observe, NUMERIC_NAMES
from ml.deep_cfr import ModelSnapshot, TraversalTask, generate_samples
from ml.memory import ReservoirMemory, sample_bytes
from ml.model import AdvantageNetwork, AveragePolicyNetwork, encode_batch
from ml.train import fit
from scripts.parallel_cfr import collect_samples, _snapshot_bytes
from training.checkpoint import atomic_bytes, CONTRACT, pack_memory, pack_tournament, write_checkpoint
from training.config import load_config
from training.evaluation import evaluate_solver, model_policy
from training.root_sampler import RootSampler
from training.runner import TrainingRunner, source_metadata

OUTPUT = Path('profiling/deep_cfr_costs')


def measured(operation: Callable, repeats: int = 1) -> tuple[dict, object]:
    durations = []
    value = None
    for _ in range(repeats):
        start = time.perf_counter()
        value = operation()
        durations.append(time.perf_counter() - start)
    return {'seconds': durations, 'median_seconds': statistics.median(durations)}, value


def profiled(name: str, operation: Callable, report: dict) -> object:
    print(f'Profiling {name}', flush=True)
    profiler = cProfile.Profile()
    start = time.perf_counter()
    result = profiler.runcall(operation)
    elapsed = time.perf_counter() - start
    profiler.dump_stats(str(OUTPUT / f'{name}.prof'))
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats('cumulative')
    stats.print_stats(60)
    stats.sort_stats('tottime').print_stats(40)
    (OUTPUT / f'{name}.txt').write_text(stream.getvalue())
    rows = [{'file': file, 'line': line, 'function': function, 'primitive_calls': values[0],
             'calls': values[1], 'self_seconds': values[2], 'cumulative_seconds': values[3]}
            for (file, line, function), values in stats.stats.items()]
    report['profiles'][name] = {'instrumented_wall_seconds': elapsed,
                              'functions': sorted(rows, key=lambda row: row['cumulative_seconds'], reverse=True)}
    return result


def tasks_for(count: int, amount: int, config: dict, snapshot_models: dict) -> tuple[list[TraversalTask], dict]:
    name = '3max' if count == 3 else 'hu'
    sampler = RootSampler(count, 10000 + count, config[name]['root_sampling'])
    def policy(obs):
        players = round(obs.numeric[NUMERIC_NAMES.index('player_count')] * 3)
        return model_policy(snapshot_models[players], obs)
    sampler.policy = policy
    rng = random.Random(20000 + count)
    tasks = [TraversalTask(index, index % count, sampler.sample(), rng.randrange(2**31),
                           config['max_nodes'], config['max_depth']) for index in range(amount)]
    return tasks, sampler.coverage.as_dict()


def run_profile() -> dict:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    config = load_config('configs/deep_cfr_pilot.json')
    report = {'version': 1, **source_metadata(), 'config': config,
              'scope': 'bounded diagnostic workloads; no pilot; duplicate replay benchmark data never exported as a policy',
              'profiles': {}, 'stages': {}, 'scaling': [], 'microbenchmarks': {}}
    bootstrap = copy.deepcopy(config)
    bootstrap.update(output_dir=str(OUTPUT / 'bootstrap'), workers=1, outer_iterations=1,
                     checkpoint_every=100, evaluation_every=100)
    for name in ('3max', 'hu'):
        bootstrap[name]['traversals_per_player'] = 4
    runner = TrainingRunner(bootstrap)
    profiled('bootstrap_iteration', runner.run_iteration, report)
    models = {len(solver.players): solver.average_model for solver in runner.solvers.values()}
    snapshots = {name: solver.snapshot() for name, solver in runner.solvers.items()}
    task_sets = {}
    collected = {}
    for name, count in (('3max', 3), ('hu', 2)):
        timing, generated = measured(lambda: tasks_for(count, 128, config, models))
        tasks, coverage = generated
        task_sets[name] = tasks
        report['stages'][f'{name}_128_roots'] = dict(**timing, coverage=coverage)
        # Same root/deal/task seeds under uniform and learned snapshots.
        for phase, snapshot in (('uniform', ModelSnapshot(0, 'hand_chip_delta', {}, True)), ('learned', snapshots[name])):
            results = profiled(f'{name}_{phase}_traversal', lambda: collect_samples(snapshot, tasks[:16], 1), report)
            if phase == 'learned':
                collected[name] = results
        for amount in (16, 128):
            for workers in (1, 2, 4, 8):
                print(f'Benchmark {name}: {amount} tasks / {workers} workers', flush=True)
                timing, results = measured(lambda: collect_samples(snapshots[name], tasks[:amount], workers))
                report['scaling'].append({'track': name, 'tasks': amount, 'workers': workers, **timing,
                    'nodes': sum(r.nodes for r in results), 'samples': sum(len(r.advantages) + len(r.strategies) for r in results),
                    'worker_seconds': sum(r.worker_seconds for r in results),
                    'worker_cpu_seconds': sum(r.worker_cpu_seconds for r in results),
                    'traversals_per_second': amount / timing['median_seconds'],
                    'worker_peak_rss_bytes': max(r.worker_peak_rss_bytes for r in results)})
                if amount == 128 and workers == 1:
                    collected[name] = results
        atomic_bytes(OUTPUT / 'report.json', (json.dumps(report, indent=2) + '\n').encode())
    # Instrument real central fit/evaluation/replay on generated state shapes.
    pooled = [s for rows in collected.values() for row in rows for s in (*row.advantages, *row.strategies)]
    report['generated_samples'] = len(pooled)
    report['microbenchmarks']['runner_deepcopy'], _ = measured(lambda: (copy.deepcopy(runner.solvers), copy.deepcopy(runner.samplers)), 3)
    for name, solver in runner.solvers.items():
        report['stages'][f'{name}_evaluation'], _ = measured(lambda: evaluate_solver(solver, runner.probes[name],
            config['evaluation_max_nodes'], config['batch_size'], solver.snapshot()))
        profiled(f'{name}_evaluation', lambda: evaluate_solver(solver, runner.probes[name],
            config['evaluation_max_nodes'], config['batch_size'], solver.snapshot()), report)
    obs = observe(task_sets['3max'][0].root)
    compact = neural_observation(obs)
    model = runner.solvers['3max'].advantage_models[obs.hero]
    batch = encode_batch([compact])
    operations = {
        'hand_clone_1000': lambda: [task_sets['3max'][0].root.clone() for _ in range(1000)],
        'observe_1000': lambda: [observe(task_sets['3max'][0].root) for _ in range(1000)],
        'features_pack_1000': lambda: [neural_observation(obs) for _ in range(1000)],
        'encode_singleton_1000': lambda: [encode_batch([compact]) for _ in range(1000)],
        'sample_bytes_1000': lambda: [sample_bytes(pooled[0]) for _ in range(1000)],
        'snapshot_serialize_1000': lambda: [_snapshot_bytes(snapshots['3max']) for _ in range(1000)],
        'model_construct_load_100': lambda: [load_model(snapshots['3max'].advantage_weights[obs.hero]) for _ in range(100)],
        'worker_payload_pickle': lambda: [pickle.dumps(task, protocol=pickle.HIGHEST_PROTOCOL) for task in task_sets['3max']],
        'result_payload_pickle': lambda: [pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL) for result in collected['3max']],
    }
    for name, operation in operations.items():
        report['microbenchmarks'][name], _ = measured(operation, 3)
    with torch.no_grad():
        report['microbenchmarks']['inference_preencoded_1000'], _ = measured(lambda: [model(batch) for _ in range(1000)], 3)
        report['microbenchmarks']['inference_end_to_end_1000'], _ = measured(lambda: [model(encode_batch([obs])) for _ in range(1000)], 3)
    for kind, network in (('advantage', AdvantageNetwork), ('strategy', AveragePolicyNetwork)):
        source = [sample for sample in pooled if sample.kind == kind]
        data = [source[index % len(source)] for index in range(10000)]
        report['stages'][f'{kind}_data'] = {'distinct_samples': len(source), 'benchmark_samples': len(data),
            'history_mean': statistics.mean(len(s.state.history) for s in data),
            'history_max': max(len(s.state.history) for s in data), 'accounted_bytes': sum(sample_bytes(s) for s in data)}
        print(f'Benchmark full-capacity {kind} fit', flush=True)
        torch.manual_seed(909)
        report['stages'][f'{kind}_fit_10000'], _ = measured(lambda: fit(network(), data, 2, 64, 42), 2)
        profiled(f'{kind}_fit_10000', lambda: fit(network(), data, 2, 64, 42), report)
        memory = ReservoirMemory(10000, 42, kind, 'hand_chip_delta', byte_budget=config['memory_byte_budget'])
        report['stages'][f'{kind}_replay_add_10000'], _ = measured(lambda: add_all(memory, data))
        report['stages'][f'{kind}_replay_deepcopy_10000'], _ = measured(lambda: copy.deepcopy(memory), 3)
        profiled(f'{kind}_replay_add', lambda: add_all(ReservoirMemory(10000, 42, kind, 'hand_chip_delta', byte_budget=config['memory_byte_budget']), data), report)
        # Fill actual seven memory slots for a capacity-scale checkpoint benchmark.
        for solver in runner.solvers.values():
            destinations = solver.advantage_memory.values() if kind == 'advantage' else (solver.strategy_memory,)
            for destination in destinations:
                candidates = [s for s in source if s.player in solver.players]
                destination.samples = [candidates[index % len(candidates)] for index in range(10000)]
                destination.seen = len(destination.samples)
                destination._sample_sizes = [sample_bytes(s) for s in destination.samples]
                destination.used_bytes = sum(destination._sample_sizes)
    print('Benchmark full-capacity checkpoint', flush=True)
    checkpoint = OUTPUT / 'capacity_checkpoint.pt'
    report['stages']['checkpoint_full_capacity'], _ = measured(lambda: save_benchmark_checkpoint(runner, checkpoint), 2)
    profiled('checkpoint_full_capacity', lambda: save_benchmark_checkpoint(runner, checkpoint), report)
    report['stages']['checkpoint_full_capacity']['bytes'] = checkpoint.stat().st_size
    report['stages']['checkpoint_full_capacity']['scope'] = 'capacity/shape benchmark; repeated samples, not a usable training checkpoint'
    atomic_bytes(OUTPUT / 'report.json', (json.dumps(report, indent=2) + '\n').encode())
    print(f'Complete: {OUTPUT}/report.json', flush=True)
    return report


def save_benchmark_checkpoint(runner: TrainingRunner, path: Path) -> None:
    """Match serialization work while rejecting the synthetic replay as resumable."""
    raw = {'contract': {**CONTRACT, 'profiling_only': True}, 'config': runner.config,
           'iteration': runner.iteration, 'metadata': runner.metadata, 'metrics': runner.metrics,
           'torch_rng': torch.get_rng_state(), 'python_rng': random.getstate(), 'tracks': {}}
    for name, solver in runner.solvers.items():
        sampler = runner.samplers[name]
        raw['tracks'][name] = {'version': solver.version, 'seed': solver.seed, 'rng': solver.rng.getstate(),
            'advantage_weights': solver.snapshot().advantage_weights,
            'average_weights': solver.average_model.state_dict(),
            'advantage_memory': {player: pack_memory(memory) for player, memory in solver.advantage_memory.items()},
            'strategy_memory': pack_memory(solver.strategy_memory), 'metrics': solver.metrics,
            'traversal_mode': solver.traversal_mode, 'sampler_rng': sampler.rng.getstate(),
            'stratum_index': sampler.stratum_index, 'root_coverage': sampler.coverage.as_dict(),
            'tournament': pack_tournament(sampler.tournament)}
    write_checkpoint(path, raw)


def load_model(weights: dict) -> AdvantageNetwork:
    model = AdvantageNetwork()
    model.load_state_dict(weights, strict=True)
    return model.eval()


def add_all(memory: ReservoirMemory, samples: list) -> None:
    for sample in samples:
        memory.add(sample)


if __name__ == '__main__':
    run_profile()
