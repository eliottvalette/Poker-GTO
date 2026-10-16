"""Persistent solver-owned knowledge generation and bounded production telemetry."""
from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import random
import signal
import time
from typing import Callable

from hybrid.adaptive_labels import LabelBudget, generate
from hybrid.knowledge import calculate
from hybrid.knowledge_curriculum import FAMILIES, scenario
from hybrid.state import BudgetExceeded, ComputeBudget
from training.checkpoint import atomic_bytes
from training.publication import TRACKS, encoded

ENGINE = 'solver_knowledge_v1'
STREETS = ('PREFLOP', 'FLOP', 'TURN', 'RIVER')
DEFAULTS = {'roots_per_family_street': 100, 'iterations_per_visit': 16,
            'max_iterations': 256, 'samples': 16, 'max_nodes': 300000,
            'max_storage_bytes': 1024 * 1024 * 1024, 'fit_every': 32, 'fit_updates': 16}


def write(path: Path, value: dict) -> None:
    atomic_bytes(path, encoded(value))


def initialize(directory: Path, track: str, config: dict | None = None) -> dict:
    if track not in TRACKS:
        raise ValueError(f'Unknown solver track: {track}')
    config = dict(DEFAULTS if config is None else config)
    if set(config) != set(DEFAULTS) or any(type(v) is not int or v < 1 for v in config.values()):
        raise ValueError('Invalid solver production configuration')
    directory.mkdir(parents=True, exist_ok=True)
    contract = {'engine': ENGINE, 'track': track, 'config': config, 'curriculum_version': 2}
    path = directory / 'contract.json'
    if path.exists():
        if json.loads(path.read_text()) != contract:
            raise ValueError('Solver service contract changed; explicit run migration required')
    else:
        if any(directory.iterdir()):
            raise ValueError('Nonempty solver directory without contract')
        write(path, contract)
    for name in ('tables', 'labels', 'references'):
        (directory / name).mkdir(exist_ok=True)
    return contract


def _state(directory: Path) -> dict:
    path = directory / 'state.json'
    if path.exists():
        return json.loads(path.read_text())
    return {'cursor': 0, 'visits': 0, 'computed': 0, 'failed': {}, 'nodes': 0,
            'cpu_seconds': 0., 'streets': {}, 'student': {'accepted': False, 'status': 'not_fitted'}}


def step(directory: Path, track: str, config: dict | None = None) -> dict:
    """One bounded, atomically recorded game update; safe to repeat after a crash."""
    contract = initialize(directory, track, config)
    cfg = contract['config']
    state = _state(directory)
    count = 20 * cfg['roots_per_family_street']
    cursor = state['cursor']
    if cursor >= count * ((cfg['max_iterations'] + cfg['iterations_per_visit'] - 1) // cfg['iterations_per_visit']):
        return {**state, 'status': 'allocation_complete'}
    round_index, coordinate = divmod(cursor, count)
    index, position = divmod(coordinate, 20)
    family = FAMILIES[position // 4]
    street = STREETS[position % 4]
    item = scenario(TRACKS[track], street, family, index)
    name = hashlib.sha256(item.identity.encode()).hexdigest()
    path = directory / 'tables' / f'{name}.json'
    target = min(cfg['max_iterations'], (round_index + 1) * cfg['iterations_per_visit'])
    completed = json.loads(path.read_text())['checkpoint']['iterations'] if path.exists() else 0
    begin = time.process_time()
    size = sum(p.stat().st_size for p in directory.rglob('*') if p.is_file())
    if size >= cfg['max_storage_bytes']:
        raise ValueError('Solver storage limit reached; increase explicitly or review the curriculum')
    if name not in state['failed'] and completed < target:
        budget = ComputeBudget(iterations=target-completed, samples=cfg['samples'], max_depth=1,
                               max_nodes=cfg['max_nodes'], seed=141, strict=False)
        try:
            result = calculate(path, item.root, item.ranges, budget, split=item.split,
                               continuation=item.policy, sample_private=True, depth_limited=True,
                               max_artifact_bytes=min(8 * 1024 * 1024, cfg['max_storage_bytes']-size))
            state['streets'][street] = {'iterations': result.checkpoint['iterations'], 'nodes': result.work.nodes,
                                       'samples': result.work.samples, 'table': path.name}
        except BudgetExceeded as error:
            state['failed'][name] = {'scenario': item.identity, 'error': str(error)}
            print(f'{track} {item.identity}: {error}', flush=True)
    label_path = directory / 'labels' / f'{name}.json'
    if name not in state['failed'] and not label_path.exists():
        holding = item.ranges.sample(random.Random(item.identity))[item.root.current_player]
        label = generate(item.root, item.ranges, holding, {p: item.policy for p in item.root.players},
                         LabelBudget(looks=(32, 128, 512), tolerance_bb=.1,
                                     max_nodes=cfg['max_nodes'], seed=141))
        payload = encoded(label)
        size = sum(p.stat().st_size for p in directory.rglob('*') if p.is_file())
        if size + len(payload) > cfg['max_storage_bytes']:
            raise ValueError('Response label exceeds solver storage limit')
        atomic_bytes(label_path, payload)
    # Reconcile from durable game artifacts after interruption between table and state writes.
    if path.exists():
        saved = json.loads(path.read_text())
        accounted = state.setdefault('accounted', {})
        old_nodes, old_batches = accounted.get(name, (0, 0))
        new_nodes = saved.get('cumulative_nodes', saved['work']['nodes'])
        new_batches = saved.get('calculation_batches', 1)
        state['nodes'] += new_nodes - old_nodes
        state['computed'] += new_batches - old_batches
        accounted[name] = (new_nodes, new_batches)
        state['streets'][street] = {'iterations': saved['checkpoint']['iterations'],
            'nodes': saved['work']['nodes'], 'samples': saved['work']['samples'], 'table': path.name}
    state['cpu_seconds'] += time.process_time() - begin
    state['cursor'] += 1
    state['visits'] += 1
    state['status'] = 'running'
    state['updated_at'] = datetime.now(timezone.utc).isoformat()
    write(directory / 'state.json', state)
    print(f'{track} solver {state["cursor"]}: {item.identity}, CFR target {target}, '
          f'{state["nodes"]} cumulative nodes', flush=True)
    return state


def fit_students(directory: Path, track: str) -> dict:
    """Optional bounded distillation; these weights never generate teacher targets."""
    from hybrid.distillation import SolverStudents
    cfg = json.loads((directory / 'contract.json').read_text())['config']
    paths = sorted((directory / 'tables').glob('*.json'))
    if not paths:
        return {'accepted': False, 'status': 'no_computed_tables'}
    # Deterministic bounded representation across the entire persisted population.
    population_count = len(paths)
    if len(paths) > 64:
        bins = {}
        for path in paths:
            contract = json.loads(path.read_text())['contract']
            bins.setdefault((contract['root']['street'], contract['split']), []).append(path)
        selected = []
        offset = 0
        target_count = min(64, sum(min(len(v), 6) for v in bins.values()))
        while len(selected) < target_count:
            for key in sorted(bins):
                if offset < min(len(bins[key]), 6) and len(selected) < target_count:
                    selected.append(bins[key][offset * (len(bins[key])-1) // max(1, min(len(bins[key]), 6)-1)])
            offset += 1
        paths = selected
    coverage = {(json.loads(p.read_text())['contract']['root']['street'],
                 json.loads(p.read_text())['contract']['split']) for p in paths}
    missing = sorted((street, split) for street in STREETS for split in ('train', 'validation', 'test')
                     if (street, split) not in coverage)
    if missing:
        return {'accepted': False, 'status': 'insufficient_split_coverage', 'missing': missing}
    checkpoint = directory / 'students.pt'
    learner = SolverStudents.load(checkpoint) if checkpoint.exists() else SolverStudents(
        TRACKS[track], seed=141, continuations=('uniform-legal-v1', 'passive-continuation-v1', 'aggressive-continuation-v1', 'terminal-river-v1'))
    metrics = learner.fit(paths, updates=cfg['fit_updates'], max_encoding_bytes=128 * 1024 * 1024, allow_dataset_update=True)
    learner.save(checkpoint)
    return {'accepted': False, 'status': 'experimental', 'metrics': metrics,
            'fit_tables': len(paths), 'population_tables': population_count}


def refine_references(directory: Path, track: str) -> None:
    """Worker-owned fixed sentinels; evaluation never trains its own candidate."""
    from hybrid.solver_experiments import bounded_roots
    for root, ranges in bounded_roots(TRACKS[track], 'test'):
        path = directory / 'references' / f'{root.street}.json'
        iterations = json.loads(path.read_text())['checkpoint']['iterations'] if path.exists() else 0
        if iterations < 256:
            calculate(path, root, ranges, ComputeBudget(iterations=min(32, 256-iterations), samples=32,
                      max_depth=1, max_nodes=1000000, seed=141), split='test')


def evaluate(directory: Path, track: str) -> dict:
    """Fixed, tractable sentinel games; not full-game win rate or generalization."""
    from hybrid.solver_experiments import bounded_roots
    from hybrid.river_solver import RiverSolver
    from evaluation import subgame_best_response
    from hybrid.knowledge import cached_policy, root_supervision
    from actions import ACTION_IDS
    from hybrid.state import instantiate
    state = _state(directory)
    rows = []
    for root, ranges in bounded_roots(TRACKS[track], 'test'):
        path = directory / 'references' / f'{root.street}.json'
        if not path.exists():
            raise ValueError(f'Worker has not produced reference table: {root.street}')
        candidate = cached_policy(path, root, ranges)
        _, records = root_supervision(path, 'strategy')
        strategies = {tuple(sorted(r.state.cards[:2])): dict(zip(ACTION_IDS, r.target)) for r in records}
        iterations = json.loads(path.read_text())['checkpoint']['iterations']
        high = ComputeBudget(iterations=256, samples=256, max_depth=1, max_nodes=1000000, seed=142)
        reference = (RiverSolver().solve(root, ranges, high) if root.street == 'RIVER'
                     else RiverSolver().solve_chance(root, ranges, high))
        losses = []
        for hand, values in reference.action_ev_by_hand.items():
            strategy = strategies[hand]
            losses.append((max(values.values()) - sum(strategy.get(a, 0) * v for a, v in values.items())) / root.blinds.big)
        gain = None
        if root.street == 'RIVER':
            worlds = [(d.probability, instantiate(root, d.hands)) for d in ranges.enumerate()]
            gain = sum(subgame_best_response(worlds, candidate.probabilities, p)['best_response_gain_bb'] for p in root.players)
        rows.append({'street': root.street, 'response_shortfall_bb': sum(losses)/len(losses),
                     'river_response_gain_bb': gain, 'iterations': iterations})
    label_count, resolved = 0, 0
    for path in (directory / 'labels').glob('*.json'):
        label = json.loads(path.read_text())
        label_count += 1
        resolved += bool(label['estimate']) and label['estimate']['status'] != 'indeterminate'
    now = datetime.now(timezone.utc).isoformat()
    return {'at': now, 'visits': state['visits'], 'computed_updates': state['computed'],
            'tables': len(list((directory / 'tables').glob('*.json'))), 'labels': label_count,
            'resolved_labels': resolved, 'failed_roots': len(state['failed']), 'nodes': state['nodes'],
            'cpu_seconds': state['cpu_seconds'], 'streets': rows, 'student_accepted': False}


def publish(storage, directory: Path, track: str) -> dict:
    from training.evaluation_history import read_json
    contract = json.loads((directory / 'contract.json').read_text())
    key = f'solver/{track}/history.json'
    history = read_json(storage, key) or {'engine': ENGINE, 'track': track, 'contract': contract, 'runs': []}
    if history['contract'] != contract:
        raise ValueError('Remote solver history belongs to a different configuration')
    now = datetime.now(timezone.utc).isoformat()
    if history['runs'] and history['runs'][-1]['at'][:13] == now[:13]:
        return history['runs'][-1]
    row = evaluate(directory, track)
    # One compact observation per UTC hour, bounded at 720 points.
    history['runs'] = [r for r in history['runs'] if r['at'][:13] != row['at'][:13]]
    history['runs'] = (history['runs'] + [row])[-720:]
    history['updated_at'] = row['at']
    storage.put(key, encoded(history), mutable=True)
    if read_json(storage, key) != history:
        raise ValueError('Solver history publication read-back mismatch')
    storage.put(f'solver/{track}/current.json', encoded({'engine': ENGINE, 'track': track,
                'updated_at': row['at'], 'history': key, 'student_accepted': False}), mutable=True)
    return row


def run(directory: Path, track: str, ready: Callable[[], None]) -> None:
    import torch
    torch.set_num_threads(1)
    initialize(directory, track)
    stop = False
    def request_stop(*_):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    with (directory / '.writer.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ready()
        while not stop:
            state = step(directory, track)
            if state['status'] == 'allocation_complete':
                print('Declared allocation complete; unresolved precision remains explicit; waiting for an extension', flush=True)
                time.sleep(30)
                continue
            if state['visits'] == 4 or state['visits'] % DEFAULTS['fit_every'] == 0:
                refine_references(directory, track)
            if state['computed'] and state['visits'] % DEFAULTS['fit_every'] == 0:
                begin = time.process_time()
                try:
                    state['student'] = fit_students(directory, track)
                except (ValueError, MemoryError) as error:
                    state['student'] = {'accepted': False, 'status': 'rejected', 'error': str(error)}
                    print(f'Optional student fitting rejected: {error}', flush=True)
                state['student']['cpu_seconds'] = time.process_time()-begin
                write(directory / 'state.json', state)


def publish_status(storage, directory: Path, track: str) -> None:
    """Publish compact progress and four scoped tables, never an ONNX substitute."""
    state = _state(directory)
    for street, row in state['streets'].items():
        payload = (directory / 'tables' / row['table']).read_bytes()
        storage.put(f'solver/{track}/tables/{street}.json', payload, mutable=True)
    summary = {**state, 'engine': ENGINE, 'track': track,
               'failed_roots': len(state['failed']), 'student_accepted': False}
    summary.pop('failed')
    summary.pop('accounted', None)
    summary['student'] = {k: v for k, v in state['student'].items() if k != 'metrics'}
    storage.put(f'solver/{track}/status.json', encoded(summary), mutable=True)
