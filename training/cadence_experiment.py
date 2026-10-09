"""Bounded generation and frozen-replay comparisons; never mutates a checkpoint."""
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import time

import torch

from ml.model import AdvantageNetwork, AveragePolicyNetwork
from ml.train import fit
from scripts.parallel_cfr import collect_samples
from training.metrics import opening_hand_class
from training.runner import TrainingRunner


def compare(checkpoint: Path, output: Path, *, track: str = 'hu') -> dict:
    """Compare nested frozen-strategy batches and independent bounded fits on CPU."""
    torch.set_num_threads(1)
    before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    runner = TrainingRunner.load_checkpoint(checkpoint)
    solver = runner.solvers[track]
    report = {'checkpoint_sha256': before, 'track': track, 'iteration': runner.iteration,
              'generation': [], 'fits': [], 'limitations': [
                  'Frozen-model samples are not additional strategy updates.',
                  'Record-level held-out loss is not independent-root strategic quality.',
                  'Serial CPU timings include feature construction and validation.']}
    snapshot = solver.snapshot()
    sampler = copy.deepcopy(runner.samplers[track])
    sampler.policy = solver.average_model.probabilities
    per_player = 512 if track == 'hu' else 128
    started = time.perf_counter()
    tasks, _ = solver.traversal_tasks(sampler.task_factory(solver.players, per_player), per_player,
                                    runner.config['max_nodes'], runner.config['max_depth'],
                                    runner.config['sample_byte_budget'], solver.traversal_mode, .6)
    root_seconds = time.perf_counter() - started
    tasks = [tasks[p * per_player + i] for i in range(per_player) for p in range(len(solver.players))]
    counters = {'advantage': Counter(), 'strategy': Counter()}
    totals = Counter()
    nodes = terminal = 0
    started, cpu_started = time.perf_counter(), time.process_time()
    boundaries = (256,512,1024) if track == 'hu' else (96,192,384)
    chunk = 32 if track == 'hu' else 24
    for offset in range(0,len(tasks),chunk):
        selected = tasks[offset:offset+chunk]
        results = collect_samples(snapshot, selected, 1, runner.config['generation_byte_budget'])
        terminal += sum(t.root.terminal for t in selected)
        for result in results:
            nodes += result.nodes
            for kind, rows in (('advantage',result.advantages),('strategy',result.strategies)):
                totals[kind] += len(rows)
                counters[kind].update(h for s in rows if (h := opening_hand_class(s.state)) is not None)
        count = offset + len(selected)
        if count in boundaries:
            report['generation'].append({'traversals':count,'nodes':nodes,'terminal_roots':terminal,
                'seconds':time.perf_counter()-started,'cpu_seconds':time.process_time()-cpu_started,
                'root_preparation_seconds_full_batch':root_seconds,
                'samples':dict(totals),'opening_classes':{k:dict(v) for k,v in counters.items()}})
            print(track, 'generation', count, report['generation'][-1]['seconds'], flush=True)
    # No newly generated sample is inserted into the source replay.
    torch.optim.Adam(solver.advantage_model.parameters())
    for kind, memory, previous, cls, seed in (
        ('advantage',solver.advantage_memory,solver.advantage_model,AdvantageNetwork,solver.seed),
        ('strategy',solver.strategy_memory,solver.average_model,AveragePolicyNetwork,solver.seed+10)):
        samples = memory.samples
        for initialization, updates, cache in (('fresh',200,False),('fresh',200,True),
                                                ('warm',200,True),('fresh',800,True),('warm',800,True)):
            torch.manual_seed(7401)
            model = cls() if initialization == 'fresh' else copy.deepcopy(previous)
            start, cpu_start = time.perf_counter(), time.process_time()
            result = fit(model,samples,50,64,seed,max_updates=updates,cache_encoding=cache)
            report['fits'].append({'kind':kind,'initialization':initialization,'update_cap':updates,
                'cache':cache,'seconds':time.perf_counter()-start,'cpu_seconds':time.process_time()-cpu_start,
                'metrics':result})
            print(track, kind, initialization, updates, cache, result['heldout_loss'], flush=True)
    report['checkpoint_unchanged'] = hashlib.sha256(checkpoint.read_bytes()).hexdigest() == before
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+'\n')
    return report


def compare_reference(checkpoint: Path, output: Path, *, track: str = 'hu') -> dict:
    """Compare neural inference and exact local refinement on a tractable range game."""
    from actions import ACTION_IDS, legal_actions
    from evaluation import subgame_best_response
    from hybrid.policy_source import NeuralAveragePolicy, TabularPolicy
    from hybrid.ranges import HandRange, JointRanges
    from hybrid.river_solver import RiverSolver
    from hybrid.state import ComputeBudget, instantiate
    from infoset import observe
    from ml.deep_cfr import FrozenStrategy
    from ml.model import encode_batch
    from scripts.strategy_collector_audit import small_root

    torch.set_num_threads(1)
    runner = TrainingRunner.load_checkpoint(checkpoint)
    solver = runner.solvers[track]
    root = small_root(len(solver.players))
    available = [c for c in range(52) if c not in root.board]
    factors = {p: HandRange({tuple(available[4*i:4*i+2]): 1.,
                            tuple(available[4*i+2:4*i+4]): 2.}) for i,p in enumerate(root.players)}
    ranges = JointRanges(factors,tuple(root.board))
    worlds = [(d.probability,instantiate(root,d.hands)) for d in ranges.enumerate()]
    prior = NeuralAveragePolicy(solver.average_model.probabilities, f'{track}-iteration-{runner.iteration}')
    rows = []
    for mode in ('neural_only','local_without_prior','local_neural_prior'):
        for iterations in ((0,) if mode == 'neural_only' else (10,200)):
            start = time.perf_counter()
            solution = None if not iterations else RiverSolver(prior if mode == 'local_neural_prior' else None).solve(
                root,ranges,ComputeBudget(iterations=iterations,max_nodes=1000000))
            policy = prior if solution is None else solution.policy
            if solution is None:
                table = {}
                def materialize(state):
                    if state.terminal:
                        return
                    obs = observe(state)
                    if obs.key() not in table:
                        table[obs.key()] = prior.probabilities(obs)
                    for action in legal_actions(state):
                        child = state.clone(); action.apply(child); materialize(child)
                for _,state in worlds:
                    materialize(state)
                policy = TabularPolicy(table)
            seconds = time.perf_counter()-start
            gains = {str(p):subgame_best_response(worlds,policy.probabilities,p)['best_response_gain_bb'] for p in root.players}
            rows.append({'mode':mode,'iterations':iterations,'solve_seconds':seconds,
                         'nodes':0 if solution is None else solution.work.nodes,'bounded_response_gain_by_player':gains})
    frozen = FrozenStrategy(solver.snapshot())
    nodes = 0
    def value(state, player):
        nonlocal nodes
        nodes += 1
        if nodes > 100000:
            raise RuntimeError('Independent regret reference node budget exceeded')
        if state.terminal:
            return state.utility(player)
        probabilities = frozen(observe(state))
        total = 0.
        for action in legal_actions(state):
            child = state.clone(); action.apply(child)
            total += probabilities[ACTION_IDS.index(action.action_id)] * value(child,player)
        return total
    by_key = {}
    for weight,state in worlds:
        obs = observe(state)
        row = by_key.setdefault(obs.key(),{'weight':0.,'ev':[0.]*len(ACTION_IDS),'observation':obs})
        row['weight'] += weight
        for action in legal_actions(state):
            child=state.clone();action.apply(child)
            row['ev'][ACTION_IDS.index(action.action_id)] += weight*value(child,obs.hero)
    regrets=[]
    for row in by_key.values():
        obs=row['observation']; probabilities=frozen(obs)
        ev=[v/row['weight'] for v in row['ev']]
        baseline=sum(p*v for p,v in zip(probabilities,ev))
        expected=[v-baseline if legal else 0. for v,legal in zip(ev,obs.legal_mask)]
        with torch.inference_mode():
            predicted=solver.advantage_model(encode_batch([obs]))[0].tolist()
        legal=[i for i,m in enumerate(obs.legal_mask) if m]
        regrets.append({'cards':obs.cards[:2], 'reference_action_ev':ev,'reference_instantaneous_regret':expected,
            'predicted_historical_regret':predicted,
            'legal_mse':sum((expected[i]-predicted[i])**2 for i in legal)/len(legal),
            'zero_baseline_mse':sum(expected[i]**2 for i in legal)/len(legal)})
    result={'track':track,'iteration':runner.iteration,'solver_comparison':rows,'regret_diagnostics':regrets,
            'reference_nodes':nodes,'limitations':[
                'Deliberately tiny river game; not a general poker strength comparison.',
                'Compute cost is reported; iteration budgets are not matched CPU budgets.',
                'A predicts historical sampled regrets, not exactly current-profile instantaneous regrets. Disagreement is a diagnostic, not an unbiased generalization error.']}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2)+'\n')
    return result


def preflight(track: str, output: Path, *, iterations: int = 2) -> dict:
    """Run actual configured batches in a new diagnostic directory, then verify reload."""
    from training.config import load_config
    if track not in ('hu','3max') or type(iterations) is not int or iterations < 1:
        raise ValueError('Explicit supported track and positive iteration count required')
    if output.exists():
        raise FileExistsError(f'Diagnostic output already exists: {output}')
    config = load_config(f'configs/train_{track}.json')
    config.update(output_dir=str(output), checkpoint_every=iterations, evaluation_every=iterations)
    runner = TrainingRunner(config)
    runner.run(iterations)
    path = output/'checkpoint.pt'
    runner.save_checkpoint(path)
    loaded = TrainingRunner.load_checkpoint(path)
    if loaded.iteration != runner.iteration:
        raise RuntimeError('Preflight checkpoint iteration mismatch')
    for name,solver in runner.solvers.items():
        other = loaded.solvers[name]
        for kind in ('advantage_memory','strategy_memory'):
            if getattr(solver,kind).samples != getattr(other,kind).samples:
                raise RuntimeError(f'Preflight replay mismatch: {name}/{kind}')
        for kind in ('advantage_model','average_model'):
            if any(not torch.equal(v,getattr(other,kind).state_dict()[k])
                   for k,v in getattr(solver,kind).state_dict().items()):
                raise RuntimeError(f'Preflight model mismatch: {name}/{kind}')
    return {'checkpoint':str(path),'iterations':runner.iteration,'reload_verified':True}
