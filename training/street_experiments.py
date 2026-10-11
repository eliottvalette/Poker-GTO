"""Bounded, reproducible street-specialist learning and algorithmic diagnostics."""
from __future__ import annotations
import copy
from dataclasses import replace
import json
from pathlib import Path
import random
import resource
import sys
import time
import torch
from actions import ACTION_IDS
from cfr_solver import regret_matching, Traversal
from infoset import observe
from ml.model import STREETS, AdvantageNetwork, AveragePolicyNetwork, encode_batch
from ml.street_training import grouped_partition
from ml.train import evaluate_loss, weight_quality
from training.config import load_config
from training.runner import TrainingRunner


def diagnostic_config(output: Path, count: int, seed: int, *, updates: int = 32,
                      traversals: int = 16) -> dict:
    track = 'hu' if count == 2 else '3max'
    config = load_config(f'configs/train_{track}_streets.json')
    config.update(output_dir=str(output), seed=seed, workers=1, trainer_threads=1,
                  checkpoint_every=1000, evaluation_every=1000, generation_batch_size=16,
                  strategy_capacity=8000, memory_byte_budget=64*1024**2)
    for name in ('hu','3max'):
        config[name].update(advantage_capacity=8000, advantage_byte_budget=64*1024**2,
                            traversals_per_player=traversals)
    for kind in ('advantage','strategy'):
        for entry in config['street_networks'][kind].values():
            entry.update(capacity=2000, byte_budget=16*1024**2, every=1,
                         min_new_samples=8, max_updates=updates)
    return config


def dataset_metrics(model, memory, seed: int) -> dict:
    result = {}
    for index, name in enumerate(STREETS):
        rows = memory.memories[name].samples
        partitions = {split: [s for s in rows if grouped_partition(s,seed) == split]
                      for split in ('train','validation','test')}
        network = model.specialists[name] if hasattr(model,'specialists') else model
        metrics = {'retained': len(rows), 'groups': len({s.root_group for s in rows}),
                   **(weight_quality(rows) if rows else {})}
        for split, samples in partitions.items():
            metrics[split] = {'samples': len(samples),
                              'loss': evaluate_loss(network,samples,128) if samples else None}
        if memory.kind == 'advantage' and rows:
            with torch.no_grad():
                batch = encode_batch([s.state for s in rows])
                values = network(batch)
                metrics['all_nonpositive_fraction'] = float((values.masked_fill(~batch['mask'], -torch.inf).max(1).values <= 0).float().mean())
        result[name] = metrics
    return result


def river_reference(model, count: int, iterations: int = 200) -> dict:
    from scripts.strategy_collector_audit import small_root
    from hybrid.ranges import HandRange, JointRanges
    from hybrid.river_solver import RiverSolver
    from hybrid.state import ComputeBudget, instantiate
    from evaluation import subgame_best_response
    root = small_root(count)
    available = [c for c in range(52) if c not in root.board]
    ranges = JointRanges({p: HandRange({tuple(available[4*p:4*p+2]):1.,
                                       tuple(available[4*p+2:4*p+4]):2.}) for p in root.players}, tuple(root.board))
    solution = RiverSolver().solve(root, ranges, ComputeBudget(max_nodes=200000, iterations=iterations))
    deals = [(d.probability, instantiate(root,d.hands)) for d in ranges.enumerate(1000)]
    br = subgame_best_response(deals, model.probabilities, root.current_player, max_nodes=200000)
    rows = []
    for hand, action_evs in solution.action_ev_by_hand.items():
        world = root.clone(); world.actor.cards = hand
        probabilities = model.probabilities(observe(world))
        expected = solution.strategy_by_hand[hand]
        mixture = sum(probabilities[ACTION_IDS.index(action)] * value for action,value in action_evs.items())
        rows.append({'holding':hand,'strategy_l1':sum(abs(probabilities[ACTION_IDS.index(a)]-p) for a,p in expected.items()),
                     'ev_under_reference_continuations':mixture,
                     'best_action_gap':max(action_evs.values())-mixture})
    return {'scope':'Small belief-conditioned river games; fixed reference continuations, not full-game exploitability',
            'iterations':solution.complete_iterations,'nodes':solution.work.nodes,
            'bounded_response':br,'holdings':rows}


def collector_variance(trials: int = 64) -> dict:
    from scripts.strategy_collector_audit import small_root, varying_policy, exact_reference, enumerated_expectation
    root = small_root(3)
    exact = exact_reference(root)['reference']
    results = {}
    for threshold in (None,3):
        masses = enumerated_expectation(root,'partial_enumeration',enumerate_from_street=threshold)
        bias = max(abs(a-b) for key,row in exact.items() for a,b in zip(row['action_mass'],masses[key]['action_mass']))
        weights=[];nodes=0;started=time.perf_counter()
        for trial in range(trials):
            for player in root.players:
                for opponent in root.players:
                    if opponent == player: continue
                    walk=Traversal(varying_policy(1),random.Random(trial*101+player*7+opponent),10000)
                    walk.average_partial(root,player,opponent,lambda o,t,w:weights.append(w/2),enumerate_from_street=threshold)
                    nodes+=walk.nodes
        total=sum(weights)
        results[str(threshold)]={'exact_action_mass_max_error':bias,'records':len(weights),'nodes':nodes,
                                 'ess':total**2/sum(w*w for w in weights),'largest_share':max(weights)/total,
                                 'seconds':time.perf_counter()-started}
    return results


def run_closed_loop(output: Path, count: int, seed: int, *, iterations: int = 3,
                    updates: int = 32, traversals: int = 16) -> dict:
    """New explicit directory only; complete checkpoints and all raw metrics retained."""
    if output.exists():
        raise FileExistsError(f'Experiment already exists: {output}')
    output.mkdir(parents=True)
    torch.set_num_threads(1)
    config=diagnostic_config(output,count,seed,updates=updates,traversals=traversals)
    (output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    runner=TrainingRunner(config)
    results=[]
    track='hu' if count==2 else '3max'
    for iteration in range(iterations):
        row=runner.run_iteration()
        solver=runner.solvers[track]
        diagnostics={kind:dataset_metrics(getattr(solver,'advantage_model' if kind=='advantage' else 'average_model'),
                                         getattr(solver,'advantage_memory' if kind=='advantage' else 'strategy_memory'),solver.seed)
                     for kind in ('advantage','strategy')}
        checkpoint=output/f'cycle_{iteration+1}.pt'
        started=time.perf_counter();runner.save_checkpoint(checkpoint)
        results.append({'iteration':runner.iteration,'metrics':row,'datasets':diagnostics,
                        'checkpoint_bytes':checkpoint.stat().st_size,'save_seconds':time.perf_counter()-started})
        print(f'{track} seed={seed} cycle={runner.iteration}: {row["wall_seconds"]:.2f}s, nodes={row["tracks"][track]["nodes"]}',flush=True)
    from training.vps import export_live
    exported=export_live(runner)
    baseline={}
    for kind,constructor in (('advantage',AdvantageNetwork),('strategy',AveragePolicyNetwork)):
        source=Path(f'runs/strategic-training-audit/{track}-{kind}-unchanged.pt')
        if not source.exists():
            raise FileNotFoundError(f'Explicit shared audit baseline missing: {source}')
        model=constructor();model.load_state_dict(torch.load(source,weights_only=True,map_location='cpu'));model.eval()
        baseline[kind]=dataset_metrics(model,getattr(solver,'advantage_memory' if kind=='advantage' else 'strategy_memory'),solver.seed)
    report={'count':count,'seed':seed,'cycles':results,'shared_baseline':baseline,
            'river_reference':river_reference(solver.average_model,count),
            'rss_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
            'export':str(exported),'status':'bounded_training_not_general_strategic_validation'}
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


def profile_collectors(output: Path, checkpoint: Path, traversals: int = 32) -> dict:
    from ml.deep_cfr import generate_samples, FrozenStrategy
    from training.root_sampler import RootSampler
    runner = TrainingRunner.load_checkpoint(checkpoint)
    solver = runner.solvers['3max']
    sampler = RootSampler(3, 41001, runner.config['3max']['root_sampling'])
    tasks,_ = solver.traversal_tasks(sampler.task_factory(solver.players,traversals),traversals,
                                     200000,64,64*1024**2)
    report={}
    for threshold in (None,3,2):
        snapshot=replace(solver.snapshot(),collector_enumerate_from_street=threshold)
        strategy=FrozenStrategy(snapshot)
        started=time.perf_counter(); records=[];nodes=0
        for task in tasks:
            generated=generate_samples(snapshot,task,strategy)
            records.extend(generated.strategies);nodes+=generated.nodes
        report[str(threshold)]={'seconds':time.perf_counter()-started,'nodes':nodes,
                               'streets':{street:{'records':len(rows),**(weight_quality(rows) if rows else {})}
                                          for index,street in enumerate(STREETS)
                                          for rows in [[s for s in records if s.state.street==index]]}}
    output.write_text(json.dumps(report,indent=2)+'\n')
    return report


def browser_fixtures(output: Path, runs: dict[int, Path]) -> None:
    """Python posterior fixtures for real browser ONNX likelihoods on all routes."""
    from dataclasses import asdict
    from actions import legal_actions
    from training.evaluation import fixed_roots
    fixtures=[]
    for count,directory in runs.items():
        runner=TrainingRunner.load_checkpoint(directory/'cycle_5.pt')
        model=next(iter(runner.solvers.values())).average_model
        for _,hand in fixed_roots(count):
            actor=hand.current_player
            occupied=[*hand.board,*(c for p in hand.players.values() for c in p.cards),*hand.deck[-5:]]
            spare=[c for c in range(52) if c not in occupied][:2]
            holdings=[hand.actor.cards,tuple(spare)]
            priors={p:[{'cards':list(player.cards),'probability':1.}] for p,player in hand.players.items()}
            priors[actor]=[{'cards':list(h),'probability':w} for h,w in zip(holdings,(.4,.6))]
            likelihood=[]
            for h in holdings:
                view=hand.clone();view.actor.cards=h
                likelihood.append(model.probabilities(observe(view)))
            for action in legal_actions(hand):
                after=hand.clone();action.apply(after)
                index=ACTION_IDS.index(action.action_id)
                masses=[w*l[index] for w,l in zip((.4,.6),likelihood)]
                before_raw,after_raw=asdict(hand),asdict(after)
                before_raw['pending']=sorted(hand.pending);after_raw['pending']=sorted(after.pending)
                fixtures.append({'count':count,'street':hand.street,'actor':actor,
                                 'state':after_raw,'ranges':priors,
                                 'transitions':[{'before':before_raw,'after':after_raw,'action':asdict(action)}],
                                 'expected':[w/sum(masses) for w in masses]})
    output.write_text(json.dumps(fixtures,allow_nan=False)+'\n')


def fixed_replay_study(output: Path, checkpoint: Path) -> dict:
    """Compare bounded incremental budgets and Adam retention on one frozen dataset."""
    from ml.street_training import fit_specialists
    runner=TrainingRunner.load_checkpoint(checkpoint)
    solver=next(iter(runner.solvers.values()))
    report={'source_checkpoint':str(checkpoint),'split':'public-root SHA256 grouped 80/10/10',
            'scope':'Historical supervised fidelity; not equilibrium convergence','models':{}}
    for kind,attribute,memory_attribute in (('advantage','advantage_model','advantage_memory'),
                                             ('strategy','average_model','strategy_memory')):
        original=getattr(solver,attribute);memory=getattr(solver,memory_attribute)
        rows={'unchanged':dataset_metrics(original,memory,solver.seed)}
        for updates,retain in ((64,True),(256,True),(64,False)):
            schedules=copy.deepcopy(solver.street_config[kind]);states=copy.deepcopy(solver.specialist_states[kind])
            for street in STREETS:
                schedules[street].update(every=1,min_new_samples=1,max_updates=updates)
                states[street]['seen_at_fit']=0
                if not retain:states[street]['optimizer']=None
            started=time.process_time()
            model,new_states,metrics=fit_specialists(original,memory,states,schedules,version=solver.version+1,
                                                   seed=solver.seed,batch_size=64,learning_rate=runner.config['learning_rate'],cache_encoding=True,allow_frozen_replay=True)
            name=f'{updates}_updates_{"persistent" if retain else "reset"}_adam'
            rows[name]={'cpu_seconds':time.process_time()-started,'fit':metrics,'datasets':dataset_metrics(model,memory,solver.seed)}
            artifact=output.parent/f'{output.stem}-{kind}-{name}.pt'
            torch.save({'weights':model.state_dict(),'optimizer_states':new_states,'source':str(checkpoint)},artifact)
            if kind=='strategy':rows[name]['river_reference']=river_reference(model,len(solver.players))
        report['models'][kind]=rows
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


class AdvantagePolicy:
    def __init__(self, model):
        self.model=model

    def probabilities(self, observation):
        with torch.no_grad():
            values=self.model(encode_batch([observation]))[0].tolist()
        return regret_matching(values,observation.legal_mask)


def strategic_probes(output: Path, runs: dict[str, Path], *, samples: int = 128) -> dict:
    from training.preflop_diagnostics import opening_matrix, paired_opening_values
    from poker_game_expresso import HandState
    result={}
    for track,directory in runs.items():
        count=2 if track=='hu' else 3
        runner=TrainingRunner.load_checkpoint(directory/'cycle_5.pt');solver=runner.solvers[track]
        old_a=AdvantageNetwork();old_b=AveragePolicyNetwork()
        for kind,model in (('advantage',old_a),('strategy',old_b)):
            model.load_state_dict(torch.load(f'runs/strategic-training-audit/{track}-{kind}-unchanged.pt',weights_only=True))
            model.eval()
        policies={'specialist_A':AdvantagePolicy(solver.advantage_model),'specialist_B':solver.average_model,
                  'shared_A':AdvantagePolicy(old_a),'shared_B':old_b}
        matrices={name:opening_matrix(policy,count,25.) for name,policy in policies.items()}
        rivers={name:river_reference(policy,count) for name,policy in policies.items()}
        probes={}
        for label,holding in {'AA':(48,49),'KK':(44,45),'AKs':(48,44),'22':(0,1),'72o':(20,1),'32o':(4,1)}.items():
            reference=paired_opening_values(old_b,count,25.,holding,samples,seed=61001)
            root=HandState.start({p:25. for p in range(count)},0,random.Random(61001));root.actor.cards=holding
            actions=reference['actions']
            predictions={}
            for name,policy in policies.items():
                probabilities=policy.probabilities(observe(root))
                mixture=sum(probabilities[ACTION_IDS.index(a)]*r['ev']['mean'] for a,r in actions.items())
                predictions[name]={'probabilities':probabilities,'ev_under_fixed_shared_B_continuations':mixture}
            probes[label]={'reference':reference,'predictions':predictions}
        result[track]={'matrices':matrices,'river':rivers,'openings':probes,
                       'scope':'Root policy decisions under identical fixed shared-B continuations; sampled action values, no equilibrium claim'}
        print(track,'strategic probes complete',flush=True)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


def postflop_probes(output: Path, runs: dict[str, Path]) -> dict:
    """Independent flop/turn action values under fixed, explicitly non-equilibrium continuations."""
    from training.evaluation import fixed_roots
    from hybrid.policy_source import NeuralAveragePolicy
    from hybrid.ranges import HandRange, JointRanges
    from hybrid.search import search
    from hybrid.state import ComputeBudget
    result = {}
    torch.set_num_threads(1)
    for track, directory in runs.items():
        count = 2 if track == 'hu' else 3
        solver = TrainingRunner.load_checkpoint(directory/'cycle_5.pt').solvers[track]
        old_a, old_b = AdvantageNetwork(), AveragePolicyNetwork()
        for kind, model in (('advantage', old_a), ('strategy', old_b)):
            model.load_state_dict(torch.load(f'runs/strategic-training-audit/{track}-{kind}-unchanged.pt', weights_only=True))
            model.eval()
        policies = {'specialist_A': AdvantagePolicy(solver.advantage_model), 'specialist_B': solver.average_model,
                    'shared_A': AdvantagePolicy(old_a), 'shared_B': old_b}
        continuation = NeuralAveragePolicy(old_b.probabilities, f'fixed-shared-{track}-audit')
        rows = []
        for label, root in fixed_roots(count):
            if root.street not in ('FLOP', 'TURN'):
                continue
            hero = root.current_player
            dead = {*root.board, *(c for p in root.players.values() for c in p.cards)}
            available = [c for c in range(52) if c not in dead]
            ranges = {p: HandRange({player.cards: 1.}) if p == hero else
                      HandRange({player.cards: 1., tuple(available[2*p:2*p+2]): 1.})
                      for p, player in root.players.items()}
            private = JointRanges(ranges, tuple(root.board))
            measurements = []
            for samples, seed in ((128, 71101), (512, 82101)):
                started = time.perf_counter()
                calculated = search(root, private, hero, {p: continuation for p in root.players},
                                    ComputeBudget(max_nodes=500000, samples=samples, max_depth=1, seed=seed))
                measurements.append({'samples': calculated.work.samples, 'nodes': calculated.work.nodes,
                                     'seconds': time.perf_counter()-started, 'action_ev': calculated.action_ev,
                                     'standard_errors': calculated.standard_errors, 'warnings': calculated.warnings})
            values = measurements[-1]['action_ev']
            obs = observe(root)
            decisions = {}
            for name, policy in policies.items():
                strategy = policy.probabilities(obs)
                ev = sum(strategy[ACTION_IDS.index(a)]*v for a,v in values.items())
                decisions[name] = {'probabilities': strategy, 'conditional_ev': ev,
                                   'gap_to_best_sampled_action': max(values.values())-ev}
            rows.append({'context': label, 'street': root.street, 'measurements': measurements,
                         'predictions': decisions,
                         'low_high_action_ev_rmse': (sum((measurements[0]['action_ev'][a]-v)**2
                                                        for a,v in values.items())/len(values))**.5})
        result[track] = rows
    report = {'scope': 'Two independent chance seeds; fixed shared-B continuations, explicit two-combo opponent factors. '
                       'Conditional EV is not historical CFR regret or equilibrium EV.', 'tracks': result}
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


def context_diagnostics(output: Path, checkpoint: Path) -> dict:
    """Held-out public-root errors split by betting context, without fitting test data."""
    from collections import defaultdict
    from training.metrics import decision_context
    runner = TrainingRunner.load_checkpoint(checkpoint)
    solver = next(iter(runner.solvers.values()))
    report = {}
    for kind, bundle, replay in (('advantage', solver.advantage_model, solver.advantage_memory),
                                  ('strategy', solver.average_model, solver.strategy_memory)):
        streets = {}
        for street in STREETS:
            grouped = defaultdict(list)
            for sample in replay.memories[street].samples:
                grouped[(grouped_partition(sample, solver.seed), decision_context(sample.state))].append(sample)
            rows = []
            for (split, context), samples in sorted(grouped.items()):
                model = bundle.specialists[street]
                with torch.no_grad():
                    outputs = model(encode_batch([s.state for s in samples])).tolist()
                l1 = []
                for s, prediction in zip(samples, outputs):
                    if kind == 'advantage':
                        prediction = regret_matching(prediction, s.state.legal_mask)
                        target = regret_matching(s.target, s.state.legal_mask)
                    else:
                        target = s.target
                    l1.append(sum(abs(a-b) for a,b in zip(prediction, target)))
                rows.append({'partition': split, 'context': context, 'records': len(samples),
                             'public_root_groups': len({s.root_group for s in samples}),
                             'weighted_loss': evaluate_loss(model, samples, 128),
                             'unweighted_mean_strategy_l1': sum(l1)/len(l1), **weight_quality(samples)})
            streets[street] = rows
        report[kind] = streets
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report
