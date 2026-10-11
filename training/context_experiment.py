"""Matched frozen-source replay allocation experiment; no production publication."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import time
import torch
from ml.context_replay import ContextReplay
from ml.memory import ReservoirMemory
from ml.street_training import grouped_partition
from ml.fixed_validation import partition_coverage
from ml.train import fit, evaluate_loss
from training.runner import TrainingRunner
from training.poker_evaluation import evaluate_policy, EvaluationBudget


def compare_context_replay(checkpoint: Path, output: Path) -> dict:
    torch.set_num_threads(1)
    runner=TrainingRunner.load_checkpoint(checkpoint)
    track=next(iter(runner.solvers));solver=runner.solvers[track]
    report={'source':str(checkpoint),'seed':451,'update_budget_per_model':64,'models':{}}
    policies={}
    caps={'first_open':200,'facing_open':60,'facing_3bet':35,'facing_4bet_or_more':25,'facing_jam':40,'other':40}
    for kind,owner in [('advantage','advantage'),('strategy','average')]:
        source=getattr(solver,'advantage_memory' if kind=='advantage' else 'strategy_memory').memories['PREFLOP'].samples
        model=getattr(solver,owner+'_model')
        evaluation=[s for s in source if grouped_partition(s,solver.seed)=='test']
        report['models'][kind]={}
        for allocation in ('uniform','contexts'):
            memory=(ReservoirMemory(400,451,kind,'hand_chip_delta',8*1024**2) if allocation=='uniform'
                    else ContextReplay(caps,8*1024**2,kind,451))
            for sample in source:memory.add(sample)
            rows=memory.samples
            splits={k:[s for s in rows if grouped_partition(s,solver.seed)==k] for k in ('train','validation','test')}
            network=copy.deepcopy(model.specialists['PREFLOP'])
            optimizer=torch.optim.Adam(network.parameters(),lr=runner.config['learning_rate'])
            optimizer.load_state_dict(copy.deepcopy(solver.specialist_states[kind]['PREFLOP']['optimizer']))
            started=time.perf_counter()
            metrics=fit(network,splits['train']+splits['validation'],100,64,451,max_updates=64,
                        cache_encoding=True,optimizer=optimizer,
                        explicit_partitions=(splits['train'],splits['validation']))
            metrics.update(seconds=time.perf_counter()-started,coverage=partition_coverage(splits),
                           fixed_test_loss=evaluate_loss(network,evaluation,128) if evaluation else None)
            report['models'][kind][allocation]=metrics
            if kind=='strategy':
                policy=copy.deepcopy(model);policy.specialists['PREFLOP']=network.eval();policies[allocation]=policy
    measured,_=evaluate_policy(policies['contexts'],policies['uniform'],len(solver.players),
                              EvaluationBudget(groups=16,cpu_seconds=60,wall_seconds=90,seed_offset=7100003))
    report['matched_poker']=measured
    report['limitation']='Small frozen stream; equal optimizer updates, measured wall costs; not general strategic acceptance'
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2)+'\n')
    return report
