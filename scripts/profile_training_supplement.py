"""Production-sized frozen traversal lots and independent-buffer checkpoint costs."""
from __future__ import annotations
import copy
from dataclasses import replace
import json
from pathlib import Path
import time
import torch
from scripts.profile_training import measured, profiled, tasks_for, save_benchmark_checkpoint, OUTPUT
from scripts.parallel_cfr import collect_samples, _snapshot_bytes, _worker_generate
from training.config import load_config
from training.runner import TrainingRunner
from training.checkpoint import atomic_bytes
from ml.memory import sample_bytes


def main():
    torch.set_num_threads(1)
    report={'profiles':{},'scaling':[],'scope':'full-iteration task counts under a frozen learned snapshot; no pilot','clone_experiment':{}}
    config=load_config('configs/deep_cfr_pilot.json')
    bootstrap=copy.deepcopy(config)
    bootstrap.update(output_dir=str(OUTPUT/'bootstrap'),workers=1,checkpoint_every=100,evaluation_every=100)
    for name in ('3max','hu'): bootstrap[name]['traversals_per_player']=4
    runner=TrainingRunner(bootstrap)
    runner.run_iteration()
    models={len(s.players):s.average_model for s in runner.solvers.values()}
    samples={}
    for name,count in (('3max',3),('hu',2)):
        amount=count*128
        timing,(tasks,coverage)=measured(lambda:tasks_for(count,amount,config,models))
        # Match production player-block task assignment, not round-robin assignment.
        tasks=[replace(task,player=task.task_id//128) for task in tasks]
        report[name+'_roots']={**timing,'coverage':coverage}
        snapshot=runner.solvers[name].snapshot()
        payload=_snapshot_bytes(snapshot)
        profiled(name+'_worker_deserialization',lambda:[_worker_generate(payload,task) for task in tasks[:16]],report)
        for workers in (4,8):
            print(f'Full iteration {name}: {amount} tasks, {workers} workers',flush=True)
            timing,results=measured(lambda:collect_samples(snapshot,tasks,workers),2)
            report['scaling'].append({'track':name,'tasks':amount,'workers':workers,**timing,
                'nodes':sum(r.nodes for r in results),'samples':sum(len(r.advantages)+len(r.strategies) for r in results),
                'worker_cpu_seconds':sum(r.worker_cpu_seconds for r in results)})
            samples[name]=[s for result in results for s in (*result.advantages,*result.strategies)]
        atomic_bytes(OUTPUT/'supplement.json',(json.dumps(report,indent=2)+'\n').encode())
    # Preserve all mutable HandState fields independently; only immutable values shared.
    hand=tasks[0].root
    def selective_clone():
        return replace(hand,players={i:replace(p) for i,p in hand.players.items()},deck=list(hand.deck),
             initial_stacks=dict(hand.initial_stacks),board=list(hand.board),pending=set(hand.pending),
             history=list(hand.history),awards=dict(hand.awards))
    assert selective_clone()==hand.clone()
    report['clone_experiment']['deepcopy_10000'],_=measured(lambda:[hand.clone() for _ in range(10000)],3)
    report['clone_experiment']['selective_10000'],_=measured(lambda:[selective_clone() for _ in range(10000)],3)
    report['clone_experiment']['scope']='isolated prototype; no production engine change or full correctness certification'
    print('Construct independently owned replay buffers for checkpoint measurement',flush=True)
    start=time.perf_counter()
    for name,solver in runner.solvers.items():
        for memory_key,memory in (*solver.advantage_memory.items(),('strategy',solver.strategy_memory)):
            source=[s for s in samples[name] if s.kind==memory.kind and (memory.kind=='strategy' or s.player==memory_key)]
            if not source: raise RuntimeError(f'No profiling samples for {name}/{memory_key}')
            records=[]
            for index in range(10000):
                s=source[index%len(source)]
                state=replace(s.state,numeric_data=memoryview(s.state.numeric_data).tobytes(),
                              history_data=memoryview(s.state.history_data).tobytes())
                records.append(replace(s,state=state))
            memory.samples=records
            memory.seen=len(records)
            memory._sample_sizes=[sample_bytes(s) for s in records]
            memory.used_bytes=sum(memory._sample_sizes)
    report['unique_buffer_setup_seconds']=time.perf_counter()-start
    path=OUTPUT/'capacity_checkpoint_unique_buffers.pt'
    report['checkpoint_unique_buffers'],_=measured(lambda:save_benchmark_checkpoint(runner,path),2)
    report['checkpoint_unique_buffers']['bytes']=path.stat().st_size
    profiled('checkpoint_unique_buffers',lambda:save_benchmark_checkpoint(runner,path),report)
    report['checkpoint_unique_buffers']['scope']='independent bytes per replay record; repeated numeric values; intentionally incompatible profiling checkpoint'
    atomic_bytes(OUTPUT/'supplement.json',(json.dumps(report,indent=2)+'\n').encode())
    print('Supplement complete',flush=True)

if __name__=='__main__': main()
