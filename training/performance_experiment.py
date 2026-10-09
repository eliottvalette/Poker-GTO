"""Reproducible CPU performance sweeps on immutable training checkpoints."""
from __future__ import annotations

import copy
import cProfile
import hashlib
import io
import json
from pathlib import Path
import platform
import pstats
import random
import resource
import statistics
import time
from typing import Callable, TypeVar

import torch

from ml.deep_cfr import FrozenStrategy, GeneratedSamples, generate_samples
from ml.train import fit
from scripts.parallel_cfr import collect_samples, traversal_workers
from training.checkpoint import atomic_bytes
from training.runner import TrainingRunner


_Result = TypeVar("_Result")


def profile_call(output: Path, name: str, operation: Callable[[], _Result]) -> _Result:
    profiler = cProfile.Profile()
    result = profiler.runcall(operation)
    profiler.dump_stats(str(output / f'{name}.prof'))
    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).sort_stats('cumulative').print_stats(50)
    (output / f'{name}.txt').write_text(stream.getvalue())
    return result


def same_samples(expected: list[GeneratedSamples], actual: list[GeneratedSamples]) -> bool:
    return [(r.task_id,r.nodes,r.value,r.advantages,r.strategies,r.coverage) for r in expected] == [
        (r.task_id,r.nodes,r.value,r.advantages,r.strategies,r.coverage) for r in actual]


def save_report(output: Path, report: dict) -> None:
    atomic_bytes(output/'report.json',(json.dumps(report,indent=2)+'\n').encode())


def sweep(checkpoint: Path, output: Path, *, track: str, repeats: int = 3, include_fits: bool = True) -> dict:
    if repeats < 2 or track not in ('hu','3max'):
        raise ValueError('At least two repeats and an explicit supported track required')
    output.mkdir(parents=True,exist_ok=True)
    report = {'checkpoint':str(checkpoint),'sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              'platform':platform.platform(),'torch':torch.__version__,'workers':[],'fits':[],
              'max_cpu_parallelism':8,'repeats':repeats}
    torch.set_num_threads(1)
    runner = TrainingRunner.load_checkpoint(checkpoint)
    solver=runner.solvers[track]; snapshot=solver.snapshot()
    report['iteration']=runner.iteration
    sampler=copy.deepcopy(runner.samplers[track]);sampler.policy=FrozenStrategy(snapshot)
    per_player=runner.config[track]['traversals_per_player']
    tasks,_=solver.traversal_tasks(sampler.task_factory(solver.players,per_player),per_player,
        runner.config['max_nodes'],runner.config['max_depth'],runner.config['sample_byte_budget'])
    # Both traverser seats, fixed prefix selection, no runtime-based task cherry-picking.
    selected=[tasks[p*per_player+i] for p in range(len(solver.players)) for i in range(min(32,per_player))]
    report['task_ids']=[t.task_id for t in selected]
    expected=collect_samples(snapshot,selected,1,runner.config['generation_byte_budget'])
    report['nodes']=sum(r.nodes for r in expected)
    report['samples']=sum(len(r.advantages)+len(r.strategies) for r in expected)
    print('workload',track,report['iteration'],len(selected),report['nodes'],report['samples'],flush=True)
    heavy=selected[max(range(len(expected)),key=lambda i:expected[i].nodes)]
    strategy=FrozenStrategy(snapshot)
    profile_call(output,'generation',lambda:generate_samples(snapshot,heavy,strategy))
    for workers in (1,2,4,6,8):
        with traversal_workers(workers) as executor:
            for chunk_size in (32,128):
                times=[];cpus=[]
                for trial in range(repeats):
                    start,cpu=time.perf_counter(),time.process_time()
                    actual=[]
                    for offset in range(0,len(selected),chunk_size):
                        actual.extend(collect_samples(snapshot,selected[offset:offset+chunk_size],workers,
                                      runner.config['generation_byte_budget'],executor))
                    elapsed=time.perf_counter()-start
                    cpus.append(time.process_time()-cpu+sum(r.worker_cpu_seconds for r in actual) if workers>1 else time.process_time()-cpu)
                    times.append(elapsed)
                    if not same_samples(expected,actual):
                        raise RuntimeError(f'Sample parity failed: workers={workers}, chunk={chunk_size}')
                    print('generation',workers,chunk_size,trial,round(elapsed,3),flush=True)
                report['workers'].append({'workers':workers,'chunk_size':chunk_size,'seconds':times,
                    'cpu_seconds':cpus,'median_warm_seconds':statistics.median(times[1:]),'exact_sample_parity':True,
                    'cold_start_in_first_measurement':workers>1 and chunk_size==32})
                save_report(output,report)
    torch.optim.Adam(solver.advantage_model.parameters())
    for kind,memory,source in ((('advantage',solver.advantage_memory,solver.advantage_model),
                               ('strategy',solver.strategy_memory,solver.average_model)) if include_fits else ()):
        samples=memory.samples
        probe=[s.state for s in random.Random(481).sample(samples,min(256,len(samples)))]
        from ml.model import encode_batch
        baseline_prediction=None
        for threads in (1,2,4,8):
            for batch_size in (64,128,256):
                times=[];losses=[];max_difference=[]
                updates=8192//batch_size
                for trial in range(repeats):
                    torch.set_num_threads(threads)
                    model=copy.deepcopy(source)
                    start=time.perf_counter()
                    metrics=fit(model,samples,4,batch_size,481,max_updates=updates,cache_encoding=True)
                    elapsed=time.perf_counter()-start
                    with torch.inference_mode():
                        prediction=model(encode_batch(probe)).clone()
                    if batch_size==64 and threads==1:
                        baseline_prediction=prediction
                    times.append(elapsed);losses.append(metrics)
                    max_difference.append(float((prediction-baseline_prediction).abs().max()))
                    print('fit',kind,threads,batch_size,trial,round(elapsed,3),flush=True)
                report['fits'].append({'kind':kind,'threads':threads,'batch_size':batch_size,'updates':updates,
                    'seconds':times,'median_seconds':statistics.median(times),'metrics':losses,
                    'prediction_max_difference_from_threads1_batch64':max_difference})
                save_report(output,report)
        torch.set_num_threads(1)
        profile_call(output,f'{kind}-fit',lambda:fit(copy.deepcopy(source),samples,4,64,481,max_updates=128,cache_encoding=True))
    report['peak_parent_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report['checkpoint_unchanged']=hashlib.sha256(checkpoint.read_bytes()).hexdigest()==report['sha256']
    save_report(output,report)
    return report


def _worker_identity() -> int:
    import os
    time.sleep(.05)
    return os.getpid()


def iteration_trial(checkpoint: Path, output: Path, *, track: str, workers: int,
                    threads: int, chunk_size: int, profiled: bool = False) -> dict:
    """One complete next iteration with explicit isolated output, no source mutation."""
    if output.exists():
        raise FileExistsError(f'Iteration benchmark output exists: {output}')
    torch.set_num_threads(1)
    runner=TrainingRunner.load_checkpoint(checkpoint)
    runner.config.update(output_dir=str(output),workers=workers,trainer_threads=threads,
                         generation_batch_size=chunk_size)
    output.mkdir(parents=True)
    with traversal_workers(workers) as executor:
        if executor is not None:
            identities = set()
            for _ in range(10):
                identities.update(f.result() for f in [executor.submit(_worker_identity) for _ in range(workers * 4)])
                if len(identities) == workers:
                    break
            else:
                raise RuntimeError(f'Worker startup incomplete: {len(identities)}/{workers}')
        # Initialize children before measuring; the warmup has no solver/replay side effects.
        from training.evaluation import fixed_roots
        from ml.deep_cfr import TraversalTask
        root=fixed_roots(len(runner.solvers[track].players))[-1][1]
        collect_samples(runner.solvers[track].snapshot(),[
            TraversalTask(i,root.current_player,root,7400+i,20000,64) for i in range(workers)],workers,executor=executor)
        started=time.perf_counter()
        row=(profile_call(output,'iteration',lambda:runner.run_iteration(executor=executor)) if profiled
             else runner.run_iteration(executor=executor))
        elapsed=time.perf_counter()-started
    checkpoint_start=time.perf_counter()
    runner.save_checkpoint(output/'checkpoint.pt')
    checkpoint_seconds=time.perf_counter()-checkpoint_start
    digest=hashlib.sha256()
    solver=runner.solvers[track]
    for model in (solver.advantage_model,solver.average_model):
        for name,value in model.state_dict().items():
            digest.update(name.encode());digest.update(value.numpy().tobytes())
    replay_digest=hashlib.sha256()
    for memory in (solver.advantage_memory,solver.strategy_memory):
        for sample in memory.samples:
            replay_digest.update(repr(sample).encode())
    report={'elapsed_seconds':elapsed,'checkpoint_seconds':checkpoint_seconds,'profiled':profiled,
            'workers':workers,'threads':threads,'chunk_size':chunk_size,
            'model_sha256':digest.hexdigest(),'replay_sha256':replay_digest.hexdigest(),
            'metrics':row,'peak_parent_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    save_report(output,report)
    print('iteration',workers,threads,chunk_size,elapsed,checkpoint_seconds,flush=True)
    return report


def native_fit_profile(checkpoint: Path, output: Path, *, track: str) -> None:
    """Attribute ten warmed optimizer steps to native CPU operators."""
    from ml.model import encode_batch
    from ml.train import _loss_with_weights, _mean_weight, _sample_weight
    torch.set_num_threads(1)
    runner=TrainingRunner.load_checkpoint(checkpoint)
    solver=runner.solvers[track]
    output.mkdir(parents=True,exist_ok=True)
    for kind,memory,source in (('advantage',solver.advantage_memory,solver.advantage_model),
                               ('strategy',solver.strategy_memory,solver.average_model)):
        rows=random.Random(981).sample(memory.samples,64)
        batch=encode_batch([s.state for s in rows]);mean=_mean_weight(rows)
        weights=[_sample_weight(s)/mean for s in rows]
        model=copy.deepcopy(source);optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)
        def step():
            optimizer.zero_grad()
            _loss_with_weights(model,rows,weights,batch).backward();optimizer.step()
        step()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],record_shapes=True) as profile:
            for _ in range(10):
                step()
        profile.export_chrome_trace(str(output/f'{kind}-native.json'))
        (output/f'{kind}-native.txt').write_text(profile.key_averages().table(sort_by='self_cpu_time_total',row_limit=30))


def fit_confirmation(checkpoint: Path, output: Path, *, track: str) -> dict:
    """Repeat the relevant unchanged-minibatch choices after source optimization."""
    torch.set_num_threads(1)
    runner=TrainingRunner.load_checkpoint(checkpoint);solver=runner.solvers[track]
    output.mkdir(parents=True,exist_ok=True)
    report={'track':track,'iteration':runner.iteration,'fits':[]}
    for kind,memory,source in (('advantage',solver.advantage_memory,solver.advantage_model),
                               ('strategy',solver.strategy_memory,solver.average_model)):
        samples=memory.samples
        for threads in (1,4,8):
            times=[];metrics=[]
            for _ in range(2):
                torch.set_num_threads(threads);model=copy.deepcopy(source)
                started=time.perf_counter()
                metrics.append(fit(model,samples,4,64,481,max_updates=128,cache_encoding=True))
                times.append(time.perf_counter()-started)
            report['fits'].append({'kind':kind,'threads':threads,'seconds':times,'metrics':metrics})
            save_report(output,report)
            print('fit confirmation',track,kind,threads,times,flush=True)
    torch.set_num_threads(1)
    for kind,memory,source in (('advantage',solver.advantage_memory,solver.advantage_model),
                               ('strategy',solver.strategy_memory,solver.average_model)):
        profile_call(output,kind+'-fit',lambda:fit(copy.deepcopy(source),memory.samples,4,64,481,max_updates=128,cache_encoding=True))
    return report


def iteration_grid(checkpoint: Path, output: Path, *, track: str) -> dict:
    """Reverse-order repeated whole-iteration comparisons from the same immutable state."""
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    torch.set_num_threads(1)
    base=TrainingRunner.load_checkpoint(checkpoint)
    report={'checkpoint':str(checkpoint),'track':track,'rows':[]}
    for index,workers in enumerate((8,6,4,4,6,8)):
        runner=copy.copy(base)
        runner.config=copy.deepcopy(base.config)
        runner.config.update(output_dir=str(output/f'trial-{index}'),workers=workers,trainer_threads=1,generation_batch_size=128)
        with traversal_workers(workers) as executor:
            identities=set()
            for _ in range(10):
                identities.update(f.result() for f in [executor.submit(_worker_identity) for _ in range(workers*4)])
                if len(identities)==workers:
                    break
            else:
                raise RuntimeError('Worker startup incomplete')
            started=time.perf_counter();metric=runner.run_iteration(executor=executor);elapsed=time.perf_counter()-started
        solver=runner.solvers[track]
        digest=hashlib.sha256()
        for model in (solver.advantage_model,solver.average_model):
            for name,value in model.state_dict().items():
                digest.update(name.encode());digest.update(value.numpy().tobytes())
        report['rows'].append({'workers':workers,'seconds':elapsed,'model_sha256':digest.hexdigest(),
            'generation_seconds':metric['tracks'][track]['generation_seconds'],'fit_seconds':metric['tracks'][track]['fit_seconds'],
            'nodes':metric['tracks'][track]['nodes']})
        save_report(output,report)
        print('whole grid',track,workers,round(elapsed,3),flush=True)
    if len({r['model_sha256'] for r in report['rows']})!=1:
        raise RuntimeError('Whole iteration worker choices changed fitted weights')
    return report
