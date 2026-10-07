"""Process workers generate samples under one frozen version; no local learning."""
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager, nullcontext
from typing import Iterator
import io
import multiprocessing
import time
import torch
from ml.memory import sample_bytes
from ml.deep_cfr import ModelSnapshot, TraversalTask, GeneratedSamples, FrozenStrategy, generate_samples


def _snapshot_bytes(snapshot: ModelSnapshot) -> bytes:
    buffer = io.BytesIO()
    torch.save({"version": snapshot.version, "objective": snapshot.objective,
                "advantage_weights": snapshot.advantage_weights,
                "uniform_initial": snapshot.uniform_initial, "player_count": snapshot.player_count}, buffer)
    return buffer.getvalue()


_worker_payload: bytes | None = None
_worker_snapshot: ModelSnapshot | None = None
_worker_strategy: FrozenStrategy | None = None


def _worker_initialize() -> None:
    torch.set_num_threads(1)


@contextmanager
def traversal_workers(workers: int) -> Iterator[ProcessPoolExecutor | None]:
    """Own worker lifetime across a training session, not an outer iteration."""
    if type(workers) is not int or workers < 1:
        raise ValueError(f"Invalid traversal worker count: {workers}")
    if workers == 1:
        yield None
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=_worker_initialize) as executor:
            yield executor


def _worker_generate(payload: bytes, task: TraversalTask) -> GeneratedSamples:
    global _worker_payload, _worker_snapshot, _worker_strategy
    started, cpu_started = time.perf_counter(), time.process_time()
    # Bytes identify all weights and schema fields, including track and version.
    # A same-number snapshot from another track cannot reuse a stale model.
    if payload != _worker_payload:
        raw = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
        snapshot = ModelSnapshot(raw["version"], raw["objective"], raw["advantage_weights"], raw["uniform_initial"], raw["player_count"])
        strategy = FrozenStrategy(snapshot)
        _worker_snapshot, _worker_strategy, _worker_payload = snapshot, strategy, payload
    result = generate_samples(_worker_snapshot, task, _worker_strategy)
    result.worker_seconds = time.perf_counter() - started
    result.worker_cpu_seconds = time.process_time() - cpu_started
    return result


def collect_samples(snapshot: ModelSnapshot, tasks: list[TraversalTask], workers: int = 1,
                    aggregate_byte_budget: int = 256 * 1024 * 1024,
                    executor: ProcessPoolExecutor | None = None) -> list[GeneratedSamples]:
    if workers < 1 or not tasks or len({t.task_id for t in tasks}) != len(tasks):
        raise ValueError(f"Invalid sample batch: workers={workers}, tasks={len(tasks)}")
    if type(aggregate_byte_budget) is not int or aggregate_byte_budget < 1:
        raise ValueError(f"Aggregate sample byte budget must be positive: {aggregate_byte_budget}")
    results: list[GeneratedSamples] = []
    used_bytes = 0
    def retain(result: GeneratedSamples) -> None:
        nonlocal used_bytes
        if result.version != snapshot.version:
            raise ValueError(f"Worker returned model version {result.version}; expected {snapshot.version}")
        size = sum(sample_bytes(s) for s in result.advantages + result.strategies)
        if used_bytes + size > aggregate_byte_budget:
            raise MemoryError(f"Aggregate generated samples require {used_bytes + size} accounted bytes; "
                              f"budget={aggregate_byte_budget}; no partial batch returned")
        used_bytes += size
        results.append(result)
    if workers == 1:
        if executor is not None:
            raise ValueError("A process executor cannot be supplied for serial traversal")
        torch.set_num_threads(1)
        strategy = FrozenStrategy(snapshot)
        for task in tasks:
            retain(generate_samples(snapshot, task, strategy))
    else:
        payload = _snapshot_bytes(snapshot)
        remaining = iter(tasks)
        with (nullcontext(executor) if executor is not None else traversal_workers(workers)) as pool:
            pending = {pool.submit(_worker_generate, payload, task)
                       for task in (next(remaining) for _ in range(min(workers, len(tasks))))}
            while pending:
                completed, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in completed:
                    retain(future.result())
                    task = next(remaining, None)
                    if task is not None:
                        pending.add(pool.submit(_worker_generate, payload, task))
    results.sort(key=lambda r: r.task_id)
    return results
