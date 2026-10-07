"""Process workers generate samples under one frozen version; no local learning."""
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import io
import multiprocessing
import torch
from ml.memory import sample_bytes
from ml.deep_cfr import ModelSnapshot, TraversalTask, GeneratedSamples, generate_samples


def _snapshot_bytes(snapshot: ModelSnapshot) -> bytes:
    buffer = io.BytesIO()
    torch.save({"version": snapshot.version, "objective": snapshot.objective,
                "advantage_weights": snapshot.advantage_weights,
                "uniform_initial": snapshot.uniform_initial, "player_count": snapshot.player_count}, buffer)
    return buffer.getvalue()


def _worker_generate(payload: bytes, task: TraversalTask) -> GeneratedSamples:
    # Ordinary bytes have explicit ownership; torch multiprocessing tensor transport
    # would otherwise create an implicit shared-memory manager for frozen weights.
    raw = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    snapshot = ModelSnapshot(raw["version"], raw["objective"], raw["advantage_weights"], raw["uniform_initial"], raw["player_count"])
    return generate_samples(snapshot, task)


def collect_samples(snapshot: ModelSnapshot, tasks: list[TraversalTask], workers: int = 1,
                    aggregate_byte_budget: int = 256 * 1024 * 1024) -> list[GeneratedSamples]:
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
        for task in tasks:
            retain(generate_samples(snapshot, task))
    else:
        payload = _snapshot_bytes(snapshot)
        remaining = iter(tasks)
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
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
