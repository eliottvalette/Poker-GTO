from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
from dataclasses import dataclass
from typing import Any

from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import cfr_solver
from cfr_solver import CFRPlusSolver, N_ACTIONS

VectorTable = dict[int, list[float]]
VisitTable = dict[int, int]
SolverState = tuple[VectorTable, VectorTable, VisitTable]
DEFAULT_TOTAL_ITERATIONS = 1_000_000
DEFAULT_ITERATIONS_PER_WORKER = 1_000


@dataclass(frozen=True)
class WorkerResult:
    regret_delta: VectorTable
    strategy_delta: VectorTable
    visit_delta: VisitTable
    elapsed_seconds: float
    infosets: int
    visits: int


def parse_stacks(value: str) -> tuple[int, int, int]:
    parts = value.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected three comma-separated stacks, e.g. 100,100,100")
    try:
        return tuple(int(part) for part in parts)  # type: ignore[return-value]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("stacks must be integers") from exc


def empty_state() -> SolverState:
    return {}, {}, {}


def is_empty_state(state: SolverState) -> bool:
    regret_sum, strategy_sum, visit_count = state
    return not regret_sum and not strategy_sum and not visit_count


def state_to_solver(state: SolverState, seed: int, stacks: tuple[int, int, int]) -> CFRPlusSolver:
    solver = CFRPlusSolver(seed=seed, stacks=stacks)
    regret_sum, strategy_sum, visit_count = state
    for key, vector in regret_sum.items():
        solver.regret_sum[key] = vector.copy()
    for key, vector in strategy_sum.items():
        solver.strategy_sum[key] = vector.copy()
    for key, visits in visit_count.items():
        solver.visit_count[key] = visits
    return solver


def solver_to_state(solver: CFRPlusSolver) -> SolverState:
    regret_sum = {key: vector.copy() for key, vector in solver.regret_sum.items()}
    strategy_sum = {key: vector.copy() for key, vector in solver.strategy_sum.items()}
    visit_count = dict(solver.visit_count)
    return regret_sum, strategy_sum, visit_count


def load_warm_start_state(path: str, seed: int, stacks: tuple[int, int, int]) -> SolverState:
    solver = CFRPlusSolver(seed=seed, stacks=stacks)
    solver.warm_start_from_policy(path)
    return solver_to_state(solver)


def run_iterations(solver: CFRPlusSolver, iterations: int) -> None:
    for _ in range(iterations):
        for hero_role in (0, 1, 2):
            game = solver.new_game()
            solver.traverse(game, hero_role=hero_role, reach_probability=1.0)


def vector_delta(current: VectorTable, base: VectorTable) -> VectorTable:
    delta: VectorTable = {}
    for key, vector in current.items():
        base_vector = base.get(key)
        if base_vector is None:
            if any(value != 0.0 for value in vector):
                delta[key] = vector.copy()
            continue

        diff = [vector[i] - base_vector[i] for i in range(N_ACTIONS)]
        if any(value != 0.0 for value in diff):
            delta[key] = diff
    return delta


def visit_delta(current: VisitTable, base: VisitTable) -> VisitTable:
    delta: VisitTable = {}
    for key, visits in current.items():
        diff = visits - base.get(key, 0)
        if diff:
            delta[key] = diff
    return delta


def worker_run(
    seed: int,
    stacks: tuple[int, int, int],
    iterations: int,
    base_state: SolverState | None,
) -> WorkerResult:
    cfr_solver.DEBUG_CFR = False
    if base_state is None:
        base_state = empty_state()

    start = time.perf_counter()
    solver = state_to_solver(base_state, seed=seed, stacks=stacks)
    run_iterations(solver, iterations)
    elapsed = time.perf_counter() - start

    current_state = solver_to_state(solver)
    regret_sum, strategy_sum, visit_count = current_state
    base_regret, base_strategy, base_visits = base_state
    return WorkerResult(
        regret_delta=vector_delta(regret_sum, base_regret),
        strategy_delta=vector_delta(strategy_sum, base_strategy),
        visit_delta=visit_delta(visit_count, base_visits),
        elapsed_seconds=elapsed,
        infosets=len(strategy_sum),
        visits=sum(visit_count.values()),
    )


def merge_vector_table(target: VectorTable, delta: VectorTable) -> None:
    for key, vector in delta.items():
        target_vector = target.get(key)
        if target_vector is None:
            target[key] = vector.copy()
        else:
            for index in range(N_ACTIONS):
                target_vector[index] += vector[index]


def merge_visit_table(target: VisitTable, delta: VisitTable) -> None:
    for key, visits in delta.items():
        target[key] = target.get(key, 0) + visits


def merge_result(state: SolverState, result: WorkerResult) -> None:
    regret_sum, strategy_sum, visit_count = state
    merge_vector_table(regret_sum, result.regret_delta)
    merge_vector_table(strategy_sum, result.strategy_delta)
    merge_visit_table(visit_count, result.visit_delta)


def clone_state(state: SolverState) -> SolverState:
    regret_sum, strategy_sum, visit_count = state
    return (
        {key: value.copy() for key, value in regret_sum.items()},
        {key: value.copy() for key, value in strategy_sum.items()},
        dict(visit_count),
    )


def resolve_rounds(args: argparse.Namespace) -> None:
    if args.rounds is not None:
        return

    iterations_per_round = args.workers * args.iterations_per_worker
    args.rounds = max(1, (args.total_iterations + iterations_per_round - 1) // iterations_per_round)


def run_sequential(args: argparse.Namespace, initial_state: SolverState) -> tuple[SolverState, list[WorkerResult], float, float]:
    total_iterations = args.workers * args.rounds * args.iterations_per_worker
    start = time.perf_counter()
    base_state = clone_state(initial_state) if not is_empty_state(initial_state) else None
    result = worker_run(args.seed, args.stacks, total_iterations, base_state)
    state = clone_state(initial_state)
    merge_start = time.perf_counter()
    merge_result(state, result)
    merge_elapsed = time.perf_counter() - merge_start
    return state, [result], time.perf_counter() - start, merge_elapsed


def run_parallel(args: argparse.Namespace, initial_state: SolverState) -> tuple[SolverState, list[WorkerResult], float, float]:
    state = clone_state(initial_state)
    independent_base = clone_state(initial_state) if not is_empty_state(initial_state) else None
    all_results: list[WorkerResult] = []
    total_merge_elapsed = 0.0
    start = time.perf_counter()
    total_iterations = args.workers * args.rounds * args.iterations_per_worker

    progress = tqdm(
        total=total_iterations,
        desc=f"CFR+ {args.mode}",
        unit="iter",
        disable=args.no_progress,
    )
    with progress, ProcessPoolExecutor(max_workers=args.workers) as pool:
        for round_index in range(args.rounds):
            base_state: SolverState | None
            if args.mode == "sync":
                base_state = clone_state(state)
            else:
                base_state = independent_base

            futures = [
                pool.submit(
                    worker_run,
                    args.seed + round_index * args.workers + worker_index,
                    args.stacks,
                    args.iterations_per_worker,
                    base_state,
                )
                for worker_index in range(args.workers)
            ]

            round_results = []
            for future in as_completed(futures):
                result = future.result()
                round_results.append(result)
                progress.update(args.iterations_per_worker)
                progress.set_postfix(
                    infosets=result.infosets,
                    merge=f"{total_merge_elapsed:.2f}s",
                    refresh=False,
                )

            merge_start = time.perf_counter()
            for result in round_results:
                merge_result(state, result)
            total_merge_elapsed += time.perf_counter() - merge_start
            all_results.extend(round_results)

    return state, all_results, time.perf_counter() - start, total_merge_elapsed


def state_to_save_solver(state: SolverState, seed: int, stacks: tuple[int, int, int]) -> CFRPlusSolver:
    solver = state_to_solver(state, seed=seed, stacks=stacks)
    # CFRPlusSolver.extract_average_policy iterates strategy_sum. These defaultdicts
    # keep compatibility with the solver methods after loading plain dict state.
    solver.regret_sum = defaultdict(lambda: [0.0] * N_ACTIONS, solver.regret_sum)
    solver.strategy_sum = defaultdict(lambda: [0.0] * N_ACTIONS, solver.strategy_sum)
    solver.visit_count = defaultdict(int, solver.visit_count)
    return solver


def print_metrics(args: argparse.Namespace, state: SolverState, results: list[WorkerResult], elapsed: float, merge_elapsed: float) -> None:
    total_iterations = args.workers * args.rounds * args.iterations_per_worker
    traversals = total_iterations * 3
    worker_cpu_seconds = sum(result.elapsed_seconds for result in results)
    max_worker_seconds = max((result.elapsed_seconds for result in results), default=0.0)
    regret_sum, strategy_sum, visit_count = state

    print(f"mode={args.mode}")
    print(f"workers={args.workers}")
    print(f"rounds={args.rounds}")
    print(f"iterations_per_worker={args.iterations_per_worker}")
    print(f"requested_total_iterations={args.total_iterations}")
    print(f"warm_start={args.warm_start or ''}")
    print(f"total_iterations={total_iterations}")
    print(f"traversals={traversals}")
    print(f"elapsed_seconds={elapsed:.6f}")
    print(f"traversals_per_second={traversals / elapsed:.2f}")
    print(f"worker_cpu_seconds={worker_cpu_seconds:.6f}")
    print(f"max_worker_seconds={max_worker_seconds:.6f}")
    print(f"merge_seconds={merge_elapsed:.6f}")
    print(f"merge_pct_wall={100.0 * merge_elapsed / elapsed:.2f}")
    print(f"strategy_infosets={len(strategy_sum)}")
    print(f"regret_infosets={len(regret_sum)}")
    print(f"visits={sum(visit_count.values())}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark and run parallel CFR training.")
    parser.add_argument("--mode", choices=("sequential", "independent", "sync"), default="sync")
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--rounds", type=int, default=None, help="sync/independent batches; auto-derived from --total-iterations when omitted")
    parser.add_argument("--iterations-per-worker", type=int, default=DEFAULT_ITERATIONS_PER_WORKER)
    parser.add_argument("--total-iterations", type=int, default=DEFAULT_TOTAL_ITERATIONS)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--stacks", type=parse_stacks, default=(100, 100, 100))
    parser.add_argument("--warm-start", default="")
    parser.add_argument("--save-policy", default="")
    parser.add_argument("--save-ui-copy", default="")
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    if args.iterations_per_worker < 1:
        raise SystemExit("--iterations-per-worker must be >= 1")
    if args.total_iterations < 1:
        raise SystemExit("--total-iterations must be >= 1")

    resolve_rounds(args)
    if args.rounds < 1:
        raise SystemExit("--rounds must be >= 1")

    cfr_solver.DEBUG_CFR = False
    initial_state = empty_state()
    if args.warm_start:
        if not os.path.exists(args.warm_start):
            raise SystemExit(f"--warm-start not found: {args.warm_start}")
        initial_state = load_warm_start_state(args.warm_start, seed=args.seed, stacks=args.stacks)

    if args.mode == "sequential":
        state, results, elapsed, merge_elapsed = run_sequential(args, initial_state)
    else:
        state, results, elapsed, merge_elapsed = run_parallel(args, initial_state)

    print_metrics(args, state, results, elapsed, merge_elapsed)

    if args.save_policy or args.save_ui_copy:
        solver = state_to_save_solver(state, seed=args.seed, stacks=args.stacks)
        if args.save_policy:
            solver.save_policy_json(args.save_policy)
        if args.save_ui_copy:
            solver.save_policy_json(args.save_ui_copy)


if __name__ == "__main__":
    main()
