from __future__ import annotations

import argparse
import cProfile
import io
import os
import pstats
import sys
import time

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import cfr_solver
from cfr_solver import CFRPlusSolver


def run_benchmark(iterations: int, seed: int, stacks: tuple[int, int, int]) -> CFRPlusSolver:
    solver = CFRPlusSolver(seed=seed, stacks=stacks)
    for _ in range(iterations):
        for hero_role in (0, 1, 2):
            game = solver.new_game()
            solver.traverse(game, hero_role=hero_role, reach_probability=1.0)
    return solver


def parse_stacks(value: str) -> tuple[int, int, int]:
    parts = value.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected three comma-separated stacks, e.g. 100,100,100")
    try:
        return tuple(int(part) for part in parts)  # type: ignore[return-value]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("stacks must be integers") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark CFR traversals without saving policy files.")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--stacks", type=parse_stacks, default=(100, 100, 100))
    parser.add_argument("--profile", action="store_true", help="print cProfile stats")
    parser.add_argument("--profile-limit", type=int, default=30)
    args = parser.parse_args()

    cfr_solver.DEBUG_CFR = False
    traversals = args.iterations * 3

    profiler = cProfile.Profile() if args.profile else None
    start = time.perf_counter()
    if profiler is not None:
        profiler.enable()
    solver = run_benchmark(args.iterations, args.seed, args.stacks)
    if profiler is not None:
        profiler.disable()
    elapsed = time.perf_counter() - start

    print(f"iterations={args.iterations}")
    print(f"traversals={traversals}")
    print(f"elapsed_seconds={elapsed:.6f}")
    print(f"traversals_per_second={traversals / elapsed:.2f}")
    print(f"infosets={len(solver.strategy_sum)}")

    if profiler is not None:
        stream = io.StringIO()
        pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumtime").print_stats(args.profile_limit)
        print(stream.getvalue())


if __name__ == "__main__":
    main()
