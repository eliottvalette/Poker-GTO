from __future__ import annotations

import argparse
import gzip
import json
import math
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from cfr_solver import ACTION_INDEX, ACTIONS, CFRPlusSolver
from infoset import build_infoset_key_fast


NAIVE_ABSTRACT_INFOS_TURN_RIVER = 3 * 169 * 24 * 7 * 7
NAIVE_ABSTRACT_INFOS_POSTFLOP = 3 * 3 * 169 * 18 * 24 * 7 * 7 * 12
NAIVE_ABSTRACT_INFOS = NAIVE_ABSTRACT_INFOS_TURN_RIVER + NAIVE_ABSTRACT_INFOS_POSTFLOP


@dataclass(frozen=True)
class RichnessEstimate:
    observed: int
    f1: int
    f2: int
    chao1: float
    chao1_low: float
    chao1_high: float
    distinct_coverage_pct: float
    sample_mass_coverage_pct: float


@dataclass(frozen=True)
class ProbeResult:
    hands: int
    decisions: int
    observed: int
    already_in_policy: int
    new_infosets: int
    hit_rate_pct: float
    richness: RichnessEstimate


def load_policy(path: Path) -> tuple[dict[int, list[float]], Counter[int]]:
    with gzip.open(path, "rt", encoding="utf-8") as file:
        raw = json.load(file)

    policy: dict[int, list[float]] = {}
    visits: Counter[int] = Counter()
    for key_string, entry in raw.items():
        key = int(key_string)
        encoded_policy = entry["policy"]
        bitmask = encoded_policy[0]
        quantized_values = encoded_policy[1:]

        probabilities = [0.0] * len(ACTIONS)
        quantized_index = 0
        total = sum(quantized_values)
        for action_index in range(len(ACTIONS)):
            if (bitmask >> action_index) & 1:
                probabilities[action_index] = quantized_values[quantized_index] / total
                quantized_index += 1

        policy[key] = probabilities
        visits[key] = int(entry["visits"])

    return policy, visits


def chao1_from_counts(counts: Iterable[int]) -> RichnessEstimate:
    values = list(counts)
    frequency_of_frequencies = Counter(values)
    observed = len(values)
    f1 = frequency_of_frequencies[1]
    f2 = frequency_of_frequencies[2]
    sample_size = sum(values)

    if f2 > 0:
        chao1 = observed + (f1 * f1) / (2.0 * f2)
        ratio = f1 / f2
        variance = f2 * (0.5 * ratio**2 + ratio**3 + 0.25 * ratio**4)
    else:
        chao1 = observed + f1 * (f1 - 1) / 2.0
        variance = f1 * (2 * f1 - 1) ** 2 / 4.0

    standard_error = math.sqrt(max(0.0, variance))
    low = max(float(observed), chao1 - 1.96 * standard_error)
    high = chao1 + 1.96 * standard_error
    distinct_coverage = 100.0 * observed / chao1 if chao1 > 0 else 100.0
    mass_coverage = 100.0 * (1.0 - f1 / sample_size) if sample_size > 0 else 100.0

    return RichnessEstimate(
        observed=observed,
        f1=f1,
        f2=f2,
        chao1=chao1,
        chao1_low=low,
        chao1_high=high,
        distinct_coverage_pct=distinct_coverage,
        sample_mass_coverage_pct=mass_coverage,
    )


def normalize_legal_probabilities(
    probabilities: list[float],
    legal_actions: tuple[str, ...],
    epsilon: float,
) -> list[float]:
    result = [0.0] * len(ACTIONS)
    legal_indices = [ACTION_INDEX[action] for action in legal_actions]
    legal_total = sum(probabilities[index] for index in legal_indices)

    if legal_total <= 0.0:
        uniform = 1.0 / len(legal_indices)
        for index in legal_indices:
            result[index] = uniform
        return result

    uniform = 1.0 / len(legal_indices)
    for index in legal_indices:
        policy_probability = probabilities[index] / legal_total
        result[index] = (1.0 - epsilon) * policy_probability + epsilon * uniform
    return result


def sample_action(probabilities: list[float], rng: random.Random) -> str:
    threshold = rng.random()
    cumulative = 0.0
    last_non_zero = ACTIONS[-1]
    for index, probability in enumerate(probabilities):
        if probability > 0.0:
            last_non_zero = ACTIONS[index]
        cumulative += probability
        if threshold <= cumulative:
            return ACTIONS[index]
    return last_non_zero


def probe_policy(
    *,
    policy: dict[int, list[float]],
    hands: int,
    seed: int,
    stacks: tuple[int, int, int],
    epsilon: float,
) -> ProbeResult:
    solver = CFRPlusSolver(seed=seed, stacks=stacks)
    rng = random.Random(seed + 10_000_019)
    policy_keys = set(policy)
    counts: Counter[int] = Counter()
    decisions = 0
    already_in_policy = 0

    for _ in range(hands):
        game = solver.new_game()
        while game.current_phase != "SHOWDOWN":
            player = game.players[game.current_role]
            key = build_infoset_key_fast(game, player)
            legal_actions = solver.legal_actions(game)

            counts[key] += 1
            decisions += 1
            if key in policy_keys:
                already_in_policy += 1
            probabilities = normalize_legal_probabilities(
                policy.get(key, [0.0] * len(ACTIONS)),
                legal_actions,
                epsilon=epsilon,
            )
            game.process_action(player, sample_action(probabilities, rng))

    observed = len(counts)
    new_infosets = len(set(counts) - policy_keys)
    hit_rate = 100.0 * already_in_policy / decisions if decisions else 100.0
    return ProbeResult(
        hands=hands,
        decisions=decisions,
        observed=observed,
        already_in_policy=already_in_policy,
        new_infosets=new_infosets,
        hit_rate_pct=hit_rate,
        richness=chao1_from_counts(counts.values()),
    )


def print_richness(label: str, estimate: RichnessEstimate) -> None:
    print(f"{label}_observed={estimate.observed}")
    print(f"{label}_f1={estimate.f1}")
    print(f"{label}_f2={estimate.f2}")
    print(f"{label}_chao1={estimate.chao1:.0f}")
    print(f"{label}_chao1_low={estimate.chao1_low:.0f}")
    print(f"{label}_chao1_high={estimate.chao1_high:.0f}")
    print(f"{label}_distinct_coverage_pct={estimate.distinct_coverage_pct:.2f}")
    print(f"{label}_sample_mass_coverage_pct={estimate.sample_mass_coverage_pct:.2f}")


def parse_stacks(value: str) -> tuple[int, int, int]:
    parts = value.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected stacks like 100,100,100")
    try:
        return tuple(int(part) for part in parts)  # type: ignore[return-value]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("stacks must be integers") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Estimate abstract infoset coverage.")
    parser.add_argument("--policy", type=Path, default=Path("policy/avg_policy.json.gz"))
    parser.add_argument("--hands", type=int, default=0, help="optional Monte Carlo probe hands")
    parser.add_argument("--seed", type=int, default=1_778_870_975)
    parser.add_argument("--stacks", type=parse_stacks, default=(100, 100, 100))
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.05,
        help="uniform exploration mixed into saved policy during probes",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    policy, visits = load_policy(args.policy)
    saved_estimate = chao1_from_counts(visits.values())

    print(f"policy_infosets={len(policy)}")
    print(f"naive_abstract_infosets={NAIVE_ABSTRACT_INFOS}")
    print(f"naive_abstract_coverage_pct={100.0 * len(policy) / NAIVE_ABSTRACT_INFOS:.4f}")
    print_richness("saved_policy", saved_estimate)

    if args.hands <= 0:
        return

    probe = probe_policy(
        policy=policy,
        hands=args.hands,
        seed=args.seed,
        stacks=args.stacks,
        epsilon=args.epsilon,
    )
    print(f"probe_hands={probe.hands}")
    print(f"probe_decisions={probe.decisions}")
    print(f"probe_observed={probe.observed}")
    print(f"probe_new_infosets={probe.new_infosets}")
    print(f"probe_decision_hit_rate_pct={probe.hit_rate_pct:.2f}")
    print_richness("probe", probe.richness)


if __name__ == "__main__":
    main()
