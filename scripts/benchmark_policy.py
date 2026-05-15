from __future__ import annotations

import argparse
import csv
import gzip
import itertools
import json
import math
import random
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config
from cfr_solver import ACTION_INDEX, ACTIONS
from infoset import build_infoset_key_fast
from infoset import hero_vs_board_bucket
from poker_game_expresso import GameInit
from poker_game_expresso import PokerGameExpresso

Policy = dict[int, list[float]]


@dataclass(frozen=True)
class HandResult:
    scenario: str
    table_hand_index: int
    base_deal_index: int
    seat_order: tuple[int, int, int]
    bot_names_by_role: tuple[str, str, str]
    net_chips_by_role: tuple[float, float, float]
    net_bb_by_role: tuple[float, float, float]


@dataclass(frozen=True)
class BotMetrics:
    scenario: str
    bot: str
    hands: int
    total_bb: float
    bb_per_100: float
    winrate_pct: float
    mean_bb_per_hand: float
    variance_bb_per_hand: float
    stdev_bb_per_hand: float
    ci95_bb_per_100: float


class Bot(Protocol):
    name: str

    def choose_action(
        self,
        game: PokerGameExpresso,
        legal_actions: tuple[str, ...],
        rng: random.Random,
    ) -> str:
        ...


def load_policy_from_source(source: str | Path) -> Policy:
    source_value = str(source)
    if source_value.startswith("git:"):
        _, ref, path = source_value.split(":", 2)
        data = subprocess.check_output(["git", "show", f"{ref}:{path}"], cwd=ROOT_DIR)
    else:
        data = Path(source_value).read_bytes()

    raw = json.loads(gzip.decompress(data).decode("utf-8"))
    policy: Policy = {}
    for key_string, entry in raw.items():
        encoded_policy = entry["policy"]
        bitmask = encoded_policy[0]
        quantized_values = encoded_policy[1:]
        total = sum(quantized_values)
        probabilities = [0.0] * len(ACTIONS)
        quantized_index = 0
        for action_index in range(len(ACTIONS)):
            if (bitmask >> action_index) & 1:
                probabilities[action_index] = quantized_values[quantized_index] / total
                quantized_index += 1
        policy[int(key_string)] = probabilities
    return policy


def legal_policy_probabilities(
    probabilities: list[float],
    legal_actions: tuple[str, ...],
    epsilon: float,
) -> list[float]:
    legal_indices = [ACTION_INDEX[action] for action in legal_actions]
    legal_total = sum(probabilities[index] for index in legal_indices)
    result = [0.0] * len(ACTIONS)

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
    fallback = ACTIONS[-1]
    for index, probability in enumerate(probabilities):
        if probability > 0.0:
            fallback = ACTIONS[index]
        cumulative += probability
        if threshold <= cumulative:
            return ACTIONS[index]
    return fallback


def choose_first_legal(preferences: tuple[str, ...], legal_actions: tuple[str, ...]) -> str:
    for action in preferences:
        if action in legal_actions:
            return action
    return legal_actions[-1]


def preflop_score(game: PokerGameExpresso) -> float:
    player = game.players[game.current_role]
    ranks = sorted((card.rank for card in player.cards), reverse=True)
    suited = player.cards[0].suit == player.cards[1].suit
    pair = ranks[0] == ranks[1]
    gap = ranks[0] - ranks[1]

    if pair:
        return min(1.0, 0.52 + ranks[0] / 14.0 * 0.44)

    score = (ranks[0] + ranks[1]) / 28.0
    if suited:
        score += 0.08
    if gap <= 2:
        score += 0.04
    score -= min(0.12, max(0, gap - 2) * 0.02)
    return max(0.0, min(1.0, score))


def board_aware_score(game: PokerGameExpresso) -> float:
    if not game.community_cards:
        return preflop_score(game)

    player = game.players[game.current_role]
    bucket = hero_vs_board_bucket(player, game.community_cards)
    bucket_scores = {
        0: 0.18,
        4: 0.42,
        5: 0.58,
        6: 0.72,
        7: 0.78,
        8: 0.84,
        9: 0.92,
    }
    return bucket_scores.get(bucket, 0.35)


class PolicyBot:
    def __init__(self, name: str, policy: Policy, epsilon: float = 0.0) -> None:
        self.name = name
        self.policy = policy
        self.epsilon = epsilon

    def choose_action(
        self,
        game: PokerGameExpresso,
        legal_actions: tuple[str, ...],
        rng: random.Random,
    ) -> str:
        player = game.players[game.current_role]
        key = build_infoset_key_fast(game, player)
        probabilities = legal_policy_probabilities(
            self.policy.get(key, [0.0] * len(ACTIONS)),
            legal_actions,
            self.epsilon,
        )
        return sample_action(probabilities, rng)


class RandomBot:
    name = "random"

    def choose_action(
        self,
        game: PokerGameExpresso,
        legal_actions: tuple[str, ...],
        rng: random.Random,
    ) -> str:
        return rng.choice(legal_actions)


class CallingStationBot:
    name = "calling_station"

    def choose_action(
        self,
        game: PokerGameExpresso,
        legal_actions: tuple[str, ...],
        rng: random.Random,
    ) -> str:
        if rng.random() < 0.03 and "RAISE" in legal_actions:
            return "RAISE"
        return choose_first_legal(("CHECK", "CALL", "ALL-IN", "FOLD"), legal_actions)


class NitBot:
    name = "nit"

    def choose_action(
        self,
        game: PokerGameExpresso,
        legal_actions: tuple[str, ...],
        rng: random.Random,
    ) -> str:
        strength = board_aware_score(game)
        facing_bet = "FOLD" in legal_actions
        if strength >= 0.86 and "RAISE" in legal_actions:
            return "RAISE"
        if strength >= 0.92 and "ALL-IN" in legal_actions and rng.random() < 0.2:
            return "ALL-IN"
        if facing_bet:
            if strength >= 0.58 and "CALL" in legal_actions:
                return "CALL"
            if strength >= 0.82 and "ALL-IN" in legal_actions:
                return "ALL-IN"
            return "FOLD"
        return choose_first_legal(("CHECK", "RAISE", "ALL-IN"), legal_actions)


class AggroBot:
    name = "aggro"

    def choose_action(
        self,
        game: PokerGameExpresso,
        legal_actions: tuple[str, ...],
        rng: random.Random,
    ) -> str:
        strength = board_aware_score(game)
        if "RAISE" in legal_actions and (strength >= 0.45 or rng.random() < 0.45):
            return "RAISE"
        if "ALL-IN" in legal_actions and (strength >= 0.78 or rng.random() < 0.12):
            return "ALL-IN"
        if "CALL" in legal_actions and rng.random() < 0.82:
            return "CALL"
        return choose_first_legal(("CHECK", "FOLD", "CALL", "ALL-IN"), legal_actions)


class ShoveFoldBot:
    name = "shove_fold"

    def choose_action(
        self,
        game: PokerGameExpresso,
        legal_actions: tuple[str, ...],
        rng: random.Random,
    ) -> str:
        strength = board_aware_score(game)
        threshold = 0.68 if game.current_phase == "PREFLOP" else 0.72
        if strength >= threshold and "ALL-IN" in legal_actions:
            return "ALL-IN"
        if "CHECK" in legal_actions:
            return "CHECK"
        return choose_first_legal(("FOLD", "CALL", "ALL-IN"), legal_actions)


def new_game(seed: int, stacks: tuple[int, int, int]) -> PokerGameExpresso:
    init = GameInit()
    init.stacks_init = list(stacks)
    init.total_bets_init = [0, 0, 0]
    init.current_bets_init = [0, 0, 0]
    init.active_init = [True, True, True]
    init.has_acted_init = [False, False, False]
    init.main_pot = 0
    init.phase = "PREFLOP"
    init.community_cards = []
    init.rng = random.Random(seed)

    game = PokerGameExpresso(init)
    game.deal_small_and_big_blind()
    return game


def play_hand(
    *,
    scenario: str,
    table_hand_index: int,
    base_deal_index: int,
    seed: int,
    stacks: tuple[int, int, int],
    role_bots: tuple[Bot, Bot, Bot],
    seat_order: tuple[int, int, int],
    max_decisions: int,
) -> HandResult:
    game = new_game(seed=seed, stacks=stacks)
    action_rng = random.Random(seed + 1_000_003)
    decisions = 0

    while game.current_phase != "SHOWDOWN":
        decisions += 1
        if decisions > max_decisions:
            raise RuntimeError(f"max decisions exceeded in scenario={scenario}, seed={seed}")

        player = game.players[game.current_role]
        legal_actions = game.update_available_actions(
            player,
            game.current_maximum_bet,
            game.number_raise_this_game_phase,
            game.main_pot,
            game.current_phase,
        )
        if not legal_actions:
            raise RuntimeError(f"no legal actions in scenario={scenario}, seed={seed}")

        action = role_bots[game.current_role].choose_action(game, legal_actions, action_rng)
        game.process_action(player, action)

    net_chips = tuple(float(game.net_stack_changes[f"Player_{role}"]) for role in range(3))
    net_bb = tuple(value / game.big_blind for value in net_chips)
    return HandResult(
        scenario=scenario,
        table_hand_index=table_hand_index,
        base_deal_index=base_deal_index,
        seat_order=seat_order,
        bot_names_by_role=tuple(bot.name for bot in role_bots),
        net_chips_by_role=net_chips,
        net_bb_by_role=net_bb,
    )


def summarize_results(results: list[HandResult]) -> list[BotMetrics]:
    values_by_scenario_bot: dict[tuple[str, str], list[float]] = defaultdict(list)
    for result in results:
        for role, bot_name in enumerate(result.bot_names_by_role):
            values_by_scenario_bot[(result.scenario, bot_name)].append(result.net_bb_by_role[role])

    metrics: list[BotMetrics] = []
    for (scenario, bot_name), values in sorted(values_by_scenario_bot.items()):
        hands = len(values)
        total = sum(values)
        mean = total / hands if hands else 0.0
        wins = sum(1 for value in values if value > 0.0)
        if hands > 1:
            variance = sum((value - mean) ** 2 for value in values) / (hands - 1)
        else:
            variance = 0.0
        stdev = math.sqrt(variance)
        metrics.append(
            BotMetrics(
                scenario=scenario,
                bot=bot_name,
                hands=hands,
                total_bb=total,
                bb_per_100=mean * 100.0,
                winrate_pct=100.0 * wins / hands if hands else 0.0,
                mean_bb_per_hand=mean,
                variance_bb_per_hand=variance,
                stdev_bb_per_hand=stdev,
                ci95_bb_per_100=1.96 * stdev / math.sqrt(hands) * 100.0 if hands else 0.0,
            )
        )
    return metrics


def build_scenarios(current_policy: Policy, old_policy: Policy | None, epsilon: float) -> dict[str, tuple[Bot, Bot, Bot]]:
    current = PolicyBot("policy_current", current_policy, epsilon=epsilon)
    bots: dict[str, Bot] = {
        "random": RandomBot(),
        "calling_station": CallingStationBot(),
        "nit": NitBot(),
        "aggro": AggroBot(),
        "shove_fold": ShoveFoldBot(),
    }

    scenarios: dict[str, tuple[Bot, Bot, Bot]] = {}
    for name, opponent in bots.items():
        scenarios[f"current_vs_2x_{name}"] = (current, opponent, opponent)

    if old_policy is not None:
        old = PolicyBot("policy_old", old_policy, epsilon=epsilon)
        scenarios["current_vs_old_vs_random"] = (current, old, bots["random"])
        scenarios["current_vs_old_vs_station"] = (current, old, bots["calling_station"])
        scenarios["current_vs_2x_old"] = (current, old, old)

    return scenarios


def seat_orders(rotate_seats: bool, duplicate_deals: bool) -> list[tuple[int, int, int]]:
    if not rotate_seats:
        return [(0, 1, 2)]
    if duplicate_deals:
        return list(itertools.permutations((0, 1, 2)))
    return [(0, 1, 2), (1, 2, 0), (2, 0, 1)]


def run_benchmark(
    *,
    scenarios: dict[str, tuple[Bot, Bot, Bot]],
    hands: int,
    seed: int,
    stacks: tuple[int, int, int],
    rotate_seats: bool,
    duplicate_deals: bool,
    max_decisions: int,
) -> list[HandResult]:
    results: list[HandResult] = []
    table_hand_index = 0
    orders = seat_orders(rotate_seats=rotate_seats, duplicate_deals=duplicate_deals)

    for scenario_index, (scenario_name, bots) in enumerate(scenarios.items()):
        for base_deal_index in range(hands):
            deal_seed = seed + scenario_index * 10_000_000 + base_deal_index
            for order_index, order in enumerate(orders):
                role_bots = tuple(bots[source_index] for source_index in order)
                hand_seed = deal_seed if duplicate_deals else deal_seed + order_index * 1_000_000
                results.append(
                    play_hand(
                        scenario=scenario_name,
                        table_hand_index=table_hand_index,
                        base_deal_index=base_deal_index,
                        seed=hand_seed,
                        stacks=stacks,
                        role_bots=role_bots,  # type: ignore[arg-type]
                        seat_order=order,
                        max_decisions=max_decisions,
                    )
                )
                table_hand_index += 1

    return results


def write_outputs(output_dir: Path, results: list[HandResult], metrics: list[BotMetrics], config: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(metrics[0]).keys()) if metrics else [])
        writer.writeheader()
        for metric in metrics:
            writer.writerow(asdict(metric))

    hands_path = output_dir / "hands.csv"
    with hands_path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "scenario",
            "table_hand_index",
            "base_deal_index",
            "seat_order",
            "bot_names_by_role",
            "net_chips_by_role",
            "net_bb_by_role",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "scenario": result.scenario,
                    "table_hand_index": result.table_hand_index,
                    "base_deal_index": result.base_deal_index,
                    "seat_order": json.dumps(result.seat_order),
                    "bot_names_by_role": json.dumps(result.bot_names_by_role),
                    "net_chips_by_role": json.dumps(result.net_chips_by_role),
                    "net_bb_by_role": json.dumps(result.net_bb_by_role),
                }
            )

    payload = {
        "config": config,
        "metrics": [asdict(metric) for metric in metrics],
        "hands": [asdict(result) for result in results],
    }
    (output_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def resolve_output_dir(configured_output_dir: Path | None) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if configured_output_dir is None:
        return Path("benchmarks") / timestamp
    return Path(str(configured_output_dir).format(timestamp=timestamp))


def parse_stacks(value: str) -> tuple[int, int, int]:
    parts = value.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected stacks like 100,100,100")
    try:
        return tuple(int(part) for part in parts)  # type: ignore[return-value]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("stacks must be integers") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark saved CFR policies against simple poker bots.")
    parser.add_argument("--policy", default=config.BENCHMARK_POLICY, help="current policy path")
    parser.add_argument(
        "--old-policy",
        default=config.BENCHMARK_OLD_POLICY,
        help="old policy path or git source like git:HEAD:policy/avg_policy.json.gz; use '' to disable",
    )
    parser.add_argument("--hands", type=int, default=config.BENCHMARK_HANDS, help="base deals per scenario")
    parser.add_argument("--seed", type=int, default=config.BENCHMARK_SEED)
    parser.add_argument("--stacks", type=parse_stacks, default=config.STACKS)
    parser.add_argument("--policy-epsilon", type=float, default=config.BENCHMARK_POLICY_EPSILON)
    parser.add_argument("--max-decisions", type=int, default=config.BENCHMARK_MAX_DECISIONS)
    parser.add_argument("--out-dir", type=Path, default=config.BENCHMARK_OUT_DIR)
    parser.add_argument("--enabled-scenarios", nargs="*", default=list(config.BENCHMARK_SCENARIOS))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.hands < 1:
        raise SystemExit("--hands must be >= 1")
    if args.max_decisions < 1:
        raise SystemExit("--max-decisions must be >= 1")
    if not 0.0 <= args.policy_epsilon <= 1.0:
        raise SystemExit("--policy-epsilon must be in [0, 1]")

    current_policy = load_policy_from_source(args.policy)
    old_policy = None
    if args.old_policy:
        try:
            old_policy = load_policy_from_source(args.old_policy)
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            print(f"[WARN] old policy unavailable, skipping old-policy scenarios: {exc}", file=sys.stderr)

    scenarios = build_scenarios(current_policy, old_policy, epsilon=args.policy_epsilon)
    if args.enabled_scenarios:
        requested_scenarios = set(args.enabled_scenarios)
        unknown_scenarios = requested_scenarios - set(scenarios)
        if unknown_scenarios:
            raise SystemExit(f"unknown enabled_scenarios: {sorted(unknown_scenarios)}")
        scenarios = {name: scenarios[name] for name in args.enabled_scenarios}

    output_dir = resolve_output_dir(args.out_dir)

    run_config = {
        "policy": str(args.policy),
        "old_policy": str(args.old_policy) if old_policy is not None else "",
        "hands": args.hands,
        "seed": args.seed,
        "stacks": args.stacks,
        "policy_epsilon": args.policy_epsilon,
        "max_decisions": args.max_decisions,
        "rotate_seats": True,
        "duplicate_deals": True,
        "enabled_scenarios": args.enabled_scenarios,
        "scenarios": list(scenarios),
    }

    results = run_benchmark(
        scenarios=scenarios,
        hands=args.hands,
        seed=args.seed,
        stacks=args.stacks,
        rotate_seats=True,
        duplicate_deals=True,
        max_decisions=args.max_decisions,
    )
    metrics = summarize_results(results)
    write_outputs(output_dir, results, metrics, run_config)

    print(f"output_dir={output_dir}")
    for metric in metrics:
        print(
            f"{metric.scenario:28s} {metric.bot:16s} "
            f"hands={metric.hands:6d} bb/100={metric.bb_per_100:9.2f} "
            f"ci95={metric.ci95_bb_per_100:8.2f} winrate={metric.winrate_pct:6.2f}%"
        )


if __name__ == "__main__":
    main()
