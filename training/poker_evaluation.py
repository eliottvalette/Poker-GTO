"""Versioned paired policy evaluation; diagnostics are not full-game exploitability."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random
import time
from typing import Callable

from actions import ACTION_IDS, legal_actions
from cfr_solver import sample_index, validate_strategy
from evaluation import subgame_best_response
from hybrid.policy_source import UniformLegalPolicy
from hybrid.reference import ProfilePolicy
from infoset import observe
from poker_game_expresso import HandState

SUITE = 'hourly-paired-policy-v1'
PROFILES = ('uniform', 'loose_passive', 'tight_aggressive', 'push_fold')
REGIONS = ('shallow', 'medium', 'asymmetric')


@dataclass(frozen=True)
class EvaluationBudget:
    groups: int = 128
    cpu_seconds: float = 120
    wall_seconds: float = 480
    decisions_per_hand: int = 128
    river_nodes: int = 20000

    def __post_init__(self) -> None:
        if (type(self.groups) is not int or not 1 <= self.groups <= 256
                or not math.isfinite(self.cpu_seconds) or self.cpu_seconds <= 0
                or not math.isfinite(self.wall_seconds) or self.wall_seconds <= 0
                or self.decisions_per_hand < 1 or self.river_nodes < 1):
            raise ValueError('Invalid evaluation budget')


class EvaluationLimit(RuntimeError):
    pass


def statistics(values: list[float]) -> dict:
    """Descriptive normal intervals over independent deal groups, never promotion tests."""
    if any(not math.isfinite(v) for v in values):
        raise ValueError('Nonfinite evaluation outcome')
    n = len(values)
    mean = math.fsum(values) / n if n else None
    m2 = math.fsum((v - mean) ** 2 for v in values) if n else 0.0
    se = math.sqrt(m2 / (n - 1) / n) if n > 1 else None
    return {'n': n, 'mean': mean, 'm2': m2, 'se': se,
            'interval': [mean - 1.96 * se, mean + 1.96 * se] if se is not None else None}


def play(root: HandState, hero: int, policy, opponent, seed: int,
         budget: EvaluationBudget, check: Callable[[], None]) -> tuple[float, int]:
    state = root.clone()
    # Separate streams prevent Hero's action draws from shifting opponents' draws.
    rngs = {seat: random.Random(seed + seat * 104729) for seat in state.players}
    decisions = 0
    while not state.terminal:
        check()
        if decisions >= budget.decisions_per_hand:
            raise RuntimeError('Evaluation hand exceeded its decision limit')
        actor = state.current_player
        observation = observe(state)
        distribution = (policy if actor == hero else opponent).probabilities(observation)
        validate_strategy(distribution, observation.legal_mask)
        selected = ACTION_IDS[sample_index(distribution, rngs[actor])]
        next(a for a in legal_actions(state) if a.action_id == selected).apply(state)
        decisions += 1
    # Physical chip settlement is converted using this hand's actual big blind.
    return state.utility(hero) / root.blinds.big, decisions


def evaluate_policy(candidate, reference, count: int,
                    budget: EvaluationBudget = EvaluationBudget()) -> tuple[dict, list[dict]]:
    if count not in (2, 3):
        raise ValueError('Evaluation supports HU and 3-max only')
    cpu, wall = time.process_time(), time.monotonic()

    def check() -> None:
        if time.process_time() - cpu >= budget.cpu_seconds or time.monotonic() - wall >= budget.wall_seconds:
            raise EvaluationLimit('Evaluation CPU or wall budget exhausted')

    rows: list[dict] = []
    groups: list[tuple[float, float]] = []
    hands = decisions = attempted = discarded = 0
    status = 'complete'
    river = None
    warnings = ['Fixed synthetic opponent pool; no claim of general poker strength.',
                'Intervals are descriptive normal approximations; repeated hourly tests are not independent.',
                'Only exported policy is evaluated, not online hybrid search or tournament win probability.']
    try:
        for group in range(budget.groups):
            pending = []
            for profile_index, profile in enumerate(PROFILES):
                opponent = UniformLegalPolicy() if profile == 'uniform' else ProfilePolicy(profile)
                for region_index, region in enumerate(REGIONS):
                    seed = 910000 + group * 1009 + profile_index * 53 + region_index * 7
                    stacks = [6.] * count if region == 'shallow' else [25.] * count if region == 'medium' else [35., 8., 18.][:count]
                    root = HandState.start(dict(enumerate(stacks)), group % count, random.Random(seed))
                    scores = [[], []]
                    for hero in range(count):
                        for index, policy in enumerate((candidate, reference)):
                            attempted += 1
                            value, work = play(root, hero, policy, opponent, seed + 700003, budget, check)
                            scores[index].append(value)
                            hands += 1
                            decisions += work
                    pending.append({'group': group, 'profile': profile, 'region': region,
                                    'candidate': math.fsum(scores[0]) / count,
                                    'reference': math.fsum(scores[1]) / count})
            # Commit only a complete balanced group; count discarded work explicitly.
            rows.extend(pending)
            groups.append(tuple(math.fsum(r[key] for r in pending) / len(pending)
                                for key in ('candidate', 'reference')))
        # Exact finite river tree; varies both opponents' cards in 3-max.
        root = HandState.start({p: 2. for p in range(count)}, 0, random.Random(73219))
        while root.street != 'RIVER' and not root.terminal:
            root.act('CALL' if root.to_call() else 'CHECK')
        if root.terminal:
            raise ValueError('River reference unexpectedly terminal')
        hero = root.current_player
        deals = [(0.25, root.clone())]
        for index in range(1, 4):
            world = root.clone()
            deck = list(world.deck) + [card for seat, player in world.players.items()
                                       if seat != hero for card in player.cards]
            random.Random(99000 + index).shuffle(deck)
            for seat, player in world.players.items():
                if seat != hero:
                    player.cards = (deck.pop(), deck.pop())
            world.deck = deck
            world.assert_invariants()
            deals.append((0.25, world))

        def policy(observation):
            check()
            return candidate.probabilities(observation)

        river = subgame_best_response(deals, policy, hero, max_nodes=budget.river_nodes)
    except EvaluationLimit as error:
        status = 'budget_limited'
        warnings.append(str(error) + '; incomplete balanced groups discarded. Path-cost stopping may bias estimates.')
    except Exception as error:
        # No missing or failed hand is silently treated as a win, loss, or successful evaluation.
        status = 'failed'
        warnings.append(f'{type(error).__name__}: {error}')
    retained_hands = len(rows) * count * 2
    discarded = hands - retained_hands
    segments = []
    for profile in PROFILES:
        for region in REGIONS:
            selected = [r for r in rows if r['profile'] == profile and r['region'] == region]
            segments.append({'profile': profile, 'region': region,
                             'gain': statistics([r['candidate'] * 100 for r in selected]),
                             'difference': statistics([(r['candidate'] - r['reference']) * 100 for r in selected])})
    report = {'suite': SUITE, 'status': status, 'budget': asdict(budget),
              'completed_groups': len(groups), 'hands': hands, 'attempted_hands': attempted,
              'discarded_hands': discarded, 'decisions': decisions,
              'cpu_seconds': time.process_time() - cpu, 'wall_seconds': time.monotonic() - wall,
              'gain': statistics([a * 100 for a, _ in groups]),
              'difference': statistics([(a - b) * 100 for a, b in groups]),
              'segments': segments, 'river': river, 'warnings': warnings,
              'units': 'BB/100', 'reference_scope': 'fixed initial exported policy',
              'river_scope': 'four-deal conditional river, one responding seat; gain in hand BB, not full-game exploitability'}
    return report, rows
