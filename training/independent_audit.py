"""Fresh HU deals and forced-branch interventions, independent of CFR collectors."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Protocol

import numpy as np
import torch

from actions import ACTION_IDS, apply_action, legal_actions
from cfr_solver import regret_matching, sample_index, validate_strategy
from hybrid.policy_source import UniformLegalPolicy
from hybrid.reference import ProfilePolicy
from hybrid.state import instantiate
from infoset import Observation, observe
from ml.model import AdvantageNetwork, AveragePolicyNetwork, encode_batch
from poker_game_expresso import ActionEvent, BlindLevel, HandPlayer, HandState
from training.metrics import decision_context
from training.poker_evaluation import statistics

SUITE = 'hu1600-independent-interventions-v1'
PROFILES = ('uniform', 'loose_passive', 'tight_aggressive', 'push_fold')
STACKS = ((6., 6.), (25., 25.), (35., 8.))
HOLDINGS = {'AA': (48, 49), 'KK': (44, 45), 'QQ': (40, 41), 'AKs': (48, 44),
            'JTs': (36, 32), 'A5s': (48, 12), '76s': (20, 16), '72o': (20, 1)}
PASS_PREFLOP = ('CALL', 'CHECK')
FACING_RAISE = ('CHECK', 'BET_RAISE_75P', 'BET_RAISE_100P')
BRANCHES = (
    ('opening', 25., ()),
    ('facing_open', 25., ('RAISE_2.5X',)),
    ('facing_3bet', 37.5, ('RAISE_2.5X', 'RAISE_3.0X')),
    ('facing_4bet', 60., ('RAISE_2.5X', 'RAISE_3.0X', 'RAISE_2.0X')),
    ('facing_jam', 8., ('ALL_IN',)),
    ('flop_raise', 37.5, PASS_PREFLOP + FACING_RAISE),
    ('turn_raise', 37.5, PASS_PREFLOP + ('CHECK', 'CHECK') + FACING_RAISE),
    ('river_raise', 37.5, PASS_PREFLOP + ('CHECK', 'CHECK') * 2 + FACING_RAISE),
)


class Policy(Protocol):
    def probabilities(self, observation: Observation) -> tuple[float, ...]: ...


class NetworkPolicy:
    def __init__(self, model: AdvantageNetwork | AveragePolicyNetwork) -> None:
        self.model = model.eval()

    def probabilities(self, observation: Observation) -> tuple[float, ...]:
        with torch.no_grad():
            row = self.model(encode_batch([observation]))[0].tolist()
        if isinstance(self.model, AdvantageNetwork):
            return regret_matching(row, observation.legal_mask)
        total = sum(row)
        if abs(total - 1.) > 1e-6:
            raise ValueError('Invalid probability mass from B')
        result = tuple(p / total for p in row)
        validate_strategy(result, observation.legal_mask)
        return result


def seed_for(*parts: object) -> int:
    text = json.dumps([SUITE, *parts], separators=(',', ':'))
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], 'big')


def deal_identity(state: HandState) -> str:
    """Physical holdings and complete runout; ignore seat swaps and flop order."""
    board = state.board + list(reversed(state.deck))[:5 - len(state.board)]
    if len(board) != 5:
        raise ValueError('A complete five-card runout is required')
    value = {'hands': sorted(sorted(p.cards) for p in state.players.values()),
             'flop': sorted(board[:3]), 'turn_river': board[3:]}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def hourly_exclusions() -> tuple[set[str], set[tuple[int, ...]]]:
    """Enumerate default hourly HU deals, including its unshifted river worlds."""
    identities = set()
    for group in range(256):
        for profile in range(4):
            for region, stacks in enumerate(STACKS):
                root = HandState.start(dict(enumerate(stacks)), group % 2,
                                       random.Random(910000 + group * 1009 + profile * 53 + region * 7))
                identities.add(deal_identity(root))
    root = HandState.start({0: 2., 1: 2.}, 0, random.Random(73219))
    while root.street != 'RIVER' and not root.terminal:
        root.act('CALL' if root.to_call() else 'CHECK')
    if root.terminal:
        raise ValueError('Hourly river exclusion unexpectedly terminal')
    hero = root.current_player
    identities.add(deal_identity(root))
    for index in range(1, 4):
        world = root.clone()
        deck = list(world.deck) + [c for seat, p in world.players.items() if seat != hero for c in p.cards]
        random.Random(99000 + index).shuffle(deck)
        for seat, player in world.players.items():
            if seat != hero:
                player.cards = (deck.pop(), deck.pop())
        world.deck = deck
        identities.add(deal_identity(world))
    return identities, {tuple(sorted(root.board[:3]) + root.board[3:])}


def rollout(root: HandState, hero: int, candidate: Policy, opponent: Policy,
            seed: int, coverage: Counter | None = None) -> float:
    """Sample exact engine settlements; no training collector or value network."""
    state = root.clone()
    rngs = [random.Random(seed_for('action', seed, seat)) for seat in range(2)]
    for _ in range(128):
        if state.terminal:
            return state.utility(hero) / root.blinds.big
        actor = state.current_player
        obs = observe(state)
        if coverage is not None and actor == hero:
            coverage[f'{state.street}/{decision_context(obs)}'] += 1
        policy = candidate if actor == hero else opponent
        probabilities = policy.probabilities(obs)
        validate_strategy(probabilities, obs.legal_mask)
        apply_action(state, ACTION_IDS[sample_index(probabilities, rngs[actor])])
    raise RuntimeError('Independent audit exceeded 128 decisions; no partial payoff returned')


@dataclass
class AuditSuite:
    games: list[dict]
    cases: list[dict]
    worlds: list[list[HandState]]
    exclusion_count: int

    def save(self, path: Path) -> None:
        import gzip
        payload = {'suite': SUITE, 'exclusions': self.exclusion_count,
                   'hourly_contract': 'Default seed_offset=0, groups 0..255, all four river worlds',
                   'range_contract': 'Uniform compatible hidden cards under forced public-action interventions; not posterior self-play ranges',
                   'games': [{**r, 'root': pack_hand(r['root'])} for r in self.games],
                   'cases': self.cases, 'worlds': [[pack_hand(w) for w in ws] for ws in self.worlds]}
        temporary = path.with_suffix(path.suffix + '.partial')
        with gzip.open(temporary, 'wt') as stream:
            json.dump(payload, stream, allow_nan=False)
        temporary.replace(path)


def pack_hand(state: HandState) -> dict:
    result = asdict(state)
    result['pending'] = sorted(state.pending)
    return result


def unpack_hand(raw: dict) -> HandState:
    fields = dict(raw)
    fields['players'] = {int(k): HandPlayer(**{**v, 'cards': tuple(v['cards'])}) for k, v in raw['players'].items()}
    fields['initial_stacks'] = {int(k): v for k, v in raw['initial_stacks'].items()}
    fields['awards'] = {int(k): v for k, v in raw['awards'].items()}
    fields['pending'] = set(raw['pending'])
    fields['blinds'] = BlindLevel(**raw['blinds'])
    fields['history'] = [ActionEvent(**e) for e in raw['history']]
    state = HandState(**fields)
    state.assert_invariants()
    return state


def make_suite(groups: int = 128, worlds: int = 128) -> AuditSuite:
    if groups < 2 or worlds < 4 or worlds % 2:
        raise ValueError('Audit needs >=2 groups and an even number of >=4 worlds')
    excluded, river_boards = hourly_exclusions()
    used = set(excluded)
    games = []
    for group in range(groups):
        for profile in PROFILES:
            for region, stacks in enumerate(STACKS):
                root = HandState.start(dict(enumerate(stacks)), group % 2,
                                       random.Random(seed_for('game', group, profile, region)))
                identity = deal_identity(root)
                if identity in used:
                    raise ValueError('New audit game overlaps an hourly or earlier audit deal')
                used.add(identity)
                games.append({'group': group, 'profile': profile, 'region': region,
                              'root': root, 'deal_sha256': identity,
                              'action_seed': seed_for('game_actions', group, profile, region)})
    cases, populations = [], []
    for branch, stack, actions in BRANCHES:
        dummy = HandState.start({0: stack, 1: stack}, 0, random.Random(seed_for('actor', branch)))
        for action in actions:
            apply_action(dummy, action)
        hero = dummy.current_player
        for label, holding in HOLDINGS.items():
            holding = tuple(sorted(holding))
            rng = random.Random(seed_for('public', branch, label))
            remaining = [c for c in range(52) if c not in holding]
            rng.shuffle(remaining)
            other = (remaining.pop(), remaining.pop())
            hands = {hero: holding, 1 - hero: other}
            deck = remaining + [hands[1][1], hands[1][0], hands[0][1], hands[0][0]]
            root = HandState.start({0: stack, 1: stack}, 0, rng, deck=deck)
            for action in actions:
                apply_action(root, action)
            if root.terminal or root.current_player != hero:
                raise ValueError(f'Invalid forced branch: {branch}')
            if root.street == 'RIVER' and tuple(sorted(root.board[:3]) + root.board[3:]) in river_boards:
                raise ValueError('New public river board overlaps hourly fixture')
            population = []
            obs = observe(root)
            for index in range(worlds):
                # Explicit rejection sampling without replacement also excludes
                # duplicate river holdings, where the runout is already fixed.
                for attempt in range(10000):
                    rng = random.Random(seed_for('hidden', branch, label, index, attempt))
                    available = [c for c in range(52) if c not in (*holding, *root.board)]
                    rng.shuffle(available)
                    world = instantiate(root, {hero: holding, 1 - hero: (available.pop(), available.pop())}, rng)
                    identity = deal_identity(world)
                    if identity not in used:
                        break
                else:
                    raise RuntimeError(f'Unique hidden world sampling exhausted: {branch}/{label}/{index}')
                used.add(identity)
                if observe(world) != obs:
                    raise ValueError(f'Hidden world changes the evaluated information set: {branch}/{label}/{index}')
                population.append(world)
            cases.append({'id': f'{branch}/{label}', 'branch': branch, 'holding': label,
                          'hero': hero, 'stack_bb': stack, 'forced_actions': list(actions),
                          'board': root.board, 'legal_mask': list(obs.legal_mask),
                          'world_sha256': [deal_identity(w) for w in population]})
            populations.append(population)
    return AuditSuite(games, cases, populations, len(excluded))


def action_values(suite: AuditSuite, continuations: dict[str, Policy],
                  output: Path | None = None) -> dict[str, np.ndarray]:
    """Independent fixed-continuation Q; hidden worlds are paired across actions."""
    results = {}
    for name, policy in continuations.items():
        values = np.full((len(suite.cases), len(suite.worlds[0]), len(ACTION_IDS)), np.nan)
        for case_index, (case, worlds) in enumerate(zip(suite.cases, suite.worlds)):
            for world_index, world in enumerate(worlds):
                for action in legal_actions(world):
                    child = world.clone()
                    action.apply(child)
                    values[case_index, world_index, ACTION_IDS.index(action.action_id)] = rollout(
                        child, case['hero'], policy, policy, seed_for('q', case['id'], world_index))
            print(f'Action values {name}: {case_index + 1}/{len(suite.cases)}', flush=True)
        results[name] = values
        if output is not None:
            np.savez_compressed(output / f'action-values-{name}.npz', values=values)
    return results


def evaluate_games(policy: Policy, suite: AuditSuite) -> dict:
    rows, coverage = [], Counter()
    opponents = {name: UniformLegalPolicy() if name == 'uniform' else ProfilePolicy(name) for name in PROFILES}
    for game in suite.games:
        for hero in range(2):
            value = rollout(game['root'], hero, policy, opponents[game['profile']], game['action_seed'], coverage)
            rows.append({k: game[k] for k in ('group', 'profile', 'region', 'deal_sha256')}
                        | {'hero': hero, 'bb': value})
    groups = sorted({row['group'] for row in rows})
    group_values = [float(np.mean([r['bb'] for r in rows if r['group'] == group])) for group in groups]
    return {'rows': rows, 'group_bb': group_values, 'bb_per_100': statistics([v * 100 for v in group_values]),
            'decision_visits': dict(coverage)}


def evaluate_cases(model: AdvantageNetwork | AveragePolicyNetwork, suite: AuditSuite,
                   q_values: dict[str, np.ndarray]) -> dict:
    observations = [observe(worlds[0]) for worlds in suite.worlds]
    with torch.no_grad():
        raw = model(encode_batch(observations)).double().numpy()
    probabilities = np.asarray([NetworkPolicy(model).probabilities(obs) for obs in observations])
    results = {}
    for name, values in q_values.items():
        rows = []
        for index, case in enumerate(suite.cases):
            legal = np.asarray(case['legal_mask'])
            q = values[index][:, legal]
            p = probabilities[index, legal]
            mixture = (q * p[None, :]).sum(1)
            # Select on one half and assess on the other to avoid max-of-noise bias.
            half = len(q) // 2
            best_first, best_last = q[:half].mean(0).argmax(), q[half:].mean(0).argmax()
            deviation = np.concatenate((q[:half, best_last] - mixture[:half],
                                        q[half:, best_first] - mixture[half:]))
            predicted = raw[index, legal]
            chosen = int(predicted.argmax())
            reliable, correct = 0, 0
            for a in range(q.shape[1]):
                for b in range(a):
                    difference = q[:, a] - q[:, b]
                    mean, se = float(difference.mean()), float(difference.std(ddof=1) / np.sqrt(len(q)))
                    if abs(mean) > 1.96 * se:
                        reliable += 1
                        correct += int(np.sign(predicted[a] - predicted[b]) == np.sign(mean))
            rows.append({'id': case['id'], 'policy_ev': statistics(mixture.tolist()),
                         'crossfit_deviation_gain': statistics(deviation.tolist()),
                         'predicted_best_action_ev': statistics(q[:, chosen].tolist()),
                         'reliable_action_pairs': reliable, 'correctly_ranked_pairs': correct,
                         'raw_outputs': raw[index].tolist(), 'probabilities': probabilities[index].tolist(),
                         'action_means': q.mean(0).tolist(),
                         'legal_actions': [a for a, yes in zip(ACTION_IDS, legal) if yes],
                         'uniform_ev': float(q.mean()), 'world_policy_ev': mixture.tolist()})
        results[name] = {'cases': rows,
                         'mean_policy_ev': float(np.mean([r['policy_ev']['mean'] for r in rows])),
                         'mean_crossfit_deviation_gain': float(np.mean([r['crossfit_deviation_gain']['mean'] for r in rows]))}
    return results
