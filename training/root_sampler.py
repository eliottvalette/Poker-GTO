"""Persistent tournament rollouts plus stratified simplex exploration, never CFR descendants."""
from __future__ import annotations
from dataclasses import replace
from collections import Counter
import random
import math
from typing import Callable
from actions import ACTION_IDS, apply_action
from blind_schedule import DEFAULT_SIMULATION_SCHEDULE
from cfr_solver import regret_matching, sample_index, validate_strategy
from infoset import observe
from features.neural import neural_observation
from poker_game_expresso import HandState
from tournament import TournamentState
from training.metrics import Coverage

ROOT_SAMPLER_VERSION = 1
SOURCES = ("on_policy", "synthetic", "stratified")


def canonical_seats(hand: HandState) -> HandState:
    result = hand.clone()
    mapping = {seat: i for i, seat in enumerate(hand.players)}
    result.players = {mapping[i]: replace(p, player_id=mapping[i]) for i, p in result.players.items()}
    result.initial_stacks = {mapping[i]: s for i, s in result.initial_stacks.items()}
    result.button = mapping[result.button]
    result.current_player = None if result.current_player is None else mapping[result.current_player]
    result.pending = {mapping[i] for i in result.pending}
    result.awards = {mapping[i]: s for i, s in result.awards.items()}
    result.history = [replace(e, player_id=mapping[e.player_id]) for e in result.history]
    result.assert_invariants()
    return result


class RootSampler:
    def __init__(self, player_count: int, seed: int, config: dict) -> None:
        if player_count not in (2, 3) or set(config) != {"mixture", "max_rollout_decisions", "max_rollout_hands", "tournament_start_players"}:
            raise ValueError(f"Invalid root configuration: players={player_count}, config={config}")
        mixture = config["mixture"]
        if set(mixture) != set(SOURCES) or any(type(w) not in (int, float) or not math.isfinite(w) or w < 0 for w in mixture.values()) or abs(sum(mixture.values()) - 1) > 1e-9:
            raise ValueError(f"Root mixture must sum to 1 over {SOURCES}: {mixture}")
        if type(config["max_rollout_decisions"]) is not int or config["max_rollout_decisions"] < 1:
            raise ValueError(f"Invalid tournament rollout budget: {config}")
        if (config["tournament_start_players"] not in (2, 3) or config["tournament_start_players"] < player_count
                or type(config["max_rollout_hands"]) is not int or config["max_rollout_hands"] < 1):
            raise ValueError(f"Invalid tournament root search configuration: {config}")
        self.player_count, self.config = player_count, config
        self.rng = random.Random(seed)
        self.tournament: TournamentState | None = None
        self.stratum_index = 0
        self.coverage = Coverage()
        self.policy: Callable | None = None

    def _new_tournament(self) -> TournamentState:
        count = self.config["tournament_start_players"]
        return TournamentState({i: 75 / count for i in range(count)},
                               rng=random.Random(self.rng.randrange(2**31)))

    def _on_policy(self) -> HandState:
        if self.tournament is None:
            self.tournament = self._new_tournament()
        for _ in range(self.config["max_rollout_hands"]):
            t = self.tournament
            if t.hand is not None and not t.hand.terminal:
                raise RuntimeError("Root sampler retained an unsettled rollout hand")
            if t.terminal or (self.player_count == 3 and len(t.active) != 3):
                t = self._new_tournament()
                self.tournament = t
            root = t.start_hand().clone()
            decisions = 0
            while not t.hand.terminal:
                decisions += 1
                if decisions > self.config["max_rollout_decisions"]:
                    raise RuntimeError(f"Root rollout decision budget exceeded: {decisions}")
                obs = observe(canonical_seats(t.hand))
                strategy = self.policy(obs) if self.policy is not None else regret_matching([0.0] * len(ACTION_IDS), obs.legal_mask)
                validate_strategy(strategy, obs.legal_mask)
                apply_action(t.hand, ACTION_IDS[sample_index(strategy, self.rng)])
            t.completed.clear()
            if len(root.players) == self.player_count:
                return canonical_seats(root)
        raise RuntimeError(f"Tournament root search exceeded {self.config['max_rollout_hands']} hands for {self.player_count} players")

    def _exploration(self, stratified: bool) -> HandState:
        if stratified:
            # Cartesian coverage: all levels x balanced/asymmetric x all buttons.
            index = self.stratum_index
            self.stratum_index += 1
            level = index % 6
            shape = (index // 6) % 2
            button = (index // 12) % self.player_count
        else:
            level = self.rng.randrange(6)
            shape = self.rng.randrange(5)
            button = self.rng.randrange(self.player_count)
        stage = DEFAULT_SIMULATION_SCHEDULE.stages[level]
        if shape == 0:
            weights = [1 + self.rng.uniform(-0.15, 0.15) for _ in range(self.player_count)]
        elif shape == 1:
            weights = [0.02, 1.0] if self.player_count == 2 else [0.02, 0.2, 1.0]
        elif shape == 2:
            weights = [0.001, 1.0] if self.player_count == 2 else [0.001, 0.001, 1.0]
        elif shape == 3:
            weights = [self.rng.gammavariate(0.5, 1) + 0.001 for _ in range(self.player_count)]
        else:
            weights = [0.15, 1.0] if self.player_count == 2 else [0.15, 1.0, 1.0]
        self.rng.shuffle(weights)
        total = sum(weights)
        stacks = {i: 75 * w / total for i, w in enumerate(weights)}
        stacks[self.player_count - 1] = 75 - sum(stacks[i] for i in range(self.player_count - 1))
        return HandState.start(stacks, button, self.rng, stage.blinds,
                               hand_number=stage.first_hand, blind_level_index=level)

    def sample(self, _solver_rng: random.Random | None = None) -> HandState:
        source = self.rng.choices(SOURCES, weights=[self.config["mixture"][s] for s in SOURCES], k=1)[0]
        root = self._on_policy() if source == "on_policy" else self._exploration(source == "stratified")
        if root.terminal:
            # Terminal roots are legitimate draws, retained in counts and not resampled.
            self.coverage.counts.setdefault("terminal_roots", Counter())[source] += 1
            for name, value in (("source", source), ("players", self.player_count),
                                ("blind_level", root.blind_level_index), ("street", "terminal")):
                self.coverage.counts.setdefault(name, Counter())[str(value)] += 1
        else:
            self.coverage.record(neural_observation(observe(root)), source)
        return root
