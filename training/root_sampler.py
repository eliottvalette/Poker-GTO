"""Persistent tournament rollouts plus stratified simplex exploration, never CFR descendants."""
from __future__ import annotations
from dataclasses import replace
from collections import Counter
from itertools import combinations
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

ROOT_SAMPLER_VERSION = 2
SOURCES = ("on_policy", "synthetic", "stratified")
HOLE_COMBOS = tuple(combinations(range(52), 2))


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
        if player_count not in (2, 3) or set(config) != {"mixture", "max_rollout_decisions", "max_rollout_hands", "tournament_start_players", "hole_card_sampling"}:
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
        if config["hole_card_sampling"] not in ("random", "stratified", "stratified_recorded_opening"):
            raise ValueError(f"Invalid hole-card sampling mode: {config['hole_card_sampling']}")
        self.rng = random.Random(seed)
        self.card_rng = random.Random(seed + 1000003)
        self.card_cycles: dict[str, list[int]] = {}
        self.card_cursors: dict[str, int] = {}
        self.hole_coverage: dict[str, Counter] = {}
        self.tournament: TournamentState | None = None
        self.stratum_index = 0
        self.coverage = Coverage()
        self.policy: Callable | None = None

    def begin_iteration(self) -> None:
        """Use fresh random permutations for each frozen strategy profile.

        Incomplete cycles must not carry across policy updates, which would
        couple the next profile to its remaining private-card strata.
        """
        self.card_cycles.clear()
        self.card_cursors.clear()

    def task_factory(self, players: tuple[int, ...], traversals_per_player: int) -> Callable[[random.Random], HandState]:
        """Bind the solver's player-major task order to explicit card strata."""
        if players != tuple(range(self.player_count)) or type(traversals_per_player) is not int or traversals_per_player < 1:
            raise ValueError(f"Invalid root task schedule: players={players}, traversals={traversals_per_player}")
        self.begin_iteration()
        scheduled = iter(player for player in players for _ in range(traversals_per_player))

        def sample_task(rng: random.Random) -> HandState:
            try:
                player = next(scheduled)
            except StopIteration:
                raise ValueError("Root task schedule exhausted") from None
            return self.sample(rng, traverser=player)

        return sample_task

    def stratify_cards(self, root: HandState, player: int, *, stream: str = "") -> HandState:
        """Uniform exact combos without replacement, then a conditional deal.

        Public roots are sampled first with their original RNG. The independent
        card stream cannot affect stack/blind/source selection or rollouts.
        Each full 1326-record cycle has class multiplicities 6/4/12.
        """
        if player not in root.players or root.street != "PREFLOP" or root.board or root.terminal:
            raise ValueError("Hole-card stratification requires a live preflop root and a seated player")
        key = f"{player}:{root.players[player].position}" + (f":{stream}" if stream else "")
        cursor = self.card_cursors.get(key, len(HOLE_COMBOS))
        if cursor == len(HOLE_COMBOS):
            order = list(range(len(HOLE_COMBOS)))
            self.card_rng.shuffle(order)
            self.card_cycles[key] = order
            cursor = 0
        combo = HOLE_COMBOS[self.card_cycles[key][cursor]]
        self.card_cursors[key] = cursor + 1
        result = root.clone()
        remaining = [card for card in range(52) if card not in combo]
        self.card_rng.shuffle(remaining)
        result.players[player].cards = combo if self.card_rng.random() < .5 else combo[::-1]
        for seat, other in result.players.items():
            if seat != player:
                other.cards = (remaining.pop(), remaining.pop())
        result.deck = remaining
        result.assert_invariants()
        self.hole_coverage.setdefault(key, Counter())[f"{combo[0]},{combo[1]}"] += 1
        return result

    def card_state(self) -> dict:
        return {"rng": self.card_rng.getstate(), "cycles": self.card_cycles,
                "cursors": self.card_cursors, "coverage": {k: dict(v) for k, v in self.hole_coverage.items()}}

    def restore_card_state(self, raw: dict) -> None:
        if set(raw) != {"rng", "cycles", "cursors", "coverage"} or set(raw["cycles"]) != set(raw["cursors"]):
            raise ValueError("Invalid stratified card checkpoint fields")
        for key, order in raw["cycles"].items():
            if sorted(order) != list(range(len(HOLE_COMBOS))) or type(raw["cursors"][key]) is not int or not 0 <= raw["cursors"][key] <= len(HOLE_COMBOS):
                raise ValueError(f"Invalid card cycle/cursor: {key}")
        self.card_rng.setstate(raw["rng"])
        self.card_cycles = {k: list(v) for k, v in raw["cycles"].items()}
        self.card_cursors = dict(raw["cursors"])
        self.hole_coverage = {k: Counter(v) for k, v in raw["coverage"].items()}

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

    def sample(self, _solver_rng: random.Random | None = None, *, traverser: int | None = None) -> HandState:
        if self.config["hole_card_sampling"] != "random" and traverser not in range(self.player_count):
            raise ValueError("Stratified hole-card roots require an explicit seated traverser")
        source = self.rng.choices(SOURCES, weights=[self.config["mixture"][s] for s in SOURCES], k=1)[0]
        root = self._on_policy() if source == "on_policy" else self._exploration(source == "stratified")
        if self.config["hole_card_sampling"] == "stratified" and not root.terminal:
            root = self.stratify_cards(root, traverser)
        elif self.config["hole_card_sampling"] == "stratified_recorded_opening" and not root.terminal:
            # Opening observations come from the actor, not necessarily the
            # traverser. Each dataset uses an independent uniform permutation.
            actor = root.current_player if self.player_count == 2 else traverser
            source_kind = ("advantage" if actor == traverser else "strategy") if self.player_count == 2 else "both"
            root = self.stratify_cards(root, actor, stream=source_kind)
        if root.terminal:
            # Terminal roots are legitimate draws, retained in counts and not resampled.
            self.coverage.counts.setdefault("terminal_roots", Counter())[source] += 1
            for name, value in (("source", source), ("players", self.player_count),
                                ("blind_level", root.blind_level_index), ("street", "terminal")):
                self.coverage.counts.setdefault(name, Counter())[str(value)] += 1
        else:
            self.coverage.record(neural_observation(observe(root)), source)
        return root
