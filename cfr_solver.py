"""Sequential reference and reusable recursive external-sampling traversal."""
from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from typing import Callable
from actions import ACTION_IDS, legal_actions
from infoset import Observation, observe
from poker_game_expresso import HandState
from tournament import TournamentState

GameState = HandState | TournamentState
Strategy = Callable[[Observation], tuple[float, ...]]
SampleSink = Callable[[Observation, tuple[float, ...], float], None]


class TraversalBudgetExceeded(RuntimeError):
    """Budget exhaustion is an explicit failure, never a fabricated leaf value."""


def regret_matching(regrets: tuple[float, ...] | list[float], mask: tuple[bool, ...]) -> tuple[float, ...]:
    if len(regrets) != len(ACTION_IDS) or len(mask) != len(ACTION_IDS) or not any(mask):
        raise ValueError(f"Invalid regret/mask dimensions: {len(regrets)}, {len(mask)}")
    if any(not math.isfinite(r) for r in regrets):
        raise ValueError(f"Nonfinite model regrets: {regrets}")
    positive = [max(r, 0) if legal else 0 for r, legal in zip(regrets, mask)]
    total = sum(positive)
    return tuple(v / total for v in positive) if total > 0 else tuple(float(m) / sum(mask) for m in mask)


def validate_strategy(strategy: tuple[float, ...], mask: tuple[bool, ...]) -> None:
    if (len(strategy) != len(mask) or any(not math.isfinite(p) or p < 0 or (p > 0 and not m)
                                         for p, m in zip(strategy, mask))
            or not math.isclose(sum(strategy), 1.0, abs_tol=1e-7)):
        raise ValueError(f"Invalid action distribution {strategy}; legal_mask={mask}")


def sample_index(strategy: tuple[float, ...], rng: random.Random) -> int:
    threshold = rng.random()
    cumulative = 0.0
    for i, p in enumerate(strategy):
        cumulative += p
        if p > 0 and threshold < cumulative:
            return i
    raise ValueError(f"Sampling failed: distribution sum={sum(strategy)} threshold={threshold}")


@dataclass
class Traversal:
    strategy: Strategy
    rng: random.Random
    max_nodes: int = 10000
    max_depth: int = 300
    nodes: int = field(default=0, init=False)

    def _visit(self, state: GameState, depth: int) -> bool:
        self.nodes += 1
        if self.nodes > self.max_nodes or depth > self.max_depth:
            raise TraversalBudgetExceeded(f"Traversal budget exceeded: nodes={self.nodes}/{self.max_nodes}, depth={depth}/{self.max_depth}")
        return state.terminal

    def _decision(self, state: GameState) -> GameState:
        if isinstance(state, TournamentState) and (state.hand is None or state.hand.terminal):
            state = state.clone()
            state.start_hand()  # Chance sampled from injected, branch-local RNG.
        return state

    def _children(self, state: GameState):
        hand = state if isinstance(state, HandState) else state.hand
        return legal_actions(hand)

    @staticmethod
    def child(state: GameState, action) -> GameState:
        child = state.clone()
        action.apply(child if isinstance(child, HandState) else child.hand)
        return child

    def regrets(self, state: GameState, traverser: int, sink: SampleSink, depth: int = 0) -> float:
        if self._visit(state, depth):
            return state.utility(traverser)
        state = self._decision(state)
        # Blind posting can finish a hand with short stacks before any decision.
        if isinstance(state, TournamentState) and state.hand.terminal:
            return self.regrets(state, traverser, sink, depth + 1)
        obs = observe(state)
        strategy = self.strategy(obs)
        validate_strategy(strategy, obs.legal_mask)
        actions = self._children(state)
        if state.current_player != traverser:
            chosen = ACTION_IDS[sample_index(strategy, self.rng)]
            action = next(a for a in actions if a.action_id == chosen)
            return self.regrets(self.child(state, action), traverser, sink, depth + 1)
        values = [0.0] * len(ACTION_IDS)
        for action in actions:
            index = ACTION_IDS.index(action.action_id)
            values[index] = self.regrets(self.child(state, action), traverser, sink, depth + 1)
        value = sum(p * v for p, v in zip(strategy, values))
        target = tuple(v - value if m else 0.0 for v, m in zip(values, obs.legal_mask))
        sink(obs, target, 1.0)
        return value

    def average(self, state: GameState, player: int, sink: SampleSink, own_reach: float = 1.0,
                sample_reach: float = 1.0, depth: int = 0) -> None:
        if self._visit(state, depth):
            return
        state = self._decision(state)
        if isinstance(state, TournamentState) and state.hand.terminal:
            self.average(state, player, sink, own_reach, sample_reach, depth + 1)
            return
        obs = observe(state)
        strategy = self.strategy(obs)
        validate_strategy(strategy, obs.legal_mask)
        actions = self._children(state)
        if state.current_player == player:
            weight = own_reach / sample_reach
            if not math.isfinite(weight):
                raise ValueError(f"Nonfinite average importance weight: {weight}")
            if weight > 0:
                sink(obs, strategy, weight)
            for action in actions:
                p = strategy[ACTION_IDS.index(action.action_id)]
                if own_reach * p > 0:
                    self.average(self.child(state, action), player, sink, own_reach * p, sample_reach, depth + 1)
        else:
            action = self.rng.choice(actions)  # Full support even when strategy probability is zero.
            self.average(self.child(state, action), player, sink, own_reach, sample_reach / len(actions), depth + 1)


class ExternalSamplingMCCFR:
    """Unclipped cumulative regrets, uniform iteration average, frozen iteration strategy."""
    def __init__(self, seed: int = 0, max_nodes: int = 10000, max_depth: int = 300):
        self.rng = random.Random(seed)
        self.max_nodes, self.max_depth = max_nodes, max_depth
        self.regret_sum: dict[str, list[float]] = {}
        self.strategy_sum: dict[str, list[float]] = {}
        self.visits: dict[str, int] = {}
        self.iteration = 0

    def current_strategy(self, obs: Observation) -> tuple[float, ...]:
        return regret_matching(self.regret_sum.get(obs.key(), [0.0] * len(ACTION_IDS)), obs.legal_mask)

    def average_strategy(self, obs: Observation) -> tuple[float, ...]:
        vector = self.strategy_sum.get(obs.key())
        if vector is None or sum(vector) <= 0:
            raise KeyError(f"Average policy unavailable for state {obs.key()}")
        result = tuple(v / sum(vector) for v in vector)
        validate_strategy(result, obs.legal_mask)
        return result

    def run_iteration(self, root_factory: Callable[[random.Random], GameState], players: tuple[int, ...]) -> dict[str, float]:
        frozen = {k: tuple(v) for k, v in self.regret_sum.items()}
        def strategy(obs):
            return regret_matching(frozen.get(obs.key(), [0.0] * len(ACTION_IDS)), obs.legal_mask)
        regrets, averages = [], []
        values = []
        nodes = 0
        rng_state = self.rng.getstate()
        try:
            for player in players:
                walk = Traversal(strategy, self.rng, self.max_nodes, self.max_depth)
                values.append(walk.regrets(root_factory(self.rng), player, lambda o, t, w: regrets.append((o, t, w))))
                nodes += walk.nodes
                walk = Traversal(strategy, self.rng, self.max_nodes, self.max_depth)
                walk.average(root_factory(self.rng), player, lambda o, t, w: averages.append((o, t, w)))
                nodes += walk.nodes
        except Exception:
            self.rng.setstate(rng_state)
            raise
        for obs, target, weight in regrets:
            key = obs.key()
            vector = self.regret_sum.setdefault(key, [0.0] * len(ACTION_IDS))
            for i, v in enumerate(target):
                vector[i] += weight * v
            self.visits[key] = self.visits.get(key, 0) + 1
        for obs, target, weight in averages:
            vector = self.strategy_sum.setdefault(obs.key(), [0.0] * len(ACTION_IDS))
            for i, v in enumerate(target):
                vector[i] += weight * v
        self.iteration += 1
        return {"nodes": float(nodes), "mean_value": sum(values) / len(values),
                "mean_positive_regret": sum(max(v, 0) for row in self.regret_sum.values() for v in row)
                / max(1, len(self.regret_sum) * self.iteration)}
