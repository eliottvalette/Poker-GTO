"""Sequential reference and reusable recursive external-sampling traversal."""
from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from typing import Callable, TYPE_CHECKING
from actions import ACTION_IDS, legal_actions
from infoset import Observation, observe
from poker_game_expresso import HandState
from tournament import TournamentState

if TYPE_CHECKING:
    from features.neural import NeuralObservation

GameState = HandState | TournamentState
Strategy = Callable[[Observation], tuple[float, ...]]
SampleSink = Callable[[Observation, tuple[float, ...], float], None]


def hand_root(state: GameState) -> HandState:
    """Extract an isolated current hand; tournament continuation is never a leaf target."""
    if isinstance(state, TournamentState):
        if state.hand is None:
            raise ValueError("Hand traversal requires an existing tournament hand; call start_hand explicitly")
        return state.hand.clone()
    if not isinstance(state, HandState):
        raise TypeError(f"Expected HandState or TournamentState, received {type(state).__name__}")
    return state.clone()


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


def settled_value(state: HandState, player: int) -> float | None:
    if not isinstance(state, HandState):
        raise TypeError("Learning payoff requires an extracted HandState; tournament winner utility is unsupported")
    if player not in state.players:
        raise ValueError(f"Traversal player {player} not seated: {tuple(state.players)}")
    if state.terminal:
        return state.utility(player)
    participant = state.players[player]
    if participant.folded:
        return participant.stack - state.initial_stacks[player]
    return None


@dataclass
class Traversal:
    strategy: Strategy
    rng: random.Random
    max_nodes: int = 10000
    max_depth: int = 300
    observer: Callable[[HandState], Observation | NeuralObservation] = field(default=observe, repr=False)
    nodes: int = field(default=0, init=False)
    settled_prunes: int = field(default=0, init=False)
    max_depth_seen: int = field(default=0, init=False)

    def _visit(self, state: HandState, depth: int) -> bool:
        self.nodes += 1
        self.max_depth_seen = max(self.max_depth_seen, depth)
        if self.nodes > self.max_nodes or depth > self.max_depth:
            raise TraversalBudgetExceeded(
                f"Traversal budget exceeded: nodes={self.nodes}/{self.max_nodes}, "
                f"depth={depth}/{self.max_depth}, hand={state.hand_number}, "
                f"street={state.street}, "
                f"actor={state.current_player}, settled_prunes={self.settled_prunes}"
            )
        return state.terminal

    def _children(self, state: HandState):
        return legal_actions(state)

    @staticmethod
    def child(state: GameState, action) -> HandState:
        child = hand_root(state)
        action.apply(child)
        return child

    def regrets(self, state: GameState, traverser: int, sink: SampleSink, depth: int = 0) -> float:
        if depth == 0:
            state = hand_root(state)
        elif not isinstance(state, HandState):
            raise TypeError("Recursive traversal must remain inside its extracted hand")
        terminal = self._visit(state, depth)
        settled = settled_value(state, traverser)
        if settled is not None:
            self.settled_prunes += int(not terminal)
            return settled
        obs = self.observer(state)
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
        if depth == 0:
            state = hand_root(state)
        elif not isinstance(state, HandState):
            raise TypeError("Recursive traversal must remain inside its extracted hand")
        terminal = self._visit(state, depth)
        if settled_value(state, player) is not None:
            self.settled_prunes += int(not terminal)
            return
        obs = self.observer(state)
        strategy = self.strategy(obs)
        validate_strategy(strategy, obs.legal_mask)
        actions = self._children(state)
        if state.current_player == player:
            weight = own_reach / sample_reach
            if not math.isfinite(weight):
                raise ValueError(f"Nonfinite average importance weight: {weight}")
            if weight > 0:
                sink(obs, strategy, weight)
        # Sample every player's action with full support. The prefix probability
        # corrects both own and opponent sampling; chance reach cancels in the
        # normalized average at a perfect-recall infoset. This preserves the
        # original reach-weighted average without enumerating its own-action tree.
        action = self.rng.choice(actions)
        if state.current_player == player:
            own_reach *= strategy[ACTION_IDS.index(action.action_id)]
        if own_reach > 0:
            self.average(self.child(state, action), player, sink, own_reach,
                         sample_reach / len(actions), depth + 1)



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
        max_depth_seen = 0
        rng_state = self.rng.getstate()
        try:
            for player in players:
                walk = Traversal(strategy, self.rng, self.max_nodes, self.max_depth)
                values.append(walk.regrets(root_factory(self.rng), player, lambda o, t, w: regrets.append((o, t, w))))
                nodes += walk.nodes
                max_depth_seen = max(max_depth_seen, walk.max_depth_seen)
                walk = Traversal(strategy, self.rng, self.max_nodes, self.max_depth)
                walk.average(root_factory(self.rng), player, lambda o, t, w: averages.append((o, t, w)))
                nodes += walk.nodes
                max_depth_seen = max(max_depth_seen, walk.max_depth_seen)
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
                "max_depth_seen": float(max_depth_seen),
                "mean_positive_regret": sum(max(v, 0) for row in self.regret_sum.values() for v in row)
                / max(1, len(self.regret_sum) * self.iteration)}
