"""Single-trajectory MCCFR; estimator and horizon limits: docs/outcome-sampling.md."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import random

from actions import ACTION_IDS, legal_actions
from cfr_solver import GameState, SampleSink, Strategy, TraversalBudgetExceeded, hand_root, sample_index, settled_value, validate_strategy
from infoset import Observation, observe


LearningRecord = tuple[Observation, tuple[float, ...], float]


class ImportanceNumericalError(ArithmeticError):
    """An estimator cannot be represented; no clipping or fabricated zero is used."""


def behavior_policy(strategy: tuple[float, ...], mask: tuple[bool, ...], epsilon: float) -> tuple[float, ...]:
    if not math.isfinite(epsilon) or not 0 < epsilon <= 1:
        raise ValueError(f"Outcome exploration epsilon must be in (0, 1], got {epsilon}")
    validate_strategy(strategy, mask)
    count = sum(mask)
    probabilities = tuple((1 - epsilon) * p + epsilon / count if legal else 0.0
                          for p, legal in zip(strategy, mask))
    if any(legal and p <= 0 for p, legal in zip(probabilities, mask)):
        raise ImportanceNumericalError(f"Behavior policy lost full support: epsilon={epsilon}, q={probabilities}")
    validate_strategy(probabilities, mask)
    return probabilities


def _log_probability(probability: float) -> float:
    return math.log(probability) if probability > 0 else -math.inf


def _finite_exp(log_value: float, label: str) -> float:
    if log_value == -math.inf:
        return 0.0  # A structural zero, not a small positive weight.
    if not math.isfinite(log_value):
        raise ImportanceNumericalError(f"Invalid {label} log value: {log_value}")
    try:
        value = math.exp(log_value)
    except OverflowError as error:
        raise ImportanceNumericalError(f"Float64 overflow in {label}: log_value={log_value}") from error
    if value == 0 or not math.isfinite(value):
        raise ImportanceNumericalError(f"Float64 under/overflow in {label}: log_value={log_value}, value={value}")
    return value


@dataclass
class OutcomeDiagnostics:
    nodes: int = 0
    decisions: int = 0
    hand_number: int = 0
    terminal_reason: str = "running"
    failure: str | None = None
    log_inverse_sample_reach: list[float] = field(default_factory=list)
    log_regret_prefix_weights: list[float] = field(default_factory=list)
    log_average_weights: list[float] = field(default_factory=list)
    log_regret_total_weights: list[float] = field(default_factory=list)
    log_continuation_ratios: list[float] = field(default_factory=list)
    max_abs_regret: float = 0.0
    max_depth_seen: int = 0


@dataclass(frozen=True)
class _Decision:
    observation: Observation | None
    strategy: tuple[float, ...]
    sampled: int
    log_q: float
    log_own: float
    log_opponents: float
    log_sample: float


@dataclass
class OutcomeSamplingTraversal:
    strategy: Strategy
    rng: random.Random
    max_nodes: int = 10000
    max_depth: int = 300
    epsilon: float = 0.6
    nodes: int = field(default=0, init=False)
    max_depth_seen: int = field(default=0, init=False)
    diagnostics: OutcomeDiagnostics = field(default_factory=OutcomeDiagnostics, init=False)

    def __post_init__(self) -> None:
        if type(self.max_nodes) is not int or self.max_nodes < 1 or type(self.max_depth) is not int or self.max_depth < 0:
            raise ValueError(f"Invalid outcome budgets: nodes={self.max_nodes}, depth={self.max_depth}")
        if not math.isfinite(self.epsilon) or not 0 < self.epsilon <= 1:
            raise ValueError(f"Outcome exploration epsilon must be in (0, 1], got {self.epsilon}")

    def run(self, root: GameState, player: int, regret_sink: SampleSink, average_sink: SampleSink) -> float:
        """Publish only after reaching an exact payoff and validating all estimates.

        Budgets/numerical errors retain diagnostics, restore traversal RNG, and
        publish no samples. Training must abort, not retry until a short path wins.
        """
        self.nodes = 0
        self.max_depth_seen = 0
        self.diagnostics = OutcomeDiagnostics()
        rng_state = self.rng.getstate()
        try:
            value, regrets, averages = self._estimate(root, player)
        except Exception as error:
            self.rng.setstate(rng_state)
            self.diagnostics.failure = str(error)
            self.diagnostics.terminal_reason = "budget_exceeded" if isinstance(error, TraversalBudgetExceeded) else "error"
            raise
        for observation, target, weight in regrets:
            regret_sink(observation, target, weight)
        for observation, target, weight in averages:
            average_sink(observation, target, weight)
        return value

    def _estimate(self, root: GameState, player: int) -> tuple[float, list[LearningRecord], list[LearningRecord]]:
        state = hand_root(root)
        records: list[_Decision] = []
        log_own = log_opponents = log_sample = 0.0
        depth = 0
        while True:
            self.nodes += 1
            self.diagnostics.nodes = self.nodes
            self.diagnostics.hand_number = state.hand_number
            self.max_depth_seen = max(self.max_depth_seen, depth)
            self.diagnostics.max_depth_seen = self.max_depth_seen
            if self.nodes > self.max_nodes or depth > self.max_depth:
                raise TraversalBudgetExceeded(
                    f"Outcome traversal budget exceeded: nodes={self.nodes}/{self.max_nodes}, "
                    f"depth={depth}/{self.max_depth}, hand={self.diagnostics.hand_number}"
                )
            payoff = settled_value(state, player)
            if payoff is not None:
                self.diagnostics.terminal_reason = "terminal" if state.terminal else "settled_payoff"
                break
            obs = observe(state)
            sigma = tuple(self.strategy(obs))
            q = behavior_policy(sigma, obs.legal_mask, self.epsilon)
            sampled = sample_index(q, self.rng)
            own_node = state.current_player == player
            records.append(_Decision(obs if own_node else None, sigma, sampled, math.log(q[sampled]),
                                     log_own, log_opponents, log_sample))
            self.diagnostics.decisions += 1
            self.diagnostics.log_inverse_sample_reach.append(-log_sample)
            if own_node:
                self.diagnostics.log_regret_prefix_weights.append(log_opponents - log_sample)
                self.diagnostics.log_average_weights.append(log_own - log_sample)
                log_own += _log_probability(sigma[sampled])
            else:
                log_opponents += _log_probability(sigma[sampled])
            log_sample += math.log(q[sampled])
            next(a for a in legal_actions(state) if a.action_id == ACTION_IDS[sampled]).apply(state)
            depth += 1
        if not math.isfinite(payoff):
            raise ImportanceNumericalError(f"Nonfinite terminal payoff: {payoff}")
        sign = math.copysign(1.0, payoff)
        log_payoff = math.log(abs(payoff)) if payoff != 0 else -math.inf
        log_suffix = 0.0
        regrets: list[LearningRecord] = []
        averages: list[LearningRecord] = []
        for record in reversed(records):
            sampled_sigma = record.strategy[record.sampled]
            self.diagnostics.log_continuation_ratios.append(log_suffix)
            if record.observation is not None:
                # Combine the selected action's 1/q and all prefix/suffix ratios
                # before exponentiation; illegal actions never get regret targets.
                log_correction = log_suffix + record.log_opponents - record.log_sample - record.log_q
                self.diagnostics.log_regret_total_weights.append(log_correction)
                base = log_payoff + log_correction
                target = []
                for index, legal in enumerate(record.observation.legal_mask):
                    coefficient = float(index == record.sampled) - sampled_sigma
                    if not legal or coefficient == 0:
                        target.append(0.0)
                    else:
                        magnitude = _finite_exp(base + math.log(abs(coefficient)), "regret target")
                        target.append(sign * math.copysign(magnitude, coefficient))
                self.diagnostics.max_abs_regret = max(self.diagnostics.max_abs_regret, max(map(abs, target)))
                regrets.append((record.observation, tuple(target), 1.0))
                weight = _finite_exp(record.log_own - record.log_sample, "average weight")
                if weight > 0:
                    averages.append((record.observation, record.strategy, weight))
            log_suffix += _log_probability(sampled_sigma) - record.log_q
        value = sign * _finite_exp(log_payoff + log_suffix, "root value")
        return value, regrets, averages
