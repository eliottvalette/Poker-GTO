"""Analysis transport keeps probabilities, utilities and uncertainty distinct."""
from dataclasses import asdict, dataclass, field, fields
from typing import Mapping
import time
from actions import SolverAction
from hybrid.ranges import Combo
from hybrid.policy_source import PolicySource
from infoset import Observation


@dataclass
class Analysis:
    legal_actions: tuple[SolverAction, ...]
    action_probabilities: dict[str, float]
    estimated_ev_by_action: dict[str, float]
    ev_units: str
    opponent_ranges: dict[int, Mapping[Combo, float]]
    own_public_range: Mapping[Combo, float]
    calculation_method_by_action: dict[str, str]
    search_nodes: int
    samples_used: int
    iterations_completed: int
    uncertainty_estimates: dict
    model_versions_used: dict[int | str, str]
    assumptions: list[str]
    warnings: list[str]
    baseline_probabilities: dict[str, float]
    current_big_blind: float
    strategy_by_hand: dict[Combo, dict[str, float]] | None = None
    computation_breakdown: dict[str, int] = field(default_factory=dict)

    def action_ev_in_current_bb(self) -> dict[str, float]:
        return {a: value / self.current_big_blind for a, value in self.estimated_ev_by_action.items()}

    def to_dict(self) -> dict:
        """JSON-compatible transport without losing the complete range distributions."""
        result = {f.name: getattr(self, f.name) for f in fields(self)}
        result["legal_actions"] = [asdict(a) for a in self.legal_actions]
        def rows(weights):
            return [{"cards": list(hand), "probability": p} for hand, p in weights.items()]
        result["opponent_ranges"] = {str(p): rows(r) for p, r in self.opponent_ranges.items()}
        result["own_public_range"] = rows(self.own_public_range)
        if self.strategy_by_hand is not None:
            result["strategy_by_hand"] = [{"cards": list(h), "probabilities": p} for h, p in self.strategy_by_hand.items()]
        return result


@dataclass
class MeasuredPolicy:
    """Opt-in profiling wrapper; preserves the complete policy output unchanged."""

    policy: PolicySource
    calls: int = 0
    seconds: float = 0.

    @property
    def version(self) -> str:
        return self.policy.version

    def probabilities(self, observation: Observation) -> tuple[float, ...]:
        started = time.perf_counter()
        result = self.policy.probabilities(observation)
        self.seconds += time.perf_counter() - started
        self.calls += 1
        return result
