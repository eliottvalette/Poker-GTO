"""Context-specific policy inspection; no aggregation over fabricated bucket states."""
from actions import ACTION_IDS
from infoset import Observation


def action_distribution(policy, observation: Observation) -> dict[str, float]:
    return {a: p for a, p in zip(ACTION_IDS, policy.query(observation)) if p > 0}
