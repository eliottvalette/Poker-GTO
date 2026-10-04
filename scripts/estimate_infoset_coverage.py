"""Coverage is measured on supplied observations, never inferred from bucket counts."""
from infoset import Observation


def policy_coverage(policy, observations: list[Observation]) -> dict[str, float]:
    if not observations:
        raise ValueError("Coverage needs at least one observation")
    covered = 0
    for observation in observations:
        try:
            policy.query(observation)
            covered += 1
        except KeyError:
            pass  # Only explicit unsupported-state errors count as missing coverage.
    return {"observations": float(len(observations)), "covered": float(covered),
            "coverage_fraction": covered / len(observations)}
