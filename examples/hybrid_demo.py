"""Small reproducible real-engine demonstrations; import without side effects."""
from hybrid.beliefs import BeliefState
from hybrid.decision_engine import HybridDecisionEngine
from hybrid.experiments import small_game
from hybrid.state import ComputeBudget


def demonstrate() -> dict:
    results = {}
    for count in (2, 3):
        root, ranges = small_game(count)
        result = HybridDecisionEngine().analyze(root, BeliefState(ranges),
                                                ComputeBudget(iterations=100, max_nodes=100000))
        results[f"river_{count}"] = result.to_dict()
    root, ranges = small_game(2, street="TURN")
    results["turn_hu"] = HybridDecisionEngine(search_mode="public_cfr").analyze(
        root, BeliefState(ranges), ComputeBudget(iterations=30, samples=176, max_depth=64,
                                                max_nodes=200000)).to_dict()
    return results
