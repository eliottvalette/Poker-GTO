"""Independent-world sampled-CFR diagnostics and posterior-model calibration."""

import json
from pathlib import Path
import time

from actions import SolverAction, legal_actions
from hybrid.behavior import likelihood
from hybrid.experiments import small_game
from hybrid.learning import Learner, LearnedBehavior
from hybrid.policy_source import UniformLegalPolicy
from hybrid.quality import scenarios
from hybrid.reference import ProfilePolicy
from hybrid.river_solver import RiverSolver
from hybrid.state import ComputeBudget


def forest_generalization(path: Path, *, seeds: int = 4) -> dict:
    rows = []
    for count in (2, 3):
        root, ranges = small_game(count, seed=87, street="TURN")
        reference = RiverSolver().solve_chance(
            root,
            ranges,
            ComputeBudget(
                iterations=150, samples=1000, max_depth=64, max_nodes=3000000, seed=917
            ),
        )
        for samples in (16, 64, 256):
            for seed in range(seeds):
                started = time.perf_counter()
                result = RiverSolver().solve_chance(
                    root,
                    ranges,
                    ComputeBudget(
                        iterations=50,
                        samples=samples,
                        max_depth=64,
                        max_nodes=1000000,
                        seed=seed,
                    ),
                )
                errors = []
                losses = []
                for hand, ev in reference.action_ev_by_hand.items():
                    errors.extend(
                        (result.action_ev_by_hand[hand][a] - v) ** 2
                        for a, v in ev.items()
                    )
                    selected = sum(
                        p * ev[a] for a, p in result.strategy_by_hand[hand].items()
                    )
                    losses.append(max(ev.values()) - selected)
                rows.append(
                    {
                        "count": count,
                        "seed": seed,
                        "samples": samples,
                        "nodes": result.work.nodes,
                        "mse_vs_exhaustive_chance": sum(errors) / len(errors),
                        "root_response_loss_under_reference_continuation": sum(losses)
                        / len(losses),
                        "strategies": [
                            {"cards": h, "probabilities": p}
                            for h, p in result.strategy_by_hand.items()
                        ],
                        "seconds": time.perf_counter() - started,
                    }
                )
    report = {
        "budgets": {
            "seeds": seeds,
            "sampled_iterations": 50,
            "reference_iterations": 150,
            "reference_samples": 1000,
        },
        "scope": "Independent exhaustive future boards; finite-CFR error remains. Root actions are evaluated under the reference continuation, not the fitted forest.",
        "rows": rows,
    }
    path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def posterior_calibration(path: Path) -> dict:
    rows = []
    models = {
        (n, p): LearnedBehavior(
            Learner.load(Path(f"artifacts/phase2/behavior/behavior-{n}-{p}-w32.pt")),
            experimental=True,
        )
        for n in (2, 3)
        for p in ("tight_passive", "loose_aggressive")
    }
    for scenario in scenarios():
        state, belief = scenario.state, scenario.beliefs
        actor = state.current_player
        for name in ("tight_passive", "loose_aggressive"):
            teacher = ProfilePolicy(name)
            actions = list(legal_actions(state))[:3]
            raises = sorted(
                a.amount_to for a in legal_actions(state) if a.category == "RAISE"
            )
            if len(raises) > 1:
                actions.append(
                    SolverAction("offtree", "RAISE", (raises[0] + raises[1]) / 2)
                )
            for action in actions:
                expected = belief.update(
                    actor,
                    lambda h: likelihood(
                        teacher, state, h, action, interpolate=True
                    ).probability,
                    model_version=teacher.version,
                ).public.marginals()[actor]
                for model_name, model in (
                    ("uniform", UniformLegalPolicy()),
                    ("learned", models[(len(state.players), name)]),
                ):
                    predicted = belief.update(
                        actor,
                        lambda h: likelihood(
                            model, state, h, action, interpolate=True
                        ).probability,
                        model_version=model.version,
                    ).public.marginals()[actor]
                    tv = (
                        sum(
                            abs(w - predicted.probability(h))
                            for h, w in expected.weights.items()
                        )
                        / 2
                    )
                    rows.append(
                        {
                            "scenario": scenario.name,
                            "profile": name,
                            "action": action.action_id,
                            "candidate": model_name,
                            "posterior_total_variation": tv,
                            "status": "diagnostic includes explicitly experimental rejected models",
                        }
                    )
    report = {
        "rows": rows,
        "scope": "Independent root posterior reconstruction, including bracketed off-tree sizes; no human calibration claim",
    }
    path.write_text(json.dumps(report, indent=2) + "\n")
    return report
