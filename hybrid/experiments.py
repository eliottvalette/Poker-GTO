"""Importable bounded fixtures, profiling and acceptance evidence generation."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import random
import time

from actions import legal_actions
from evaluation import subgame_best_response
from hybrid.beliefs import BeliefState
from hybrid.decision_engine import HybridDecisionEngine
from hybrid.policy_source import UniformLegalPolicy, category_policy
from hybrid.ranges import HandRange, JointRanges
from hybrid.river_solver import RiverSolver
from hybrid.state import ComputeBudget, instantiate
from poker_game_expresso import HandState
from cfr_solver import sample_index
from actions import ACTION_IDS
from hybrid.policy_source import distribution
from infoset import observe


def small_game(count: int, seed: int = 3, street: str = "RIVER") -> tuple[HandState, JointRanges]:
    root = HandState.start({i: 1.5 for i in range(count)}, 0, random.Random(seed))
    while root.street != street:
        root.act("CALL" if root.to_call() else "CHECK")
    cards = [c for c in range(52) if c not in root.board]
    factors = {p: HandRange({tuple(cards[4*i:4*i+2]): 1, tuple(cards[4*i+2:4*i+4]): 2})
               for i, p in enumerate(root.players)}
    ranges = JointRanges(factors, tuple(root.board))
    return instantiate(root, ranges.enumerate()[0].hands), ranges


def opponent_pool() -> dict:
    return {
        "uniform": UniformLegalPolicy(),
        "passive": category_policy({"FOLD": .05, "CALL": 5, "CHECK": 5, "RAISE": .05, "ALL_IN": .05}, "passive-v1"),
        "tight": category_policy({"FOLD": 5, "CALL": .2, "CHECK": 5, "RAISE": .1, "ALL_IN": .1}, "tight-v1"),
        "aggressive": category_policy({"FOLD": .1, "CALL": .2, "CHECK": .1, "RAISE": 3, "ALL_IN": 2}, "aggressive-v1"),
        "shove_fold": category_policy({"FOLD": 1, "CALL": .01, "CHECK": .01, "RAISE": 0, "ALL_IN": 1}, "shove-fold-v1"),
    }


def write_browser_fixtures(path: Path) -> list[dict]:
    rows = []
    for count, street in ((2, "RIVER"), (3, "RIVER"), (2, "TURN")):
        root, ranges = small_game(count, street=street)
        budget = ComputeBudget(iterations=30 if street == "RIVER" else 5, max_nodes=100000,
                               samples=176, max_depth=64)
        started = time.perf_counter()
        result = HybridDecisionEngine(search_mode="public_cfr").analyze(root, BeliefState(ranges), budget)
        elapsed = time.perf_counter() - started
        state = asdict(root)
        state["pending"] = sorted(root.pending)
        rows.append({"count": count, "street": street, "searchMode": "public_cfr", "pythonSeconds": elapsed, "state": state,
                     "ranges": {p: [{"cards": h, "probability": w} for h, w in r.weights.items()]
                                for p, r in ranges.ranges.items()},
                     "budget": {"maxNodes": budget.max_nodes, "iterations": budget.iterations,
                                "samples": budget.samples, "maxDepth": budget.max_depth,
                                "maxJointCandidates": budget.max_joint_candidates, "seed": budget.seed},
                     "expected": {"probabilities": result.action_probabilities, "ev": result.estimated_ev_by_action,
                                  "nodes": result.search_nodes, "iterations": result.iterations_completed}})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2) + "\n")
    return rows


def river_budget_benchmark(path: Path) -> list[dict]:
    rows = []
    for count in (2, 3):
        root, ranges = small_game(count)
        roots = [(d.probability, instantiate(root, d.hands)) for d in ranges.enumerate()]
        previous = None
        for iterations in (1, 10, 100, 500):
            started = time.perf_counter()
            result = RiverSolver().solve(root, ranges, ComputeBudget(iterations=iterations, max_nodes=1000000))
            elapsed = time.perf_counter() - started
            responses = {p: subgame_best_response(roots, result.policy.probabilities, p) for p in root.players}
            rows.append({"player_count": count, "iterations": iterations, "nodes": result.work.nodes,
                         "seconds": elapsed, "expected_chip_utilities": result.expected_utilities,
                         "best_response": responses,
                         "root_policy_total_variation_from_previous_budget": None if previous is None else
                         sum(sum(abs(v - previous[h][a]) for a, v in row.items()) / 2
                             for h, row in result.strategy_by_hand.items()) / len(result.strategy_by_hand),
                         "root_strategy_by_hand": [{"cards": h, "probabilities": p} for h, p in result.strategy_by_hand.items()],
                         "root_action_ev_by_hand": [{"cards": h, "action_ev": p} for h, p in result.action_ev_by_hand.items()],
                         "scope": "specified range-conditioned finite river game, not whole-game exploitability"})
            previous = result.strategy_by_hand
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2) + "\n")
    return rows


def self_play_labels(count: int, hands: int, seed: int, policy, *, max_decisions: int = 128) -> tuple[list, list[dict]]:
    """Actual policy play; labels are exact teacher action likelihoods, not invented outcomes."""
    from hybrid.learning import Supervision, observation_features
    rng = random.Random(seed)
    labels, outcomes = [], []
    for hand_index in range(hands):
        state = HandState.start({p: 3.0 for p in range(count)}, hand_index % count, rng)
        steps = 0
        while not state.terminal:
            if steps >= max_decisions:
                raise RuntimeError(f"Self-play decision limit: hand={hand_index}, limit={max_decisions}")
            obs = observe(state)
            probabilities = distribution(policy, state)
            labels.append(Supervision(observation_features(obs), probabilities, obs.legal_mask, policy.version,
                                      count, {"max_decisions_per_hand": max_decisions, "street": state.street}, 0., f"{seed}:{hand_index}"))
            selected = ACTION_IDS[sample_index(probabilities, rng)]
            next(a for a in legal_actions(state) if a.action_id == selected).apply(state)
            steps += 1
        outcomes.append({"hand": hand_index, "decisions": steps,
                         "chip_utilities": {p: state.utility(p) for p in state.players}})
    return labels, outcomes


def learning_experiment(directory: Path, *, epochs: tuple[int, ...] = (50, 100, 200),
                        behavior_hands: int = 12, value_games: int = 8, value_iterations: int = 100) -> dict:
    """Small fixed-data fit sweep and exact-label continuation workflow, independently per track."""
    import math
    import torch
    from hybrid.learning import Learner, LearnedBehavior
    from hybrid.continuation import exact_river_labels
    if not epochs or tuple(sorted(set(epochs))) != epochs or epochs[0] < 1:
        raise ValueError("Explicit positive increasing fit measurement epochs required")
    if any(type(n) is not int or n < 2 for n in (behavior_hands, value_games, value_iterations)):
        raise ValueError("Learning experiment counts must be integers >=2")
    directory.mkdir(parents=True, exist_ok=True)
    report = {"scope": "bounded workflow/held-out diagnostics, not a poker-quality acceptance",
              "budgets": {"behavior_hands": behavior_hands, "value_games": value_games,
                          "value_iterations": value_iterations, "measurement_epochs": epochs}, "tracks": {}}
    for count in (2, 3):
        started = time.perf_counter()
        labels, outcomes = self_play_labels(count, behavior_hands, 901, opponent_pool()["passive"])
        # Complete source labels remain in the checkpoint, including groups and budgets.
        behavior = Learner(labels, "behavior", seed=77)
        sweep = []
        for epoch in epochs:
            sweep.append(behavior.fit_to(epoch))
        baseline = sum(math.log(sum(labels[i].mask)) for i in behavior.held_indices) / len(behavior.held_indices)
        behavior.validate(baseline, baseline * .98)
        behavior.save(directory / f"behavior-{count}.pt")
        behavior.export(directory / f"behavior-{count}.json")
        # A fresh independent self-play seed measures calibration outside the split.
        validation, _ = self_play_labels(count, 4, 1907, opponent_pool()["passive"])
        with torch.no_grad():
            logits = behavior.model(torch.tensor([r.features for r in validation], dtype=torch.float32))
            mask = torch.tensor([r.mask for r in validation], dtype=torch.bool)
            probabilities = logits.masked_fill(~mask, -torch.inf).softmax(1)
            targets = torch.tensor([r.target for r in validation])
            log_loss = float(-(targets * probabilities.clamp_min(1e-12).log()).sum(1).mean())
            brier = float((probabilities - targets).square().sum(1).mean())
        values = []
        for seed in range(value_games):
            root, ranges = small_game(count, seed=seed + 40)
            values.extend(exact_river_labels(root, ranges, ComputeBudget(iterations=value_iterations, max_nodes=1000000), str(seed)))
        value_model = Learner(values, "continuation", seed=77)
        value_sweep = [value_model.fit_to(epoch) for epoch in epochs]
        train_mean = sum(values[i].target[0] for i in value_model.train_indices) / len(value_model.train_indices)
        value_baseline = sum((values[i].target[0] - train_mean)**2 for i in value_model.held_indices) / len(value_model.held_indices)
        value_model.validate(value_baseline, .01)
        value_model.save(directory / f"continuation-{count}.pt")
        value_model.export(directory / f"continuation-{count}.json")
        root, ranges = small_game(count)
        roots = [(d.probability, instantiate(root, d.hands)) for d in ranges.enumerate()]
        assistance = []
        for prior in (None, LearnedBehavior(behavior, experimental=True)):
            tick = time.perf_counter()
            solved = RiverSolver(prior).solve(root, ranges, ComputeBudget(iterations=30))
            gains = [subgame_best_response(roots, solved.policy.probabilities, p)["best_response_gain_bb"] for p in root.players]
            assistance.append({"prior": None if prior is None else prior.version, "nodes": solved.work.nodes,
                               "seconds": time.perf_counter() - tick, "bounded_response_gain_sum": sum(gains)})
        report["tracks"][str(count)] = {"behavior_fit": sweep, "behavior_validation": behavior.validation,
                                        "independent_behavior_log_loss": log_loss, "independent_behavior_brier": brier,
                                        "continuation_fit": value_sweep, "continuation_validation": value_model.validation,
                                        "prior_comparison": assistance, "self_play": outcomes,
                                        "seconds": time.perf_counter() - started}
    (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def profile_components(path: Path, *, behavior_checkpoint: Path) -> dict:
    """Short independently timed operations; no optimization without measurement."""
    from features.equity import hand_equity
    from hybrid.learning import Learner, LearnedBehavior
    from ml.deep_cfr import DeepCFRSolver, TraversalTask, generate_samples
    root, ranges = small_game(2)
    def measure(callback, repeats):
        started = time.perf_counter()
        for _ in range(repeats):
            callback()
        seconds = time.perf_counter() - started
        return {"repeats": repeats, "total_seconds": seconds, "seconds_per_call": seconds / repeats}
    hero, other = (p.cards for p in root.players.values())
    beliefs = BeliefState(ranges)
    snapshot = DeepCFRSolver((0, 1)).snapshot()
    task = TraversalTask(0, 0, root, 101, 1000, 64)
    learned = LearnedBehavior(Learner.load(behavior_checkpoint), experimental=True)
    results = {
        "range_construction_1326": measure(HandRange.uniform, 10),
        "range_update": measure(lambda: beliefs.update(0, lambda _: .5, model_version="known"), 100),
        "exact_equity_turn": measure(lambda: hand_equity(hero, other, tuple(root.board[:4])), 10),
        "monte_carlo_equity_turn_128": measure(lambda: hand_equity(hero, other, tuple(root.board[:4]), max_exact=1, samples=128), 10),
        "hand_clone": measure(root.clone, 1000),
        "cfr_build_and_100_iterations": measure(lambda: RiverSolver().solve(root, ranges, ComputeBudget(iterations=100)), 5),
        "joint_chance_sample": measure(lambda: ranges.sample(random.Random(42)), 100),
        "neural_behavior_inference": measure(lambda: learned.probabilities(observe(root)), 20),
        "deep_cfr_replay_generation": measure(lambda: generate_samples(snapshot, task), 10),
    }
    solver = RiverSolver().solve(root, ranges, ComputeBudget(iterations=100))
    results["cfr_build_and_100_iterations"]["visited_nodes"] = solver.work.nodes
    results["limitations"] = ["Microbenchmarks use narrow synthetic ranges; broad range and long history costs differ.",
                               "Fitting and browser timings must be supplied by the matching experiment reports."]
    path.write_text(json.dumps(results, indent=2) + "\n")
    return results


def fixed_deep_cfr_fit(directory: Path, *, traversals: int = 8, epochs: tuple[int, ...] = (50, 100, 200)) -> dict:
    """New, explicitly tiny frozen replay; never presented as the deleted HU audit data."""
    import torch
    from ml.deep_cfr import DeepCFRSolver, TraversalTask, generate_samples
    from ml.memory import ReservoirMemory
    from ml.model import AdvantageNetwork, AveragePolicyNetwork
    from ml.train import fit
    from training.metrics import Coverage
    directory.mkdir(parents=True, exist_ok=True)
    report = {"scope": "new first-iteration short-stack HU replay; fitting diagnostic, not hand-strength quality", "budgets": {"traversals_per_player": traversals, "epochs": epochs}, "models": {}}
    memories = {kind: ReservoirMemory(10000, 17, kind, "hand_chip_delta") for kind in ("advantage", "strategy")}
    snapshot = DeepCFRSolver((0, 1)).snapshot()
    coverage = Coverage()
    for player in (0, 1):
        for i in range(traversals):
            root = HandState.start({0: 1.5, 1: 1.5}, i % 2, random.Random(500 + i))
            result = generate_samples(snapshot, TraversalTask(i, player, root, 900 + i, 5000, 64))
            coverage.merge(result.coverage)
            for kind, samples in (("advantage", result.advantages), ("strategy", result.strategies)):
                for sample in samples:
                    memories[kind].add(sample)
    report["coverage"] = coverage.as_dict()
    for kind, cls in (("advantage", AdvantageNetwork), ("strategy", AveragePolicyNetwork)):
        memory = memories[kind]
        memory.save(directory / f"{kind}-replay.json.gz")
        measurements = []
        started = time.perf_counter()
        with torch.random.fork_rng():
            torch.manual_seed(771)
            model = cls()
            fit(model, memory.samples, epochs[-1], 128, 77, measurement_epochs=epochs,
                on_measurement=lambda epoch, metrics: measurements.append({"epoch": epoch, **metrics}))
        report["models"][kind] = {"records": len(memory.samples), "measurements": measurements,
                                  "seconds": time.perf_counter() - started}
    (directory / "fit-report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def matched_opponent_experiment(path: Path, average_artifacts: dict[int, Path] | None = None, *, behavior_checkpoints: dict[int, Path]) -> list[dict]:
    """Exact matched deals against a heterogeneous held-out behavior pool."""
    from hybrid.learning import Learner, LearnedBehavior
    rows = []
    for count in (2, 3):
        root, ranges = small_game(count, seed=1201)
        hero = root.current_player
        roots = [(d.probability, instantiate(root, d.hands)) for d in ranges.enumerate()]
        neural = LearnedBehavior(Learner.load(behavior_checkpoints[count]), experimental=True)
        candidates = {"uniform": UniformLegalPolicy(), "neural_behavior_only": neural}
        for iterations, name, prior in ((5, "previous_local_policy", None), (50, "hybrid_no_model", None),
                                        (50, "hybrid_neural_prior", neural), (500, "tabular_reference", None)):
            candidates[name] = RiverSolver(prior).solve(root, ranges, ComputeBudget(iterations=iterations, max_nodes=1000000)).policy
        if average_artifacts is not None:
            from ml.deep_cfr import NeuralAveragePolicy as Artifact
            from hybrid.policy_source import NeuralAveragePolicy
            artifact = Artifact(average_artifacts[count])
            average = NeuralAveragePolicy(artifact.query, f"published-average/{count}/{artifact.iteration}")
            candidates[average.version] = average
            candidates["hybrid_published_average_prior"] = RiverSolver(average).solve(
                root, ranges, ComputeBudget(iterations=50, max_nodes=1000000)).policy
        for opponent_name, opponent in opponent_pool().items():
            for name, candidate in candidates.items():
                def profile(obs):
                    return candidate.probabilities(obs) if obs.hero == hero else opponent.probabilities(obs)
                started = time.perf_counter()
                evaluation = subgame_best_response(roots, profile, hero, max_nodes=100000)
                rows.append({"players": count, "hero": hero, "candidate": name, "opponent": opponent_name,
                             "evaluation_seed": 1201, "chip_ev": evaluation["policy_value_bb"],
                             "bounded_response_gain": evaluation["best_response_gain_bb"],
                             "nodes": evaluation["nodes"], "seconds": time.perf_counter() - started,
                             "scope": "exact expectation on matched finite river deals; not full-game poker strength"})
    path.write_text(json.dumps(rows, indent=2) + "\n")
    return rows


def chance_precision_experiment(path: Path) -> list[dict]:
    from features.equity import hand_equity
    from hybrid.search import search
    root, _ = small_game(2, street="TURN")
    root.board = [0, 5, 10, 31]
    holdings = {0: (48, 49), 1: (44, 45)}
    root = instantiate(root, holdings)
    ranges = JointRanges({p: HandRange({h: 1}) for p, h in holdings.items()}, tuple(root.board))
    hero = root.current_player
    other = next(p for p in root.players if p != hero)
    exact = 3 * hand_equity(holdings[hero], holdings[other], tuple(root.board)).equity - 1.5
    passive = category_policy({"CALL": 1, "CHECK": 1}, "always-call-check")
    rows = []
    for samples in (16, 64, 256, 1024):
        measurements = []
        for seed in range(8):
            start = time.perf_counter()
            result = search(root, ranges, hero, {p: passive for p in root.players},
                            ComputeBudget(samples=samples, max_depth=1, seed=seed + 1800))
            measurements.append({"seed": seed + 1800, "ev": result.action_ev["ALL_IN"],
                                 "standard_error": result.standard_errors["ALL_IN"],
                                 "seconds": time.perf_counter() - start, "nodes": result.work.nodes})
        rows.append({"samples": samples, "exact_chip_ev": exact, "independent_seeds": measurements,
                     "mean_squared_error": sum((m["ev"] - exact)**2 for m in measurements) / len(measurements),
                     "mean_standard_error": sum(m["standard_error"] for m in measurements) / len(measurements)})
    path.write_text(json.dumps(rows, indent=2) + "\n")
    return rows


def behavior_calibration_experiment(path: Path, *, behavior_checkpoints: dict[int, Path]) -> list[dict]:
    import math
    import torch
    from hybrid.learning import Learner
    rows = []
    for count in (2, 3):
        learner = Learner.load(behavior_checkpoints[count])
        for population in ("passive", "aggressive"):
            labels, _ = self_play_labels(count, 8, 7013, opponent_pool()[population])
            with torch.no_grad():
                logits = learner.model(torch.tensor([r.features for r in labels], dtype=torch.float32))
                masks = torch.tensor([r.mask for r in labels], dtype=torch.bool)
                probabilities = logits.masked_fill(~masks, -torch.inf).softmax(1).tolist()
            log_loss = -sum(sum(t * math.log(max(p, 1e-12)) for t, p in zip(r.target, prediction))
                            for r, prediction in zip(labels, probabilities)) / len(labels)
            baseline = sum(math.log(sum(r.mask)) for r in labels) / len(labels)
            bins = []
            for index in range(10):
                entries = [(p, t) for r, prediction in zip(labels, probabilities)
                           for p, t, legal in zip(prediction, r.target, r.mask)
                           if legal and min(9, int(p * 10)) == index]
                if entries:
                    bins.append({"lower": index / 10, "count": len(entries),
                                 "mean_prediction": sum(p for p, _ in entries) / len(entries),
                                 "true_frequency": sum(t for _, t in entries) / len(entries)})
            total = sum(b["count"] for b in bins)
            ece = sum(b["count"] * abs(b["mean_prediction"] - b["true_frequency"]) for b in bins) / total
            # Mixture over distinct hypothetical holdings at a fixed public root.
            # Analytic passive likelihood is hand-independent, so its posterior
            # equals the prior; learned deviations directly measure range error.
            from hybrid.learning import LearnedBehavior
            from hybrid.behavior import likelihood
            root, ranges = small_game(count, seed=501)
            actor = root.current_player
            selected = legal_actions(root)[-1]
            prior = BeliefState(ranges)
            truth = prior.update(actor, lambda h: likelihood(opponent_pool()[population], root, h, selected).probability,
                                 model_version=population)
            predicted = prior.update(actor, lambda h: likelihood(LearnedBehavior(learner, experimental=True), root, h, selected).probability,
                                     model_version="learned")
            true_marginal, predicted_marginal = truth.public.marginals()[actor], predicted.public.marginals()[actor]
            tv = sum(abs(w - predicted_marginal.probability(h)) for h, w in true_marginal.weights.items()) / 2
            rows.append({"player_count": count, "population": population, "held_out_population": population == "aggressive",
                         "labels": len(labels), "log_loss": log_loss, "uniform_log_loss": baseline,
                         "calibration_error": ece, "bins": bins, "posterior_total_variation": tv,
                         "accepted_for_population": log_loss < baseline and ece < .05,
                         "scope": "known synthetic probabilities; not calibrated real-player behavior"})
    path.write_text(json.dumps(rows, indent=2) + "\n")
    return rows


def continuation_comparison(path: Path, *, continuation_checkpoints: dict[int, Path]) -> list[dict]:
    """Independent held-out value probes; rejected models are never enabled online."""
    import torch
    from hybrid.learning import Learner
    from hybrid.continuation import continuation_features
    from hybrid.leaf_evaluation import rollout
    from hybrid.state import WorkCounter
    rows = []
    for count in (2, 3):
        model = Learner.load(continuation_checkpoints[count])
        for seed in range(90, 94):
            root, ranges = small_game(count, seed)
            tick = time.perf_counter()
            reference = RiverSolver().solve(root, ranges, ComputeBudget(iterations=500, max_nodes=1000000))
            exact_seconds = time.perf_counter() - tick
            for holding, probabilities in reference.strategy_by_hand.items():
                hero = root.current_player
                target = sum(p * reference.action_ev_by_hand[holding][a] for a, p in probabilities.items())
                tick = time.perf_counter()
                with torch.no_grad():
                    predicted = float(model.model(torch.tensor([continuation_features(root, ranges, holding)], dtype=torch.float32))[0, 0])
                inference_seconds = time.perf_counter() - tick
                rng = random.Random(8800 + seed)
                private = ranges.conditioned(hero, holding)
                work = WorkCounter(ComputeBudget(max_nodes=100000))
                values = []
                tick = time.perf_counter()
                # Roll out the independently solved reference policy so both
                # estimators target the SAME continuation assumption.
                for _ in range(64):
                    state = instantiate(root, private.sample(rng), rng)
                    values.append(rollout(state, hero, {p: reference.policy for p in root.players}, rng, work))
                rollout_seconds = time.perf_counter() - tick
                mean = sum(values) / len(values)
                rows.append({"players": count, "seed": seed, "holding": holding,
                             "exact_profile_value": target, "reference_iterations": 500,
                             "learned_prediction_diagnostic_only": predicted, "learned_squared_error": (predicted-target)**2,
                             "rollout_mean": mean, "rollout_squared_error": (mean-target)**2,
                             "exact_solve_seconds": exact_seconds, "inference_seconds": inference_seconds,
                             "rollout_seconds": rollout_seconds, "rollout_samples": 64,
                             "online_learned_enabled": model.accepted,
                             "gate": model.validation,
                             "scope": "independent held-out seeds; reference policy's exact conditional terminal expectation"})
    path.write_text(json.dumps(rows, indent=2) + "\n")
    return rows


def budget_latency_experiment(path: Path, *, behavior_checkpoints: dict[int, Path]) -> dict:
    from hybrid.diagnostics import MeasuredPolicy
    from hybrid.learning import Learner, LearnedBehavior
    import statistics
    rows = []
    for count in (2, 3):
        neural = LearnedBehavior(Learner.load(behavior_checkpoints[count]), experimental=True)
        for iterations in (10, 100, 500):
            for seed in range(5):
                root, ranges = small_game(count, seed + 2100)
                for assisted in (False, True):
                    prior = MeasuredPolicy(neural) if assisted else None
                    tick = time.perf_counter()
                    result = RiverSolver(prior).solve(root, ranges, ComputeBudget(iterations=iterations, max_nodes=1000000))
                    elapsed = time.perf_counter() - tick
                    rows.append({"players": count, "seed": seed + 2100, "iterations": iterations,
                                 "assisted": assisted, "seconds": elapsed, "nodes": result.work.nodes,
                                 "policy_calls": prior.calls if prior else 0,
                                 "inference_seconds": prior.seconds if prior else 0,
                                 "model_inference_fraction": prior.seconds / elapsed if prior else 0,
                                 "exact_leaf_fraction": 1., "rollout_leaf_fraction": 0., "learned_leaf_fraction": 0.})
    summary = []
    for count in (2, 3):
        for iterations in (10, 100, 500):
            for assisted in (False, True):
                timings = sorted(r["seconds"] for r in rows if (r["players"], r["iterations"], r["assisted"]) == (count, iterations, assisted))
                summary.append({"players": count, "iterations": iterations, "assisted": assisted,
                                "median_seconds": statistics.median(timings), "maximum_seconds": max(timings),
                                "replicates": len(timings)})
    result = {"scope": "small exact river fixtures; five held-out seeds per setting, max rather than a misleading five-point p99", "summary": summary, "measurements": rows}
    path.write_text(json.dumps(result, indent=2) + "\n")
    return result
