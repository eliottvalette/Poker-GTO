"""Diverse grouped data, capacity comparisons and explicit model-quality gates."""

from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import time

import torch

from actions import ACTION_IDS, legal_actions
from cfr_solver import sample_index
from hybrid.continuation import continuation_features
from hybrid.learning import Learner, Supervision, observation_features
from hybrid.policy_source import distribution
from hybrid.quality import scenarios
from hybrid.reference import ProfilePolicy
from hybrid.search import search
from hybrid.state import ComputeBudget, instantiate
from hybrid.value_features import suit_aware_features
from infoset import observe
from poker_game_expresso import HandState


def behavior_data(
    count: int, profile: str, *, hands: int = 12, seed: int = 701
) -> tuple[list[Supervision], list[dict]]:
    teacher = ProfilePolicy(profile)
    rng = random.Random(seed)
    records, outcomes = [], []
    for index in range(hands):
        stacks = [3.0, 10.0, 25.0]
        state = HandState.start(
            {p: stacks[(index + p) % 3] for p in range(count)}, index % count, rng
        )
        steps = 0
        while not state.terminal:
            if steps >= 100:
                raise RuntimeError(
                    "Self-play exceeded explicit 100-decision hand budget"
                )
            obs = observe(state)
            target = teacher.probabilities(obs)
            records.append(
                Supervision(
                    observation_features(obs),
                    target,
                    obs.legal_mask,
                    teacher.version,
                    count,
                    {
                        "profile": profile,
                        "seed": seed,
                        "street": state.street,
                        "max_decisions": 100,
                    },
                    0.0,
                    f"trajectory/{seed}/{index}",
                )
            )
            selected = ACTION_IDS[sample_index(target, rng)]
            next(a for a in legal_actions(state) if a.action_id == selected).apply(
                state
            )
            steps += 1
        outcomes.append(
            {
                "seed": seed,
                "hand": index,
                "steps": steps,
                "utilities": {p: state.utility(p) for p in state.players},
            }
        )
    return records, outcomes


def prediction_metrics(learner: Learner, records: list[Supervision]) -> dict:
    with torch.no_grad():
        prediction = learner.model(
            torch.tensor([r.features for r in records], dtype=torch.float32)
        )
        mask = torch.tensor([r.mask for r in records])
        p = prediction.masked_fill(~mask, -torch.inf).softmax(1)
        targets = torch.tensor([r.target for r in records])
        loss = float(-(targets * p.clamp_min(1e-12).log()).sum(1).mean())
        brier = float((p - targets).square().sum(1).mean())
        confidence, chosen = p.max(1)
        accuracy = targets.gather(1, chosen[:, None])[:, 0]
        bins = []
        for i in range(10):
            take = (confidence >= i / 10) & (
                confidence < (i + 1) / 10 if i < 9 else confidence <= 1
            )
            if take.any():
                bins.append(
                    {
                        "count": int(take.sum()),
                        "confidence": float(confidence[take].mean()),
                        "true_probability": float(accuracy[take].mean()),
                    }
                )
    return {
        "log_loss": loss,
        "brier": brier,
        "calibration_bins": bins,
        "ece": sum(
            b["count"] * abs(b["confidence"] - b["true_probability"]) for b in bins
        )
        / len(records),
        "uniform_log_loss": sum(math.log(sum(r.mask)) for r in records) / len(records),
    }


def fit_behaviors(
    directory: Path,
    *,
    hands: int = 12,
    epochs: tuple[int, ...] = (20, 50),
    widths: tuple[int, ...] = (32, 96),
    profiles: tuple[str, ...] = ("tight_passive", "loose_aggressive"),
) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    report = {
        "budget": {
            "hands": hands,
            "epochs": epochs,
            "widths": widths,
            "profiles": profiles,
        },
        "models": [],
    }
    for count in (2, 3):
        for profile in profiles:
            records, outcomes = behavior_data(count, profile, hands=hands)
            validation, _ = behavior_data(
                count, profile, hands=max(4, hands // 2), seed=1701
            )
            unseen, _ = behavior_data(count, "push_fold", hands=4, seed=2701)
            for width in widths:
                started = time.perf_counter()
                learner = Learner(records, "behavior", seed=81, hidden_width=width)
                sweep = [learner.fit_to(epoch) for epoch in epochs]
                baseline = sum(
                    math.log(sum(records[i].mask)) for i in learner.held_indices
                ) / len(learner.held_indices)
                learner.validate(baseline, baseline * 0.95)
                metrics = prediction_metrics(learner, validation)
                learner.accepted = (
                    learner.accepted
                    and metrics["log_loss"] < metrics["uniform_log_loss"]
                    and metrics["ece"] < 0.1
                )
                learner.validation.update(
                    accepted=learner.accepted,
                    profile=profile,
                    independent_validation=metrics,
                    scope="selected synthetic profile only; OOD populations not accepted",
                )
                name = f"behavior-{count}-{profile}-w{width}"
                learner.save(directory / f"{name}.pt")
                resumed = Learner.load(directory / f"{name}.pt")
                learner.fit_to(epochs[-1] + 1)
                resumed.fit_to(epochs[-1] + 1)
                deterministic = all(
                    torch.equal(a, b)
                    for a, b in zip(
                        learner.model.parameters(), resumed.model.parameters()
                    )
                )
                if not deterministic:
                    raise AssertionError("Behavior optimizer resume mismatch")
                learner.validate(baseline, baseline * 0.95)
                metrics = prediction_metrics(learner, validation)
                learner.accepted = (
                    learner.accepted
                    and metrics["log_loss"] < metrics["uniform_log_loss"]
                    and metrics["ece"] < 0.1
                )
                learner.validation.update(
                    accepted=learner.accepted,
                    profile=profile,
                    independent_validation=metrics,
                    scope="selected synthetic profile only; OOD populations not accepted",
                )
                # Keep checkpoint and exports at the exact same final optimizer step.
                learner.save(directory / f"{name}.pt")
                learner.export(directory / f"{name}.json")
                exported = learner.export_onnx(directory / f"{name}.onnx")
                report["models"].append(
                    {
                        "name": name,
                        "count": count,
                        "profile": profile,
                        "width": width,
                        "labels": len(records),
                        "trajectories": hands,
                        "sweep": sweep,
                        "validation": learner.validation,
                        "ood": prediction_metrics(learner, unseen),
                        "outcomes": outcomes,
                        "resume_exact": deterministic,
                        "onnx_error": exported["maximum_absolute_error"],
                        "seconds": time.perf_counter() - started,
                        "coverage": {
                            street: sum(
                                r.compute_budget["street"] == street for r in records
                            )
                            for street in ("PREFLOP", "FLOP", "TURN", "RIVER")
                        },
                    }
                )
    (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def fit_values(
    directory: Path,
    *,
    per_cell: int = 2,
    samples: int = 128,
    epochs: tuple[int, ...] = (20, 50),
    widths: tuple[int, ...] = (32, 96),
    profile: str = "conservative",
) -> dict:
    """Independent seeded terminal-rollout labels; finite MC error is retained.

    This experiment changes representation and capacity on exactly the same labels.
    It does not claim fixed-policy continuation labels are equilibrium values.
    """
    directory.mkdir(parents=True, exist_ok=True)
    report = {
        "budget": {
            "per_cell": per_cell,
            "label_samples": samples,
            "epochs": epochs,
            "widths": widths,
        },
        "models": [],
    }
    all_records = {
        (n, kind): [] for n in (2, 3) for kind in ("class169-v1", "combo1326-v2")
    }
    started = time.perf_counter()
    for scenario in scenarios("train", per_cell):
        root, belief = scenario.state, scenario.beliefs
        policy = ProfilePolicy(profile)
        hero = root.current_player
        feasible = belief.public.marginals()[hero]
        for holding in list(feasible.weights)[:2]:
            private = belief.private_view(hero, holding)
            world = instantiate(root, private.sample(random.Random(141)))
            budget = ComputeBudget(
                samples=samples, max_nodes=500000, max_depth=1, seed=171
            )
            result = search(
                world, private, hero, dict.fromkeys(root.players, policy), budget
            )
            probabilities = distribution(policy, world)
            value = sum(
                probabilities[ACTION_IDS.index(a)] * v
                for a, v in result.action_ev.items()
            )
            # Triangle bound remains valid without assuming action sample independence.
            se = sum(
                probabilities[ACTION_IDS.index(a)] * (result.standard_errors[a] or 0.0)
                for a in result.action_ev
            )
            for kind, features in (
                ("class169-v1", continuation_features),
                ("combo1326-v2", suit_aware_features),
            ):
                all_records[(len(root.players), kind)].append(
                    Supervision(
                        features(world, belief.public, holding),
                        (value,),
                        (),
                        "independent-terminal-rollouts/" + policy.version,
                        len(root.players),
                        {
                            **asdict(budget),
                            "street": root.street,
                            "feature_schema": kind,
                            "profile": profile,
                        },
                        se,
                        scenario.name,
                    )
                )
    report["label_seconds"] = time.perf_counter() - started
    for (count, kind), records in all_records.items():
        for width in widths:
            started = time.perf_counter()
            learner = Learner(records, "continuation", seed=29, hidden_width=width)
            sweep = [learner.fit_to(epoch) for epoch in epochs]
            mean = sum(records[i].target[0] for i in learner.train_indices) / len(
                learner.train_indices
            )
            baseline = sum(
                (records[i].target[0] - mean) ** 2 for i in learner.held_indices
            ) / len(learner.held_indices)
            # Explicit nearest-observable-vector table baseline with held-root exclusion.
            nearest = []
            for i in learner.held_indices:
                row = records[i]
                j = min(
                    learner.train_indices,
                    key=lambda j: sum(
                        (a - b) ** 2 for a, b in zip(row.features, records[j].features)
                    ),
                )
                nearest.append((row.target[0] - records[j].target[0]) ** 2)
            table_mse = sum(nearest) / len(nearest)
            learner.validate(min(baseline, table_mse), 0.01)
            # A raw-value pass is necessary, never sufficient for online promotion.
            raw_pass = learner.accepted
            learner.accepted = False
            learner.validation.update(
                accepted=False,
                raw_value_gate_passed=raw_pass,
                online_gate="not passed; independent online decision advantage required",
                feature_schema=kind,
                label_standard_errors=[
                    records[i].standard_error for i in learner.held_indices
                ],
            )
            name = f"value-{count}-{kind}-w{width}"
            learner.save(directory / f"{name}.pt")
            learner.export(directory / f"{name}.json")
            export = learner.export_onnx(directory / f"{name}.onnx")
            report["models"].append(
                {
                    "name": name,
                    "count": count,
                    "schema": kind,
                    "width": width,
                    "labels": len(records),
                    "independent_roots": len({r.split_group for r in records}),
                    "sweep": sweep,
                    "mean_baseline_mse": baseline,
                    "nearest_table_mse": table_mse,
                    "validation": learner.validation,
                    "seconds": time.perf_counter() - started,
                    "onnx_error": export["maximum_absolute_error"],
                }
            )
    (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
