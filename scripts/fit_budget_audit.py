"""Controlled CPU fits of immutable feature-v2 replay; never resume CFR."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import io
from itertools import combinations
import json
from pathlib import Path
import random
import time

import torch

from actions import ACTION_IDS
from cfr_solver import regret_matching
from features.neural import NeuralObservation
from infoset import Observation, NUMERIC_NAMES, observe
from ml.memory import TrainingSample
from ml.model import AdvantageNetwork, AveragePolicyNetwork, encode_batch
from ml.train import fit
from poker_game_expresso import HandState
from training.checkpoint import CONTRACT, atomic_bytes
from training.config import config_hash

EPOCHS = (2, 5, 10, 20, 50)
HOLDINGS = {"AA": (48, 49), "KK": (44, 45), "AKs": (44, 48),
            "AKo": (45, 48), "72o": (1, 20), "32o": (1, 4)}


def read_frozen_replay(path: Path, track: str) -> tuple[dict, dict]:
    """Require the exact historical contract, checksum, and config fingerprint."""
    envelope = torch.load(path, map_location="cpu", weights_only=True)
    if set(envelope) != {"sha256", "payload"} or hashlib.sha256(envelope["payload"]).hexdigest() != envelope["sha256"]:
        raise ValueError(f"Invalid frozen checkpoint checksum: {path}")
    raw = torch.load(io.BytesIO(envelope["payload"]), map_location="cpu", weights_only=True)
    expected = {**CONTRACT, "checkpoint": 2, "feature": 2,
                "model": "cards8_numeric32_historyGRU32_head64_features2", "root_sampler": 1}
    del expected["external_strategy_collectors"]
    if raw["contract"] != expected or config_hash(raw["config"]) != raw["config_hash"]:
        raise ValueError(f"Frozen replay requires the original feature-v2 checkpoint contract: {path}")
    if track not in raw["tracks"] or raw["tracks"][track]["traversal_mode"] != "external_sampling":
        raise ValueError(f"Missing external-sampling track {track}: {path}")
    metadata = {"path": str(path.resolve()), "payload_sha256": envelope["sha256"],
                "bytes": path.stat().st_size, "iteration": raw["iteration"], "config": raw["config"]}
    return raw["tracks"][track], metadata


def replay_samples(track: dict, kind: str) -> list[TrainingSample]:
    memory = track[f"{kind}_memory"]
    samples = [TrainingSample(**{**row, "state": NeuralObservation(**row["state"])}) for row in memory["samples"]]
    if len(samples) != min(memory["capacity"], memory["seen"]) or any(s.state.feature_version != 2 for s in samples):
        raise ValueError("Frozen replay sample count or feature contract mismatch")
    return samples


def with_private_cards(obs: Observation, combo: tuple[int, int]) -> Observation:
    """Change private cards and their two observable history tokens together."""
    count = round(obs.numeric[NUMERIC_NAMES.index("player_count")] * 3)
    history = [list(row) for row in obs.history]
    for index, card in enumerate(combo):
        history[count + index][10] = card / 51
    return replace(obs, cards=(*combo, *obs.cards[2:]), history=tuple(tuple(row) for row in history))


def opening_predictions(model: AdvantageNetwork | AveragePolicyNetwork, count: int, batch_size: int) -> dict:
    root = HandState.start({i: 75.0 / count for i in range(count)}, 0, random.Random(7))
    base = observe(root)
    combos = list(combinations(range(52), 2))
    observations = [with_private_cards(base, combo) for combo in combos]
    output, reverse_output = [], []
    with torch.no_grad():
        for offset in range(0, len(observations), batch_size):
            batch = observations[offset:offset + batch_size]
            output.extend(model(encode_batch(batch, model.feature_version)).tolist())
            reversed_batch = [with_private_cards(o, (o.cards[1], o.cards[0])) for o in batch]
            reverse_output.extend(model(encode_batch(reversed_batch, model.feature_version)).tolist())
    probabilities = ([regret_matching(row, base.legal_mask) for row in output]
                     if isinstance(model, AdvantageNetwork) else output)
    reversed_probabilities = ([regret_matching(row, base.legal_mask) for row in reverse_output]
                             if isinstance(model, AdvantageNetwork) else reverse_output)
    values = torch.tensor(probabilities, dtype=torch.float64)
    return {"player_count": count, "stacks": [75.0 / count] * count, "hero": base.hero,
            "actions": ACTION_IDS, "holdings": {name: probabilities[combos.index(combo)] for name, combo in HOLDINGS.items()},
            "action_spread": (values.max(0).values - values.min(0).values).tolist(),
            "max_card_order_gap": max(abs(a - b) for row, reverse in zip(probabilities, reversed_probabilities) for a, b in zip(row, reverse)),
            "combos": combos, "outputs": output, "probabilities": probabilities,
            "reversed_probabilities": reversed_probabilities}


def run_fit_budget_audit(checkpoints: dict[str, Path], output: Path,
                        epochs: tuple[int, ...] = EPOCHS) -> dict:
    """Retain every fit milestone and all 1326 holding predictions for both models."""
    if not checkpoints or set(checkpoints) - {"hu", "3max"} or not epochs or tuple(sorted(set(epochs))) != epochs:
        raise ValueError("Explicit HU/3max checkpoints and increasing fit budgets required")
    if output.exists():
        raise FileExistsError(f"Fit experiment already exists: {output}; choose an explicitly different experiment or inspect it")
    output.mkdir(parents=True)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    report = {"feature_version": 2, "sampling": "uniform_shuffle", "epochs": epochs,
              "scope": "fitting budget only; original replay, encoding and weights retained", "tracks": {}}
    try:
        for name, path in checkpoints.items():
            track, source = read_frozen_replay(path, name)
            config = source["config"]
            result = {"source": source, "models": {}}
            report["tracks"][name] = result
            for kind, constructor in (("advantage", AdvantageNetwork), ("strategy", AveragePolicyNetwork)):
                samples = replay_samples(track, kind)
                seed = track["seed"] + (10 if kind == "strategy" else 0)
                initialization = seed + (track["version"] + 1) * 101
                order = list(range(len(samples)))
                random.Random(seed).shuffle(order)
                cut = max(1, min(len(order) - 1, round(len(order) * .2)))
                model_result = {"seed": seed, "initialization_seed": initialization,
                                "heldout_indices": order[:cut], "training_indices": order[cut:], "measurements": []}
                result["models"][kind] = model_result
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(initialization)
                    model = constructor(feature_version=2)
                    started = time.perf_counter()

                    def record(epoch: int, metrics: dict[str, float]) -> None:
                        predictions = opening_predictions(model, 2 if name == "hu" else 3, config["batch_size"])
                        stem = f"{name}-{kind}-epoch-{epoch:02d}"
                        atomic_bytes(output / f"{stem}.json", (json.dumps(predictions, allow_nan=False) + "\n").encode())
                        buffer = io.BytesIO()
                        torch.save({"feature_version": 2, "kind": kind, "epoch": epoch,
                                    "source_sha256": source["payload_sha256"], "weights": model.state_dict()}, buffer)
                        atomic_bytes(output / f"{stem}.pt", buffer.getvalue())
                        row = {"epoch": epoch, "elapsed_seconds": time.perf_counter() - started, **metrics,
                               "holdings": predictions["holdings"], "action_spread": predictions["action_spread"],
                               "max_card_order_gap": predictions["max_card_order_gap"], "artifact": stem}
                        model_result["measurements"].append(row)
                        atomic_bytes(output / "report.json", (json.dumps(report, indent=2, allow_nan=False) + "\n").encode())
                        print(f"{name}/{kind}: epoch={epoch} train={metrics['train_loss']:.6g} heldout={metrics['heldout_loss']:.6g} elapsed={row['elapsed_seconds']:.1f}s", flush=True)

                    fit(model, samples, max(epochs), config["batch_size"], seed, learning_rate=config["learning_rate"],
                        measurement_epochs=epochs, on_measurement=record)
        return report
    finally:
        torch.set_num_threads(previous_threads)
