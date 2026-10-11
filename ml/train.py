"""Central CPU training with explicit held-out metrics. No automatic training job."""
from __future__ import annotations
import random
import math
from collections.abc import Callable
import torch
from ml.memory import TrainingSample
from ml.model import AdvantageNetwork, AveragePolicyNetwork, StreetNetworks, encode_batch


def _sample_weight(sample: TrainingSample) -> float:
    weight = sample.weight * sample.iteration
    if not math.isfinite(weight) or weight <= 0:
        raise ValueError(f"Invalid iteration-weighted sample: weight={sample.weight}, iteration={sample.iteration}, product={weight}")
    return weight


def _mean_weight(samples: list[TrainingSample]) -> float:
    weights = [_sample_weight(sample) for sample in samples]
    if not weights:
        raise ValueError("Mean weight requires nonempty samples")
    scale = max(weights)
    # Summing raw, finite weights can overflow float64. The scaled mean is <=1.
    return scale * (math.fsum(weight / scale for weight in weights) / len(weights))


def _loss_with_weights(model, samples: list[TrainingSample], normalized: list[float],
                       batch: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
    if not samples:
        raise ValueError("Loss requires nonempty samples")
    if batch is None:
        batch = encode_batch([s.state for s in samples], model.feature_version)
    target = torch.tensor([s.target for s in samples])
    if not torch.isfinite(target).all():
        raise ValueError("Training target cannot be represented as finite float32")
    nonzero = torch.tensor([[value != 0 for value in sample.target] for sample in samples], dtype=torch.bool)
    if ((target == 0) & nonzero).any():
        raise ValueError("Nonzero training target underflowed in float32")
    if len(normalized) != len(samples) or any(not math.isfinite(w) or w <= 0 for w in normalized):
        raise ValueError("Explicit finite positive loss weights required for every sample")
    weight = torch.tensor(normalized, dtype=torch.float32)
    if not torch.isfinite(weight).all() or (weight <= 0).any():
        raise ValueError("Loss weights overflowed or underflowed in float32")
    output = model(batch)
    if isinstance(model, AdvantageNetwork) or isinstance(model, StreetNetworks) and model.kind == "advantage":
        losses = ((output - target).square() * batch["mask"]).sum(1) / batch["mask"].sum(1)
    elif isinstance(model, AveragePolicyNetwork) or isinstance(model, StreetNetworks) and model.kind == "strategy":
        losses = (target * (target.clamp_min(1e-12).log() - output.clamp_min(1e-12).log())).sum(1)
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")
    loss = (losses * weight).mean()
    if not torch.isfinite(loss):
        raise ValueError(f"Nonfinite weighted training loss: {loss}")
    return loss


def loss_for(model, samples: list[TrainingSample], mean_weight: float | None = None) -> torch.Tensor:
    """Iteration/reach-weighted objective with a fixed population normalization."""
    if not samples:
        raise ValueError("Loss requires nonempty samples")
    if mean_weight is None:
        mean_weight = _mean_weight(samples)
    if not math.isfinite(mean_weight) or mean_weight <= 0:
        raise ValueError(f"Invalid population mean weight: {mean_weight}")
    normalized = [_sample_weight(sample) / mean_weight for sample in samples]
    if any(value == 0 for value in normalized):
        raise ValueError("Positive normalized sample weight underflowed in float64")
    return _loss_with_weights(model, samples, normalized)


def weight_quality(samples: list[TrainingSample]) -> dict[str, float]:
    """Concentration diagnostics for the actual weighted objective, without clipping."""
    values = [_sample_weight(sample) for sample in samples]
    if not values:
        raise ValueError("Weight diagnostics require nonempty samples")
    scale = max(values)
    ratios = [value / scale for value in values]
    total = math.fsum(ratios)
    masses = sorted((value / total for value in ratios), reverse=True)
    return {"weight_effective_samples": 1 / math.fsum(p * p for p in masses),
            "largest_weight_share": masses[0], "top10_weight_share": math.fsum(masses[:10])}


def evaluate_loss(model, samples: list[TrainingSample], batch_size: int) -> float:
    """Report the globally weighted objective while bounding encoded batches."""
    if not samples or batch_size < 1:
        raise ValueError(f"Invalid evaluation batch: samples={len(samples)}, batch_size={batch_size}")
    mean_weight = _mean_weight(samples)
    with torch.no_grad():
        batch_losses = [float(loss_for(model, samples[offset:offset + batch_size], mean_weight))
                        * len(samples[offset:offset + batch_size])
                        for offset in range(0, len(samples), batch_size)]
    return math.fsum(batch_losses) / len(samples)


class EncodedReplay:
    """Fit-local immutable tensors; no cache persists across replay updates."""

    def __init__(self, samples: list[TrainingSample], feature_version: int,
                 byte_budget: int = 256 * 1024 * 1024) -> None:
        from infoset import HISTORY_WIDTH
        from features.neural import numeric_names
        longest = max(len(s.state.history_data) // (8 * HISTORY_WIDTH) for s in samples)
        estimate = len(samples) * (longest * HISTORY_WIDTH * 4
                                  + len(numeric_names(feature_version)) * 4 + 7 * 8 + 3 * 8 + len(samples[0].target))
        if estimate > byte_budget:
            raise MemoryError(f"Encoded replay requires {estimate} bytes; budget={byte_budget}")
        self.batch = encode_batch([s.state for s in samples], feature_version)
        self.indices = {id(s): i for i, s in enumerate(samples)}

    def select(self, samples: list[TrainingSample]) -> dict[str, torch.Tensor]:
        indices = torch.tensor([self.indices[id(s)] for s in samples], dtype=torch.long)
        lengths = self.batch['lengths'].index_select(0, indices)
        return {key: (value[:, :int(lengths.max())] if key == 'history' else value).index_select(0, indices)
                for key, value in self.batch.items()}


def fit(model, samples: list[TrainingSample], epochs: int, batch_size: int, seed: int,
        learning_rate: float = 3e-4, evaluation_fraction: float = 0.2,
        sampling: str = "uniform_shuffle", measurement_epochs: tuple[int, ...] = (),
        on_measurement: Callable[[int, dict[str, float]], None] | None = None,
        max_updates: int | None = None, cache_encoding: bool = False,
        optimizer: torch.optim.Optimizer | None = None,
        explicit_partitions: tuple[list[TrainingSample], list[TrainingSample]] | None = None) -> dict[str, float]:
    if isinstance(model, StreetNetworks):
        raise ValueError("Fit independent specialists through fit_specialists; pooled optimization is unsupported")
    if max_updates is not None and (type(max_updates) is not int or max_updates < 1):
        raise ValueError("max_updates must be a positive integer or None")
    if max_updates is not None and measurement_epochs:
        raise ValueError("Epoch sweeps and update-capped fitting require separate experiments")
    if (any(type(e) is not int or not 1 <= e <= epochs for e in measurement_epochs)
            or tuple(sorted(set(measurement_epochs))) != measurement_epochs
            or bool(measurement_epochs) != (on_measurement is not None)):
        raise ValueError("Measurement epochs must be unique, increasing, within the fit budget, and paired with a callback")
    if sampling not in ("uniform_shuffle", "weighted_replacement"):
        raise ValueError(f"Unknown optimizer sampling mode: {sampling}")
    if len(samples) < 2 or epochs < 1 or batch_size < 1 or not 0 < evaluation_fraction < 1:
        raise ValueError(f"Invalid fit configuration: samples={len(samples)}, epochs={epochs}, batch={batch_size}")
    if next(model.parameters()).device.type != "cpu":
        raise ValueError("This training interface explicitly requires CPU; GPU compute needs a separate approved run")
    for s in samples:
        s.validate()
        if s.state.feature_version != model.feature_version:
            raise ValueError(f"Fit feature mismatch: sample={s.state.feature_version}, model={model.feature_version}")
        expected_kind = "advantage" if isinstance(model, AdvantageNetwork) else "strategy"
        if s.kind != expected_kind:
            raise ValueError(f"Training {expected_kind} model with {s.kind} sample")
    rng = random.Random(seed)
    ordered = list(samples)
    rng.shuffle(ordered)
    cut = max(1, min(len(ordered) - 1, round(len(ordered) * evaluation_fraction)))
    evaluation, training = ordered[:cut], ordered[cut:]
    if explicit_partitions is not None:
        training, evaluation = map(list, explicit_partitions)
        if not training or not evaluation or {id(s) for s in training} & {id(s) for s in evaluation}:
            raise ValueError("Explicit training/validation partitions must be nonempty and disjoint")
    encoded = EncodedReplay(samples, model.feature_version) if cache_encoding else None
    updates = 0
    completed_epochs = 0
    mean_weight = _mean_weight(training)
    optimizer = optimizer if optimizer is not None else torch.optim.Adam(model.parameters(), lr=learning_rate)
    if {id(p) for group in optimizer.param_groups for p in group["params"]} != {id(p) for p in model.parameters()}:
        raise ValueError("Optimizer parameters do not belong to this model")
    model.train()
    weights = [_sample_weight(sample) for sample in training]
    scale = max(weights)
    probabilities = [weight / scale for weight in weights]
    if any(p == 0 for p in probabilities):
        raise ValueError("Positive sampling weight underflowed in float64")
    quality = {**weight_quality(samples),
               "training_weight_effective_samples": weight_quality(training)["weight_effective_samples"],
               "heldout_weight_effective_samples": weight_quality(evaluation)["weight_effective_samples"]}

    def measure() -> dict[str, float]:
        model.eval()
        def loss(rows: list[TrainingSample]) -> float:
            if encoded is None:
                return evaluate_loss(model, rows, batch_size)
            mean = _mean_weight(rows)
            with torch.no_grad():
                return math.fsum(float(_loss_with_weights(model, part, [_sample_weight(s) / mean for s in part],
                                                         encoded.select(part))) * len(part)
                                 for offset in range(0, len(rows), batch_size)
                                 for part in [rows[offset:offset + batch_size]]) / len(rows)
        return {"train_loss": loss(training),
                "heldout_loss": loss(evaluation), "updates_completed": updates,
                "epochs_completed": completed_epochs,
                "training_samples": float(len(training)), "heldout_samples": float(len(evaluation)), **quality}

    metrics = None
    for epoch in range(1, epochs + 1):
        if sampling == "uniform_shuffle":
            rng.shuffle(training)
            epoch_samples = training
        else:
            # p(i)=w(i)/sum(w): unweighted sampled losses estimate the SAME
            # weighted objective without a rare sample producing one huge update.
            epoch_samples = rng.choices(training, weights=probabilities, k=len(training))
        for offset in range(0, len(epoch_samples), batch_size):
            optimizer.zero_grad()
            batch = epoch_samples[offset:offset + batch_size]
            normalized = [_sample_weight(s) / mean_weight for s in batch] if sampling == 'uniform_shuffle' else [1.0] * len(batch)
            objective = _loss_with_weights(model, batch, normalized, encoded.select(batch) if encoded else None)
            objective.backward()
            optimizer.step()
            updates += 1
            if max_updates is not None and updates >= max_updates:
                break
        if offset + len(batch) == len(epoch_samples):
            completed_epochs += 1
        if epoch in measurement_epochs:
            metrics = measure()
            on_measurement(epoch, dict(metrics))
            model.train()
        if max_updates is not None and updates >= max_updates:
            break
    if epochs not in measurement_epochs:
        metrics = measure()
    model.eval()
    return metrics
