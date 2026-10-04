"""Central CPU training with explicit held-out metrics. No automatic training job."""
from __future__ import annotations
import random
import math
import torch
from ml.memory import TrainingSample
from ml.model import AdvantageNetwork, AveragePolicyNetwork, encode_batch


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


def loss_for(model, samples: list[TrainingSample], mean_weight: float | None = None) -> torch.Tensor:
    """Weighted objective; SGD passes the fixed training population mean weight.

    Uniformly shuffled minibatches then estimate sum(w * loss) / sum(w)
    without the bias from separately normalizing each random minibatch.
    """
    if not samples:
        raise ValueError("Loss requires nonempty samples")
    batch = encode_batch([s.state for s in samples])
    target = torch.tensor([s.target for s in samples])
    if not torch.isfinite(target).all():
        raise ValueError("Training target cannot be represented as finite float32")
    nonzero = torch.tensor([[value != 0 for value in sample.target] for sample in samples], dtype=torch.bool)
    if ((target == 0) & nonzero).any():
        raise ValueError("Nonzero training target underflowed in float32")
    if mean_weight is None:
        mean_weight = _mean_weight(samples)
    if not math.isfinite(mean_weight) or mean_weight <= 0:
        raise ValueError(f"Invalid population mean weight: {mean_weight}")
    # Normalize in float64 before conversion; importance weights can exceed the
    # float32 range although their normalized ratios are bounded by sample count.
    normalized = [_sample_weight(sample) / mean_weight for sample in samples]
    if any(value == 0 for value in normalized):
        raise ValueError("Positive normalized sample weight underflowed in float64")
    weight = torch.tensor(normalized, dtype=torch.float32)
    if not torch.isfinite(weight).all():
        raise ValueError(f"Nonfinite normalized sample weights for population mean={mean_weight}")
    if (weight <= 0).any():
        raise ValueError("Positive normalized sample weight underflowed in float32; importance weights cannot be dropped")
    output = model(batch)
    if isinstance(model, AdvantageNetwork):
        losses = ((output - target).square() * batch["mask"]).sum(1) / batch["mask"].sum(1)
    elif isinstance(model, AveragePolicyNetwork):
        losses = (target * (target.clamp_min(1e-12).log() - output.clamp_min(1e-12).log())).sum(1)
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")
    loss = (losses * weight).mean()
    if not torch.isfinite(loss):
        raise ValueError(f"Nonfinite weighted training loss: {loss}")
    return loss


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


def fit(model, samples: list[TrainingSample], epochs: int, batch_size: int, seed: int,
        learning_rate: float = 3e-4, evaluation_fraction: float = 0.2) -> dict[str, float]:
    if len(samples) < 2 or epochs < 1 or batch_size < 1 or not 0 < evaluation_fraction < 1:
        raise ValueError(f"Invalid fit configuration: samples={len(samples)}, epochs={epochs}, batch={batch_size}")
    if next(model.parameters()).device.type != "cpu":
        raise ValueError("This training interface explicitly requires CPU; GPU compute needs a separate approved run")
    for s in samples:
        s.validate()
        expected_kind = "advantage" if isinstance(model, AdvantageNetwork) else "strategy"
        if s.kind != expected_kind:
            raise ValueError(f"Training {expected_kind} model with {s.kind} sample")
    rng = random.Random(seed)
    ordered = list(samples)
    rng.shuffle(ordered)
    cut = max(1, min(len(ordered) - 1, round(len(ordered) * evaluation_fraction)))
    evaluation, training = ordered[:cut], ordered[cut:]
    mean_weight = _mean_weight(training)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for _ in range(epochs):
        rng.shuffle(training)
        for offset in range(0, len(training), batch_size):
            optimizer.zero_grad()
            loss_for(model, training[offset:offset + batch_size], mean_weight).backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        train_loss = evaluate_loss(model, training, batch_size)
        evaluation_loss = evaluate_loss(model, evaluation, batch_size)
    return {"train_loss": train_loss, "heldout_loss": evaluation_loss,
            "training_samples": float(len(training)), "heldout_samples": float(len(evaluation))}
