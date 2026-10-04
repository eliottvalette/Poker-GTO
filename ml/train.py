"""Central CPU training with explicit held-out metrics. No automatic training job."""
from __future__ import annotations
import random
import math
import torch
from ml.memory import TrainingSample
from ml.model import AdvantageNetwork, AveragePolicyNetwork, encode_batch


def loss_for(model, samples: list[TrainingSample], mean_weight: float | None = None) -> torch.Tensor:
    """Weighted objective; SGD passes the fixed training population mean weight.

    Uniformly shuffled minibatches then estimate sum(w * loss) / sum(w)
    without the bias from separately normalizing each random minibatch.
    """
    if not samples:
        raise ValueError("Loss requires nonempty samples")
    batch = encode_batch([s.state for s in samples])
    target = torch.tensor([s.target for s in samples])
    weight = torch.tensor([s.weight * s.iteration for s in samples], dtype=torch.float32)
    output = model(batch)
    if isinstance(model, AdvantageNetwork):
        losses = ((output - target).square() * batch["mask"]).sum(1) / batch["mask"].sum(1)
    elif isinstance(model, AveragePolicyNetwork):
        losses = (target * (target.clamp_min(1e-12).log() - output.clamp_min(1e-12).log())).sum(1)
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")
    if mean_weight is None:
        mean_weight = math.fsum(s.weight * s.iteration for s in samples) / len(samples)
    if not math.isfinite(mean_weight) or mean_weight <= 0:
        raise ValueError(f"Invalid population mean weight: {mean_weight}")
    loss = (losses * weight).mean() / mean_weight
    if not torch.isfinite(loss):
        raise ValueError(f"Nonfinite weighted training loss: {loss}")
    return loss


def evaluate_loss(model, samples: list[TrainingSample], batch_size: int) -> float:
    """Report the globally weighted objective while bounding encoded batches."""
    if not samples or batch_size < 1:
        raise ValueError(f"Invalid evaluation batch: samples={len(samples)}, batch_size={batch_size}")
    mean_weight = math.fsum(s.weight * s.iteration for s in samples) / len(samples)
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
    mean_weight = math.fsum(s.weight * s.iteration for s in training) / len(training)
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
