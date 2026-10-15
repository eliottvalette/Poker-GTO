"""Observable activation, gradient and target-scale diagnostics for fixed data."""
from __future__ import annotations

import math
import torch
from torch import nn

from ml.memory import TrainingSample
from ml.model import AdvantageNetwork, encode_batch
from ml.train import loss_for, weight_quality


def numerical_health(model: nn.Module, samples: list[TrainingSample]) -> dict:
    if not samples:
        raise ValueError("Numerical health requires observations")
    batch = encode_batch([s.state for s in samples], model.feature_version)
    captured = []
    hooks = [layer.register_forward_pre_hook(lambda _layer, inputs: captured.append(inputs[0].detach()))
             for layer in model.head if isinstance(layer, nn.ReLU)]
    try:
        output = model(batch)
        objective = loss_for(model, samples)
        parameters = list(model.parameters())
        gradients = torch.autograd.grad(objective, parameters, allow_unused=True)
    finally:
        for hook in hooks:
            hook.remove()
    targets = torch.tensor([s.target for s in samples])[batch["mask"]]
    values = output.detach()[batch["mask"]]
    if not torch.isfinite(values).all():
        raise ValueError("Nonfinite numerical-health output")
    return {
        "samples": len(samples), "loss": float(objective.detach()), **weight_quality(samples),
        "target_abs_max": float(targets.abs().max()),
        "target_rms": float(targets.square().mean().sqrt()),
        "prediction_abs_max": float(values.abs().max()),
        "head_dead_fraction": max((float((x <= 0).all(1).float().mean()) for x in captured), default=None),
        "head_preactivation_max": max((float(x.max()) for x in captured), default=None),
        "all_nonpositive_fraction": float((output.detach().masked_fill(~batch["mask"], -torch.inf).max(1).values <= 0).float().mean())
        if isinstance(model, AdvantageNetwork) else None,
        "gradient_norms": {name: float(g.norm()) if g is not None else None
                           for (name, _), g in zip(model.named_parameters(), gradients)},
        "encoder_gradient_norm": math.sqrt(sum(float(g.square().sum()) for (name, _), g in
                                               zip(model.named_parameters(), gradients) if name.startswith("encoder.") and g is not None)),
    }
