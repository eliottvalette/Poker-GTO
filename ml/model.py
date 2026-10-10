"""Modest exact-card, numeric MLP and full-history GRU encoder."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from actions import ACTION_IDS
from infoset import HISTORY_WIDTH, Observation
from features import FEATURE_SCHEMA_VERSION, SUPPORTED_FEATURE_VERSIONS
from features.neural import NeuralObservation, numeric_names

def model_architecture(feature_version: int) -> str:
    if feature_version not in SUPPORTED_FEATURE_VERSIONS:
        raise ValueError(f"Unsupported model feature schema: {feature_version}")
    return f"cards8_numeric32_historyGRU32_head64_features{feature_version}"


MODEL_ARCHITECTURE = model_architecture(FEATURE_SCHEMA_VERSION)


def encode_batch(observations: list[Observation | NeuralObservation],
                 feature_version: int = FEATURE_SCHEMA_VERSION) -> dict[str, torch.Tensor]:
    from ml.encoding import encode_arrays
    return {name: torch.from_numpy(value) for name, value in encode_arrays(observations, feature_version).items()}




class StateEncoder(nn.Module):
    def __init__(self, feature_version: int = FEATURE_SCHEMA_VERSION):
        super().__init__()
        self.cards = nn.Embedding(53, 8)
        self.street = nn.Embedding(4, 4)
        self.position = nn.Embedding(3, 4)
        self.numeric = nn.Sequential(nn.Linear(len(numeric_names(feature_version)), 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())
        self.history = nn.GRU(HISTORY_WIDTH, 32, batch_first=True)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        sequence = pack_padded_sequence(batch["history"], batch["lengths"].cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.history(sequence)
        return torch.cat((self.cards(batch["cards"]).flatten(1), self.street(batch["street"]),
                          self.position(batch["position"]), self.numeric(batch["numeric"]), hidden[-1]), dim=1)


class AdvantageNetwork(nn.Module):
    def __init__(self, feature_version: int = FEATURE_SCHEMA_VERSION):
        super().__init__()
        model_architecture(feature_version)
        self.feature_version = feature_version
        self.encoder = StateEncoder(feature_version)
        self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, len(ACTION_IDS)))

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        values = self.head(self.encoder(batch))
        if not torch.isfinite(values).all():
            raise ValueError("Advantage network produced nonfinite values")
        return values


class AveragePolicyNetwork(nn.Module):
    def __init__(self, feature_version: int = FEATURE_SCHEMA_VERSION):
        super().__init__()
        model_architecture(feature_version)
        self.feature_version = feature_version
        self.encoder = StateEncoder(feature_version)
        self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, len(ACTION_IDS)))

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        logits = self.head(self.encoder(batch))
        if not torch.isfinite(logits).all() or not batch["mask"].any(dim=1).all():
            raise ValueError("Average policy has invalid outputs or empty legal masks")
        return logits.masked_fill(~batch["mask"], -torch.inf).softmax(dim=1)

    def probabilities(self, observation: Observation | NeuralObservation) -> tuple[float, ...]:
        """Convert float32 softmax output to a normalized float64 distribution.

        Tensor softmax sums may differ from one by float32 rounding. Check that
        bound explicitly, then normalize at the numerical transport boundary.
        """
        from cfr_solver import validate_strategy
        with torch.no_grad():
            values = tuple(self(encode_batch([observation], self.feature_version))[0].tolist())
        total = sum(values)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Average-policy softmax mass outside float32 tolerance: {total}")
        result = tuple(value / total for value in values)
        validate_strategy(result, observation.legal_mask)
        return result
