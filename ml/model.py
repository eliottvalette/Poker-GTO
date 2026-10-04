"""Modest exact-card, numeric MLP and full-history GRU encoder."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from actions import ACTION_IDS
from infoset import HISTORY_WIDTH, NUMERIC_NAMES, STATE_VERSION, Observation

MODEL_ARCHITECTURE = "cards8_numeric32_historyGRU32_head64_v3"


def encode_batch(observations: list[Observation]) -> dict[str, torch.Tensor]:
    if not observations:
        raise ValueError("Cannot encode an empty observation batch")
    for o in observations:
        if (o.version != STATE_VERSION or len(o.cards) != 7 or any(c not in range(53) for c in o.cards)
                or len(o.numeric) != len(NUMERIC_NAMES) or len(o.legal_mask) != len(ACTION_IDS)
                or not any(o.legal_mask) or not o.history or o.street not in range(4)
                or any(len(e) != HISTORY_WIDTH for e in o.history)):
            raise ValueError(f"Invalid neural observation contract: {o}")
    lengths = torch.tensor([len(o.history) for o in observations], dtype=torch.long)
    history = torch.zeros(len(observations), int(lengths.max()), HISTORY_WIDTH)
    for i, o in enumerate(observations):
        history[i, :len(o.history)] = torch.tensor(o.history)
    numeric = torch.tensor([o.numeric for o in observations], dtype=torch.float32)
    if not torch.isfinite(numeric).all() or not torch.isfinite(history).all():
        raise ValueError("Nonfinite numerical/history input")
    return {"cards": torch.tensor([o.cards for o in observations], dtype=torch.long),
            "street": torch.tensor([o.street for o in observations], dtype=torch.long),
            "position": torch.tensor([round(o.numeric[NUMERIC_NAMES.index("hero_position")] * 2) for o in observations], dtype=torch.long),
            "numeric": numeric, "history": history, "lengths": lengths,
            "mask": torch.tensor([o.legal_mask for o in observations], dtype=torch.bool)}


class StateEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.cards = nn.Embedding(53, 8)
        self.street = nn.Embedding(4, 4)
        self.position = nn.Embedding(3, 4)
        self.numeric = nn.Sequential(nn.Linear(len(NUMERIC_NAMES), 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())
        self.history = nn.GRU(HISTORY_WIDTH, 32, batch_first=True)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        sequence = pack_padded_sequence(batch["history"], batch["lengths"].cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.history(sequence)
        return torch.cat((self.cards(batch["cards"]).flatten(1), self.street(batch["street"]),
                          self.position(batch["position"]), self.numeric(batch["numeric"]), hidden[-1]), dim=1)


class AdvantageNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = StateEncoder()
        self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, len(ACTION_IDS)))

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        values = self.head(self.encoder(batch))
        if not torch.isfinite(values).all():
            raise ValueError("Advantage network produced nonfinite values")
        return values


class AveragePolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = StateEncoder()
        self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, len(ACTION_IDS)))

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        logits = self.head(self.encoder(batch))
        if not torch.isfinite(logits).all() or not batch["mask"].any(dim=1).all():
            raise ValueError("Average policy has invalid outputs or empty legal masks")
        return logits.masked_fill(~batch["mask"], -torch.inf).softmax(dim=1)
