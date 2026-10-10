"""Shared NumPy feature encoder for training and CPU ONNX evaluation."""
from __future__ import annotations
import numpy as np
from actions import ACTION_IDS
from infoset import HISTORY_WIDTH, STATE_VERSION, Observation
from features import FEATURE_SCHEMA_VERSION
from features.neural import NEURAL_NUMERIC_NAMES as NUMERIC_NAMES, NeuralObservation, neural_observation, numeric_names

def encode_arrays(observations: list[Observation | NeuralObservation],
                 feature_version: int = FEATURE_SCHEMA_VERSION) -> dict[str, np.ndarray]:
    if not observations:
        raise ValueError("Cannot encode an empty observation batch")
    observations = [neural_observation(o, feature_version) for o in observations]
    # Decode packed float64 buffers directly; avoid one Python tuple and tensor per history.
    width = len(numeric_names(feature_version))
    lengths_list = [len(o.history_data) // (8 * HISTORY_WIDTH) for o in observations]
    for o, length in zip(observations, lengths_list):
        if (o.version != STATE_VERSION or len(o.cards) != 7 or any(c not in range(53) for c in o.cards)
                or len(o.numeric_data) != width * 8 or len(o.legal_mask) != len(ACTION_IDS)
                or not any(o.legal_mask) or not length or o.street not in range(4)
                or len(o.history_data) % (8 * HISTORY_WIDTH)):
            raise ValueError(f"Invalid neural observation contract: {o}")
    numeric64 = np.stack([np.frombuffer(o.numeric_data, dtype='<f8') for o in observations])
    history = np.zeros((len(observations), max(lengths_list), HISTORY_WIDTH), dtype=np.float32)
    with np.errstate(over='ignore', invalid='ignore'):
        numeric = numeric64.astype(np.float32)
        for i, (o, length) in enumerate(zip(observations, lengths_list)):
            history[i, :length] = np.frombuffer(o.history_data, dtype='<f8').reshape(length, HISTORY_WIDTH)
    if not np.isfinite(numeric).all() or not np.isfinite(history).all():
        raise ValueError("Nonfinite numerical/history input")
    return {"cards": np.asarray([o.cards for o in observations], dtype=np.int64),
            "street": np.asarray([o.street for o in observations], dtype=np.int64),
            "position": np.rint(numeric64[:, NUMERIC_NAMES.index("hero_position")] * 2).astype(np.int64),
            "numeric": numeric, "history": history,
            "lengths": np.asarray(lengths_list, dtype=np.int64),
            "mask": np.asarray([o.legal_mask for o in observations], dtype=np.bool_)}
