"""Strict CPU-only inference from the same small artifacts consumed by the UI."""
from __future__ import annotations

import hashlib
import numpy as np
import onnxruntime as ort

from actions import ACTION_IDS, ACTION_SCHEMA_VERSION
from cfr_solver import validate_strategy
from features.neural import numeric_names
from infoset import STATE_VERSION, Observation
from ml.encoding import encode_arrays


class OnnxPolicy:
    def __init__(self, model: bytes, manifest: dict, count: int):
        if (manifest.get('version') != 4 or manifest.get('state_version') != STATE_VERSION
                or manifest.get('action_schema_version') != ACTION_SCHEMA_VERSION
                or manifest.get('actions') != list(ACTION_IDS)
                or manifest.get('supported_player_counts') != [count]
                or manifest.get('objective') != 'hand_chip_delta'
                or hashlib.sha256(model).hexdigest() != manifest.get('model_sha256')):
            raise ValueError('Incompatible or corrupted evaluation ONNX artifact')
        self.feature_version = manifest['feature_schema_version']
        if manifest.get('numeric_names') != list(numeric_names(self.feature_version)):
            raise ValueError('Evaluation feature schema mismatch')
        self.count = count
        self.manifest = manifest
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(model, options, providers=['CPUExecutionProvider'])
        self.inputs = tuple(i.name for i in self.session.get_inputs())
        if set(self.inputs) != {'cards', 'street', 'position', 'numeric', 'history', 'mask'}:
            raise ValueError('Unexpected ONNX inputs')

    def probabilities(self, observation: Observation) -> tuple[float, ...]:
        from infoset import NUMERIC_NAMES
        count = round(observation.numeric[NUMERIC_NAMES.index('player_count')] * 3)
        if count != self.count:
            raise ValueError('Evaluation policy player-count mismatch')
        arrays = encode_arrays([observation], self.feature_version)
        output = self.session.run(['probabilities'], {k: arrays[k] for k in self.inputs})[0]
        if output.shape != (1, len(ACTION_IDS)) or not np.isfinite(output).all():
            raise ValueError('Invalid ONNX probabilities')
        total = float(output[0].astype(np.float64).sum())
        if abs(total - 1.0) > 1e-6 or np.any(output[0] < 0) or np.any(output[0][~arrays['mask'][0]] != 0):
            raise ValueError('ONNX output is not a legal probability distribution')
        # Match PyTorch/browser normalization after float32 softmax roundoff.
        values = tuple(float(v) / total for v in output[0])
        validate_strategy(values, observation.legal_mask)
        return values
