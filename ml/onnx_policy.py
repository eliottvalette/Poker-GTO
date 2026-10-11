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


def validate_street_catalog(manifest: dict, count: int) -> None:
    from ml.model import STREETS
    import re
    if (manifest.get('version') != 5 or manifest.get('layout') != 'independent_streets_v1'
            or manifest.get('supported_player_counts') != [count]
            or set(manifest.get('routes', {})) != set(STREETS)
            or type(manifest.get('iteration')) is not int or manifest['iteration'] < 1):
        raise ValueError('Invalid street ONNX catalog')
    sources, files = set(), set()
    for street, route in manifest['routes'].items():
        inner = route.get('manifest', {})
        if (route.get('street') != street or route.get('player_count') != count
                or route.get('model_kind') != 'AVERAGE'
                or type(route.get('model_version')) is not int
                or not 1 <= route['model_version'] <= manifest['iteration']
                or not re.fullmatch(r'[a-f0-9]{64}', route.get('checkpoint_source', ''))
                or not re.fullmatch(r'[a-z0-9_]+\.onnx', route.get('model_file', ''))
                or route.get('model_hash') != inner.get('model_sha256')
                or inner.get('supported_player_counts') != [count]
                or inner.get('iteration') != manifest['iteration']
                or route.get('feature_schema') != inner.get('feature_schema_version')
                or route.get('action_schema') != inner.get('action_schema_version')):
            raise ValueError(f'Invalid street ONNX route: {street}')
        sources.add(route['checkpoint_source'])
        files.add(route['model_file'])
    if len(sources) != 1 or len(files) != 4:
        raise ValueError('Street routes must reference one checkpoint and four distinct files')


class StreetOnnxPolicy:
    """Lazy CPU sessions, identical public routing to the browser."""
    def __init__(self, manifest: dict, count: int, load_bytes):
        validate_street_catalog(manifest, count)
        self.manifest, self.count, self.load_bytes = manifest, count, load_bytes
        self.sessions = {}

    def probabilities(self, observation: Observation) -> tuple[float, ...]:
        from ml.model import STREETS
        if observation.street not in range(4):
            raise ValueError('Invalid public street')
        street = STREETS[observation.street]
        if street not in self.sessions:
            route = self.manifest['routes'][street]
            self.sessions[street] = OnnxPolicy(self.load_bytes(route['model_file']), route['manifest'], self.count)
        return self.sessions[street].probabilities(observation)
