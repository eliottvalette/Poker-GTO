"""Bounded candidate-versus-active evaluation before changing live policy selection."""
from __future__ import annotations
from dataclasses import replace
import json
from ml.onnx_policy import OnnxPolicy, StreetOnnxPolicy
from training.poker_evaluation import evaluate_policy
from training.publication import TRACKS, encoded, validate_pointer


def assess_candidate(storage, track: str, pointer: dict, candidate, budget) -> dict:
    active = validate_pointer(json.loads(storage.get(f'{track}/current.json')),track)
    if active['release_id'] == pointer['release_id']:
        return {'status':'active', 'active_iteration':active['iteration'], 'reason':'already_activated'}
    prefix = f"{track}/releases/{active['release_id']}"
    manifest = json.loads(storage.get(f'{prefix}/average_{track}.json'))
    reference = (StreetOnnxPolicy(manifest,TRACKS[track],lambda name:storage.get(f'{prefix}/{name}'))
                 if manifest.get('version') == 5 else
                 OnnxPolicy(storage.get(f'{prefix}/average_{track}.onnx'),manifest,TRACKS[track]))
    report, _ = evaluate_policy(candidate,reference,TRACKS[track],replace(budget,seed_offset=3000007))
    interval = report['difference']['interval']
    accepted = report['status'] == 'complete' and interval is not None and interval[0] > 0
    result = {'version':1, 'status':'accepted' if accepted else 'not_accepted',
              'active_iteration':active['iteration'], 'compared_release':active['release_id'],
              'difference':report['difference'], 'seed_offset':3000007,
              'reason':'positive_lower_95_bound' if accepted else 'improvement_not_established',
              'scope':'controlled opponent pool on separate seeds; repeated gates are not general poker certification'}
    if report['status'] != 'complete':
        result.update(reason=report['status'],warnings=report['warnings'])
    if accepted:
        current = validate_pointer(json.loads(storage.get(f'{track}/current.json')),track)
        if current != active:
            raise ValueError('Active policy changed during candidate assessment; refusing stale promotion')
        selected = {**pointer, 'history':[active['release_id'],*active['history']][:23],
                    'activation':result}
        storage.put(f'{track}/current.json',encoded(selected),mutable=True)
        if json.loads(storage.get(f'{track}/current.json')) != selected:
            raise ValueError('Candidate activation read-back mismatch')
    return result
