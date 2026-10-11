"""Frozen historical supervision probes, separate from independent poker evaluation."""
from __future__ import annotations
from collections import Counter
import random
import torch
from ml.memory import TrainingSample
from ml.train import evaluate_loss, weight_quality
from ml.model import encode_batch
from training.metrics import decision_context, opening_hand_class
from features.neural import NeuralObservation


def partition_coverage(partitions: dict) -> dict:
    result = {}
    for split, samples in partitions.items():
        openings = Counter(h for s in samples if (h := opening_hand_class(s.state)) is not None)
        result[split] = {'samples':len(samples), 'root_groups':len({s.root_group for s in samples}),
                         'contexts':dict(Counter(decision_context(s.state) for s in samples)),
                         'opening_classes':dict(sorted(openings.items())),
                         **(weight_quality(samples) if samples else {})}
    return result


def fixed_metrics(model, state: dict, partitions: dict, seed: int, version: int, kind: str) -> dict:
    from training.checkpoint import _pack_sample
    if 'validation_bank' not in state:
        if min(len(partitions[k]) for k in ('validation','test')) < 64:
            return {'status':'waiting_for_64_records_per_heldout_split'}
        rng = random.Random(seed)
        state['validation_bank'] = {'version':1, 'source_iteration':version, 'splits':{
            k:[_pack_sample(s) for s in rng.sample(partitions[k],min(128,len(partitions[k])))]
            for k in ('validation','test')}}
    bank = state['validation_bank']
    result = {'source_iteration':bank['source_iteration'],
              'scope':'fixed historical regret/strategy targets; not independent action-EV ground truth'}
    for split, packed in bank['splits'].items():
        samples = [TrainingSample(**{**s,'state':NeuralObservation(**s['state'])}) for s in packed]
        weights = [s.iteration*s.weight for s in samples]
        total = sum(weights)
        if kind == 'advantage':
            baseline = sum(w*sum(t*t for t,legal in zip(s.target,s.state.legal_mask) if legal)/sum(s.state.legal_mask)
                           for s,w in zip(samples,weights))/total
            with torch.no_grad():
                batch = encode_batch([s.state for s in samples],model.feature_version)
                all_nonpositive = float((model(batch).masked_fill(~batch['mask'],-torch.inf).max(1).values <= 0).float().mean())
            extra = {'zero_predictor_loss':baseline, 'all_nonpositive_fraction':all_nonpositive}
        else:
            import math
            baseline = sum(w*sum(t*math.log(t*sum(s.state.legal_mask)) for t in s.target if t>0)
                           for s,w in zip(samples,weights))/total
            extra = {'uniform_policy_kl':baseline}
        result[split] = {'samples':len(samples), 'root_groups':len({s.root_group for s in samples}),
                         'trajectories':len({s.trajectory_id for s in samples}),
                         'loss':evaluate_loss(model,samples,128), **weight_quality(samples), **extra}
    return result
