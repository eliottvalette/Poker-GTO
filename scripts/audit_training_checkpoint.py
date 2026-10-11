"""Read-only, bounded diagnostics of a checksummed production replay snapshot."""
from __future__ import annotations

from collections import Counter
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from actions import ACTION_IDS
from features.neural import numeric_names
from ml.model import AdvantageNetwork, AveragePolicyNetwork, encode_batch
from training.checkpoint import read_checkpoint, unpack_memory
from training.metrics import opening_hand_class
from training.preflop_diagnostics import opening_matrix
from cfr_solver import regret_matching
from hybrid.state import instantiate
from poker_game_expresso import HandState
from training.poker_evaluation import play, EvaluationBudget, statistics
from ml.train import fit, evaluate_loss
from ml.train import EncodedReplay, _loss_with_weights


class AdvantagePolicy:
    def __init__(self, model: AdvantageNetwork):
        self.model = model

    def probabilities(self, observation):
        with torch.no_grad():
            values = self.model(encode_batch([observation], self.model.feature_version))[0].tolist()
        return regret_matching(values, observation.legal_mask)


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def audit_checkpoint(path: Path, track: str, output: Path) -> dict:
    started = time.monotonic()
    torch.set_num_threads(1)
    raw = read_checkpoint(path)
    data = raw['tracks'][track]
    count = {'hu': 2, '3max': 3}[track]
    output.mkdir(parents=True, exist_ok=True)
    report = {'track': track, 'iteration': raw['iteration'], 'source_sha256': file_hash(path),
              'source_bytes': path.stat().st_size, 'config': raw['config'], 'metadata': raw['metadata'],
              'captured_at': datetime.now(timezone.utc).isoformat(),
              'average_model_iteration': data['metrics'][-1]['average_model_iteration'], 'replay': {}}
    models = {}
    for kind, constructor, weight_key in (('advantage', AdvantageNetwork, 'advantage_weights'),
                                         ('strategy', AveragePolicyNetwork, 'average_weights')):
        model = constructor(raw['config']['feature_schema_version'])
        model.load_state_dict(data[weight_key], strict=True)
        model.eval()
        models[kind] = model
        memory = unpack_memory(data[f'{kind}_memory'])
        samples = memory.samples
        predictions = []
        with torch.no_grad():
            for offset in range(0, len(samples), 128):
                predictions.extend(model(encode_batch([s.state for s in samples[offset:offset+128]],
                                                       model.feature_version)).tolist())
        prediction = np.asarray(predictions, dtype=np.float64)
        target = np.asarray([s.target for s in samples], dtype=np.float64)
        mask = np.asarray([s.state.legal_mask for s in samples], dtype=bool)
        weight = np.asarray([s.iteration * s.weight for s in samples])
        numeric = np.asarray([s.state.numeric for s in samples])
        names = numeric_names(model.feature_version)
        street = np.asarray([s.state.street for s in samples])
        labels = np.asarray([opening_hand_class(s.state) or '' for s in samples])
        opening = labels != ''
        indices = list(range(len(samples)))
        random.Random(data['seed'] + (0 if kind == 'advantage' else 10)).shuffle(indices)
        heldout = np.zeros(len(samples), dtype=bool)
        heldout[indices[:round(len(samples)*.2)]] = True
        if kind == 'advantage':
            losses = ((prediction-target)**2 * mask).sum(1) / mask.sum(1)
            baseline_loss = (target**2 * mask).sum(1) / mask.sum(1)
            policies = np.asarray([regret_matching(p.tolist(), tuple(m.tolist())) for p, m in zip(prediction, mask)])
        else:
            losses = (target * (np.log(target.clip(1e-12)) - np.log(prediction.clip(1e-12)))).sum(1)
            baseline_loss = (target * (np.log(target.clip(1e-12)) + np.log(mask.sum(1))[:, None])).sum(1)
            policies = prediction

        def summarize(selected: np.ndarray) -> dict:
            w = weight[selected]
            if not len(w):
                return {'records': 0, 'weight_share': 0.0}
            mass = w / w.sum()
            result = {'records': int(len(w)), 'weight_share': float(w.sum()/weight.sum()),
                      'ess': float(1/(mass**2).sum()), 'largest_share_within_group': float(mass.max()),
                      'weighted_loss': float(np.dot(mass, losses[selected])),
                      'weighted_baseline_loss': float(np.dot(mass, baseline_loss[selected])),
                      'unweighted_loss': float(losses[selected].mean()),
                      'mean_target': (mass[:, None] * target[selected]).sum(0).tolist(),
                      'mean_prediction': (mass[:, None] * prediction[selected]).sum(0).tolist(),
                      'mean_policy': (mass[:, None] * policies[selected]).sum(0).tolist()}
            if kind == 'advantage':
                nonpositive = np.where(mask[selected], prediction[selected], -np.inf).max(1) <= 0
                result['nonpositive_fraction'] = float(nonpositive.mean())
                result['nonpositive_weight_share_within_group'] = float(np.dot(mass, nonpositive))
            else:
                result['weighted_tv'] = float(np.dot(mass, np.abs(target[selected]-prediction[selected]).sum(1)/2))
            return result

        initial = numeric[:, [names.index(f'initial_{i}') for i in range(count)]] * 25
        matching = opening & np.isclose(initial, 75/count, atol=1e-8).all(1)
        matching &= np.isclose(numeric[:, names.index('chip_unit_big_blind')], 1)
        near = opening & (np.abs(initial - 75/count) <= 5).all(1)
        top_indices = np.argsort(weight)[-20:][::-1]
        report['replay'][kind] = {
            'seen': memory.seen, 'strata': memory.diagnostics(),
            'all': summarize(np.ones(len(samples), dtype=bool)),
            'training_partition_now': summarize(~heldout), 'heldout_partition_now': summarize(heldout),
            'opening': summarize(opening), 'overview_geometry': summarize(matching),
            'near_overview_stacks_within_5bb': summarize(near),
            'by_street': {str(i): summarize(street == i) for i in range(4)},
            'opening_by_class': {label: summarize(labels == label) for label in sorted(set(labels)-{''})},
            'overview_by_class': {label: summarize(matching & (labels == label)) for label in sorted(set(labels)-{''})},
            'top_weight_records': [{'index': int(i), 'weight_share': float(weight[i]/weight.sum()),
                'raw_sample_weight': samples[i].weight, 'iteration': samples[i].iteration,
                'street': int(street[i]), 'cards': samples[i].state.cards,
                'initial_stacks_bb': initial[i].tolist(), 'target': target[i].tolist(),
                'prediction': prediction[i].tolist(), 'heldout_partition_now': bool(heldout[i]),
                'history': samples[i].state.history} for i in top_indices],
            'opening_initial_stack_bins': dict(Counter('under2' if min(v)<2 else '2to8' if min(v)<8 else
                    '8to20' if min(v)<20 else '20plus' for v in initial[opening])),
        }
        np.savez_compressed(output/f'{track}-{kind}-records.npz', predictions=prediction, targets=target,
                            weights=weight, numeric=numeric, mask=mask, street=street, opening_label=labels,
                            iteration=np.asarray([s.iteration for s in samples]), heldout=heldout)
        del memory, samples
        print(track, kind, 'opening mass', report['replay'][kind]['opening']['weight_share'], flush=True)
    report['probes'] = {}
    for kind, model in models.items():
        policy = AdvantagePolicy(model) if kind == 'advantage' else model
        report['probes'][kind] = {str(stack): opening_matrix(policy, count, stack) for stack in (6., 25., 37.5)}
    report['wall_seconds'] = time.monotonic()-started
    write_json(output/f'{track}-audit.json', report)
    torch.save({'track': track, 'iteration': raw['iteration'], 'feature_schema_version': raw['config']['feature_schema_version'],
                'advantage_weights': data['advantage_weights'], 'average_weights': data['average_weights'],
                'source_sha256': report['source_sha256']}, output/f'{track}-models.pt')
    return report


def frozen_replay_study(path: Path, track: str, output: Path, updates: int = 800) -> dict:
    """Equal-update fitting probes; these partitions are not unseen evaluation data.

    The opening-mass arm intentionally changes allocation of approximation error
    between disjoint contexts. It is not an unbiased replacement for the original
    global objective, and nothing from this study is promoted to production.
    """
    if not 1 <= updates <= 1600:
        raise ValueError('Frozen study requires a bounded 1..1600 update budget')
    torch.set_num_threads(1)
    raw = read_checkpoint(path)
    data = raw['tracks'][track]
    count = {'hu': 2, '3max': 3}[track]
    result = {'iteration': raw['iteration'], 'updates_per_arm': updates, 'models': {}}
    for kind, constructor, weight_key in (('advantage', AdvantageNetwork, 'advantage_weights'),
                                         ('strategy', AveragePolicyNetwork, 'average_weights')):
        memory = unpack_memory(data[f'{kind}_memory'])
        samples = memory.samples
        openings = [opening_hand_class(s.state) is not None for s in samples]
        weights = np.asarray([s.weight*s.iteration for s in samples])
        opening_mass = weights[openings].sum()/weights.sum()
        factor = .2*(1-opening_mass)/(.8*opening_mass)
        seed = data['seed'] + (0 if kind == 'advantage' else 10)
        indices = list(range(len(samples)))
        random.Random(seed).shuffle(indices)
        heldout_indices = indices[:round(len(samples)*.2)]
        heldout_opening = [samples[i] for i in heldout_indices if openings[i]]
        heldout_other = [samples[i] for i in heldout_indices if not openings[i]]
        groups = defaultdict(list)
        for i, sample in enumerate(samples):
            if openings[i]:
                state = sample.state
                key = (state.cards, state.numeric_data, state.history_data, state.legal_mask)
                groups[key].append(i)
        with np.load(output/f'{track}-{kind}-records.npz') as archive:
            records = {key: archive[key] for key in ('targets', 'predictions', 'mask')}
        decomposition = []
        for group in groups.values():
            w = weights[group]/weights[group].sum()
            targets = records['targets'][group]
            predictions = records['predictions'][group]
            mean = (w[:, None]*targets).sum(0)
            mask = records['mask'][group[0]]
            if kind == 'strategy':
                noise = (targets*(np.log(targets.clip(1e-12))-np.log(mean.clip(1e-12)))).sum(1)
                residual = (mean*(np.log(mean.clip(1e-12))-np.log(predictions.clip(1e-12)))).sum(1)
            else:
                noise = ((targets-mean)**2*mask).sum(1)/mask.sum()
                residual = ((predictions-mean)**2*mask).sum(1)/mask.sum()
            decomposition.append({'indices': group, 'holding': opening_hand_class(samples[group[0]].state),
                'weight': float(weights[group].sum()), 'records': len(group),
                'noise': float(np.dot(w, noise)), 'approximation_loss': float(np.dot(w, residual)),
                'target_mean': mean.tolist(), 'prediction_mean': (w[:, None]*predictions).sum(0).tolist()})
        write_json(output/f'{track}-{kind}-opening-groups.json', {'groups': decomposition})
        result['models'][kind] = {'opening_multiplier': float(factor), 'unique_opening_observations': len(groups),
            'repeated_observations': sum(len(v)>1 for v in groups.values()), 'arms': {}}
        for arm in ('unchanged', 'warm_original', 'warm_opening_mass20', 'fresh_original'):
            torch.manual_seed(4242)
            model = constructor(raw['config']['feature_schema_version'])
            if arm != 'fresh_original':
                model.load_state_dict(data[weight_key], strict=True)
            model.eval()
            started = time.monotonic()
            fit_metrics = None
            if arm != 'unchanged':
                rows = [replace(s, weight=s.weight*factor) if opening else s
                        for s, opening in zip(samples, openings)] if arm == 'warm_opening_mass20' else samples
                fit_metrics = fit(model, rows, 4, raw['config']['batch_size'], seed,
                    raw['config']['learning_rate'], max_updates=updates, cache_encoding=True)
            arm_result = {'wall_seconds': time.monotonic()-started, 'fit': fit_metrics,
                'opening_loss_original_weights': evaluate_loss(model, heldout_opening, 128),
                'other_loss_original_weights': evaluate_loss(model, heldout_other, 128)}
            root_policy = AdvantagePolicy(model) if kind == 'advantage' else model
            arm_result['opening_matrix'] = opening_matrix(root_policy, count, 75/count)
            result['models'][kind]['arms'][arm] = arm_result
            torch.save(model.state_dict(), output/f'{track}-{kind}-{arm}.pt')
            print(track, kind, arm, 'opening loss', arm_result['opening_loss_original_weights'], flush=True)
            write_json(output/f'{track}-frozen-fit.json', result)
        del memory, samples
    return result


def gradient_audit(path: Path, track: str, output: Path) -> dict:
    """Measure a full training-partition pass at frozen weights, without updates."""
    torch.set_num_threads(1)
    started = time.monotonic()
    raw = read_checkpoint(path)
    data = raw['tracks'][track]
    result = {'iteration': raw['iteration'], 'models': {}}
    for kind, constructor, key in (('advantage', AdvantageNetwork, 'advantage_weights'),
                                   ('strategy', AveragePolicyNetwork, 'average_weights')):
        memory = unpack_memory(data[f'{kind}_memory'])
        samples = memory.samples
        rng = random.Random(data['seed'] + (0 if kind == 'advantage' else 10))
        rng.shuffle(samples)
        training = samples[round(len(samples)*.2):]
        weights = np.asarray([s.weight*s.iteration for s in training])
        normalization = float(weights.mean())
        encoding = EncodedReplay(training, raw['config']['feature_schema_version'])
        model = constructor(raw['config']['feature_schema_version'])
        model.load_state_dict(data[key]); model.eval()
        rng.shuffle(training)
        rows = []
        for offset in range(0, len(training), raw['config']['batch_size']):
            batch = training[offset:offset+raw['config']['batch_size']]
            normalized = [s.weight*s.iteration/normalization for s in batch]
            model.zero_grad()
            loss = _loss_with_weights(model, batch, normalized, encoding.select(batch))
            loss.backward()
            norm = float(torch.sqrt(sum(p.grad.square().sum() for p in model.parameters() if p.grad is not None)))
            rows.append({'batch': offset//raw['config']['batch_size'], 'loss': float(loss.detach()), 'gradient_norm': norm,
                         'max_normalized_sample_weight': max(normalized)})
        values = np.asarray([row['gradient_norm'] for row in rows])
        w = weights/weights.sum()
        result['models'][kind] = {'batches': rows, 'gradient_norm_quantiles':
            dict(zip(('min','median','p90','p99','max'), np.quantile(values, (0,.5,.9,.99,1)).tolist())),
            'largest_training_weight_share': float(w.max()), 'training_ess': float(1/(w*w).sum())}
        del memory, samples, encoding
        print(track, kind, 'gradients',result['models'][kind]['gradient_norm_quantiles'],flush=True)
    result['wall_seconds'] = time.monotonic()-started
    write_json(output/f'{track}-gradients.json',result)
    return result


def clipped_fit_study(path: Path, track: str, output: Path) -> dict:
    """Isolate norm clipping at the same split, ordering and 800-update budget."""
    torch.set_num_threads(1)
    raw = read_checkpoint(path)
    data = raw['tracks'][track]
    result = {'iteration': raw['iteration'], 'clip_norm': 1.0, 'updates': 800, 'models': {}}
    count = {'hu': 2, '3max': 3}[track]
    for kind, constructor, key in (('advantage', AdvantageNetwork, 'advantage_weights'),
                                   ('strategy', AveragePolicyNetwork, 'average_weights')):
        memory = unpack_memory(data[f'{kind}_memory'])
        rows = memory.samples
        rng = random.Random(data['seed']+(0 if kind == 'advantage' else 10))
        rng.shuffle(rows)
        heldout, training = rows[:round(len(rows)*.2)], rows[round(len(rows)*.2):]
        scale = sum(s.weight*s.iteration for s in training)/len(training)
        encoding = EncodedReplay(training, raw['config']['feature_schema_version'])
        model = constructor(raw['config']['feature_schema_version'])
        model.load_state_dict(data[key]);model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=raw['config']['learning_rate'])
        steps = 0
        started = time.monotonic()
        while steps < 800:
            rng.shuffle(training)
            for offset in range(0,len(training),raw['config']['batch_size']):
                batch = training[offset:offset+raw['config']['batch_size']]
                optimizer.zero_grad()
                loss = _loss_with_weights(model,batch,[s.weight*s.iteration/scale for s in batch],encoding.select(batch))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                optimizer.step()
                steps += 1
                if steps == 800:
                    break
        model.eval()
        opening = [s for s in heldout if opening_hand_class(s.state)]
        other = [s for s in heldout if not opening_hand_class(s.state)]
        policy = AdvantagePolicy(model) if kind == 'advantage' else model
        result['models'][kind] = {'wall_seconds':time.monotonic()-started,
            'opening_loss_original_weights':evaluate_loss(model,opening,128),
            'other_loss_original_weights':evaluate_loss(model,other,128),
            'opening_matrix':opening_matrix(policy,count,75/count)}
        torch.save(model.state_dict(),output/f'{track}-{kind}-clip1.pt')
        print(track,kind,'clip1 opening loss',result['models'][kind]['opening_loss_original_weights'],flush=True)
        write_json(output/f'{track}-clipped-fit.json',result)
        del memory, rows, encoding
    return result


def evaluate_refit(output: Path, track: str, samples: int = 1024) -> dict:
    """Paired root games against unchanged B opponents, with a new chance seed."""
    if not 2 <= samples <= 2048:
        raise ValueError('Refit comparison requires 2..2048 paired deals')
    torch.set_num_threads(1)
    metadata = torch.load(output/f'{track}-models.pt',map_location='cpu',weights_only=True)
    count = {'hu':2,'3max':3}[track]
    original = AveragePolicyNetwork(metadata['feature_schema_version'])
    original.load_state_dict(metadata['average_weights']);original.eval()
    candidate = AveragePolicyNetwork(metadata['feature_schema_version'])
    candidate.load_state_dict(torch.load(output/f'{track}-strategy-warm_opening_mass20.pt',map_location='cpu',weights_only=True));candidate.eval()
    root = HandState.start({i:75/count for i in range(count)},0,random.Random(49101))
    hero = root.current_player
    result = {'samples_per_holding':samples,'seed':49101,'opponents':'unchanged checkpoint B',
              'candidate':'warm_opening_mass20','holdings':{}}
    started = time.monotonic()
    for name, holding in (('AA',(48,49)),('22',(0,1)),('72o',(20,1))):
        rows = []
        for sample in range(samples):
            rng = random.Random(49101+sample*1009)
            deck = [c for c in range(52) if c not in holding];rng.shuffle(deck)
            hands = {hero:holding}
            for seat in root.players:
                if seat != hero:
                    hands[seat] = (deck.pop(),deck.pop())
            world = instantiate(root,hands,rng)
            values = [play(world,hero,policy,original,49101+sample*1009,EvaluationBudget(),lambda:None)[0]
                      for policy in (original,candidate)]
            rows.append(values)
        result['holdings'][name] = {'original':statistics([r[0] for r in rows]),
            'candidate':statistics([r[1] for r in rows]),
            'paired_gain':statistics([r[1]-r[0] for r in rows])}
        np.save(output/f'{track}-refit-{name}-paired-outcomes.npy',np.asarray(rows))
        print(track,name,'candidate gain',result['holdings'][name]['paired_gain'],flush=True)
    result['wall_seconds'] = time.monotonic()-started
    write_json(output/f'{track}-refit-evaluation.json',result)
    return result
