"""Frozen HU replay ablations with grouped validation and independent strategy tests."""
from __future__ import annotations

import cProfile
import gzip
import hashlib
import json
from pathlib import Path
import pstats
import resource
import time

import numpy as np
import torch

from ml.model import AdvantageNetwork, AveragePolicyNetwork, encode_batch
from ml.train import balance_decisions, fit
from training.checkpoint import atomic_bytes, read_checkpoint, unpack_memory
from training.config import config_hash
from training.independent_audit import (NetworkPolicy, action_values, evaluate_cases,
                                        evaluate_games, make_suite)
from hybrid.policy_source import UniformLegalPolicy
from training.metrics import decision_context, opening_hand_class
from training.poker_evaluation import statistics

BUDGETS = (256, 512, 1024, 2048)
SEEDS = (0, 1, 2)
SPLIT_SEED = 20261024
SOURCE_SHA256 = '30fcacd18122eab20f7e61cb436273ec64a1d072246e6f6813a06b547ea38171'


def write_json(path: Path, value: dict) -> None:
    atomic_bytes(path, (json.dumps(value, indent=2, allow_nan=False) + '\n').encode())


def heldout_group(group: str) -> bool:
    if not group:
        raise ValueError('Independent replay validation requires root_group provenance')
    return int(hashlib.sha256(f'{SPLIT_SEED}:{group}'.encode()).hexdigest()[:8], 16) % 5 == 0


def predict(model: AdvantageNetwork | AveragePolicyNetwork, samples: list) -> np.ndarray:
    with torch.no_grad():
        return np.concatenate([model(encode_batch([s.state for s in samples[i:i + 256]])).numpy()
                               for i in range(0, len(samples), 256)])


def replay_metadata(samples: list, output: Path, kind: str) -> dict:
    heldout = np.asarray([heldout_group(s.root_group) for s in samples])
    weights = np.asarray([s.weight * s.iteration for s in samples], dtype=float)
    strata = np.asarray(['first_open' if opening_hand_class(s.state) else f'street_{s.state.street}' for s in samples])
    contexts = np.asarray([f'{s.state.street}/{decision_context(s.state)}' for s in samples])
    versions = np.asarray([s.model_version for s in samples])
    balanced = np.zeros(len(samples))
    for selected in (~heldout, heldout):
        for stratum in sorted(set(strata)):
            indices = selected & (strata == stratum)
            if not indices.any():
                raise ValueError(f'{kind} partition lacks {stratum}')
            balanced[indices] = weights[indices] / weights[indices].sum()
    data = {'heldout': heldout, 'weights': weights, 'balanced_weights': balanced,
            'strata': strata, 'contexts': contexts, 'versions': versions,
            'targets': np.asarray([s.target for s in samples]),
            'masks': np.asarray([s.state.legal_mask for s in samples]),
            'root_groups': np.asarray([s.root_group for s in samples]),
            'trajectories': np.asarray([s.trajectory_id for s in samples])}
    np.savez_compressed(output / f'{kind}-replay.npz', **data)
    coverage = {}
    for context in sorted(set(contexts)):
        selected = contexts == context
        w = balanced[selected]
        coverage[context] = {'records': int(selected.sum()), 'heldout_records': int((selected & heldout).sum()),
                             'root_groups': len(set(data['root_groups'][selected])),
                             'trajectories': len(set(data['trajectories'][selected])),
                             'balanced_replay_ess': float(w.sum() ** 2 / np.square(w).sum()),
                             'source_version_min': int(versions[selected].min()),
                             'source_version_max': int(versions[selected].max())}
    write_json(output / f'{kind}-coverage.json', coverage)
    return data


def replay_metrics(predictions: np.ndarray, data: dict, kind: str, artifact: Path) -> dict:
    targets, masks = data['targets'], data['masks']
    if kind == 'advantage':
        losses = (((predictions - targets) ** 2) * masks).sum(1) / masks.sum(1)
        baseline = (targets ** 2 * masks).sum(1) / masks.sum(1)
        tv = np.full(len(losses), np.nan)
    else:
        losses = (targets * (np.log(np.clip(targets, 1e-12, None)) - np.log(np.clip(predictions, 1e-12, None)))).sum(1)
        baseline = (targets * np.log(np.clip(targets, 1e-12, None) * masks.sum(1)[:, None])).sum(1)
        tv = np.abs(predictions - targets).sum(1) / 2
    if not np.isfinite(predictions).all() or not np.isfinite(losses).all() or not np.isfinite(baseline).all():
        raise ValueError('Nonfinite replay predictions or evaluation losses')
    np.savez_compressed(artifact, predictions=predictions, losses=losses, tv=tv)
    groups = {'train': ~data['heldout'], 'heldout': data['heldout']}
    groups.update({f'heldout/{c}': data['heldout'] & (data['contexts'] == c) for c in set(data['contexts'])})
    for lo, hi in ((0, 363), (364, 999), (1000, 1299), (1300, 1599)):
        groups[f'heldout/teacher_versions_{lo}_{hi}'] = data['heldout'] & (data['versions'] >= lo) & (data['versions'] <= hi)
    result = {}
    for name, selected in sorted(groups.items()):
        if not selected.any():
            result[name] = {'records': 0}
            continue
        w = data['balanced_weights'][selected]
        w = w / w.sum()
        raw_w = data['weights'][selected] / data['weights'][selected].sum()
        result[name] = {'records': int(selected.sum()), 'balanced_loss': float((w * losses[selected]).sum()),
                        'historical_weighted_loss': float((raw_w * losses[selected]).sum()),
                        'baseline_balanced_loss': float((w * baseline[selected]).sum()),
                        'balanced_tv': None if kind == 'advantage' else float((w * tv[selected]).sum())}
    return result


def run_fits(source: Path, output: Path) -> dict:
    """All four caps share initialization and minibatch prefixes within a seed."""
    if (output / 'fits.json').exists():
        raise FileExistsError(output / 'fits.json')
    checksum = hashlib.sha256()
    with source.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(block)
    if checksum.hexdigest() != SOURCE_SHA256:
        raise ValueError('Audit requires the frozen HU 1600 input checksum')
    torch.set_num_threads(1)
    raw = read_checkpoint(source)
    if raw['iteration'] != 1600 or set(raw['tracks']) != {'hu'} or config_hash(raw['config']) != raw['config_hash']:
        raise ValueError('Unexpected source checkpoint identity or configuration')
    track = raw['tracks']['hu']
    output.mkdir(parents=True, exist_ok=True)
    report = {'source_sha256': checksum.hexdigest(), 'source_run_id': raw['metadata']['run_id'],
              'split_seed': SPLIT_SEED, 'budgets': BUDGETS, 'seed_replicates': SEEDS,
              'partition': 'Public root-group hash, 80/20 expected; exact whole replay retained',
              'objective': 'Equal mass per opening/street within each partition, historical weights within strata',
              'models': {}, 'partitions': {}}
    # Warm lazy optimizer imports before timing comparable fit calls.
    torch.optim.Adam(AdvantageNetwork().parameters())
    for kind, constructor, key in (('advantage', AdvantageNetwork, 'advantage_weights'),
                                    ('strategy', AveragePolicyNetwork, 'average_weights')):
        samples = unpack_memory(track[kind + '_memory']).samples
        data = replay_metadata(samples, output, kind)
        train = balance_decisions([s for s, h in zip(samples, data['heldout']) if not h])
        validation = balance_decisions([s for s, h in zip(samples, data['heldout']) if h])
        pool = train + validation
        train_groups = {s.root_group for s in train}
        validation_groups = {s.root_group for s in validation}
        if train_groups & validation_groups:
            raise ValueError('Public roots leak across audit partitions')
        report['partitions'][kind] = {'train_records': len(train), 'heldout_records': len(validation),
                                       'train_groups': len(train_groups), 'heldout_groups': len(validation_groups),
                                       'overlapping_groups': 0}
        model = constructor()
        model.load_state_dict(track[key])
        model.eval()
        torch.save({'kind': kind, 'weights': model.state_dict(), 'source_sha256': SOURCE_SHA256}, output / f'source-{kind}.pt')
        report['models'][f'source-{kind}'] = {
            'kind': kind, 'source_model': True,
            'validation_scope': 'Production model has seen these public groups; diagnostic only',
            'replay': replay_metrics(predict(model, samples), data, kind, output / f'source-{kind}-replay.npz')}
        for replicate in SEEDS:
            initialization = track['seed'] + 1600 * 101 + replicate * 1000003 + (10 if kind == 'strategy' else 0)
            sampling = track['seed'] + replicate * 1000003 + (10 if kind == 'strategy' else 0)
            for budget in BUDGETS:
                name = f'{kind}-seed{replicate}-updates{budget}'
                torch.manual_seed(initialization)
                model = constructor()
                wall, cpu = time.monotonic(), time.process_time()
                profiler = cProfile.Profile()
                metrics = profiler.runcall(fit, model, pool, 32, raw['config']['batch_size'], sampling,
                                           learning_rate=raw['config']['learning_rate'], max_updates=budget,
                                           cache_encoding=True, sampling='weighted_replacement',
                                           explicit_partitions=(train, validation))
                elapsed, cpu_elapsed = time.monotonic() - wall, time.process_time() - cpu
                if metrics['updates_completed'] != budget:
                    raise ValueError(f'Fit did not reach requested cap: {name}')
                profiler.dump_stats(str(output / f'{name}.prof'))
                torch.save({'kind': kind, 'weights': model.state_dict(), 'source_sha256': SOURCE_SHA256,
                            'initialization_seed': initialization, 'sampling_seed': sampling, 'updates': budget}, output / f'{name}.pt')
                predictions = predict(model, samples)
                result = {'kind': kind, 'replicate': replicate, 'updates': budget,
                          'fit_wall_seconds': elapsed, 'fit_cpu_seconds': cpu_elapsed,
                          'initialization_seed': initialization, 'sampling_seed': sampling,
                          'fit_metrics': metrics,
                          'replay': replay_metrics(predictions, data, kind, output / f'{name}-replay.npz')}
                report['models'][name] = result
                write_json(output / 'fits.json', report)
                print(f'{name}: {elapsed:.2f}s, heldout={metrics["heldout_loss"]:.6f}', flush=True)
    report['peak_rss_bytes_macos'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report['complete'] = True
    write_json(output / 'fits.json', report)
    return report


def load_model(path: Path) -> AdvantageNetwork | AveragePolicyNetwork:
    payload = torch.load(path, weights_only=True, map_location='cpu')
    if payload['source_sha256'] != SOURCE_SHA256:
        raise ValueError(f'Foreign audit model: {path}')
    if payload['kind'] not in ('advantage', 'strategy'):
        raise ValueError(f'Invalid audit network kind: {path}')
    model = AdvantageNetwork() if payload['kind'] == 'advantage' else AveragePolicyNetwork()
    model.load_state_dict(payload['weights'])
    return model.eval()


def run_strategic_evaluation(output: Path, *, groups: int = 256, worlds: int = 256) -> dict:
    fits = json.loads((output / 'fits.json').read_text())
    if not fits.get('complete') or (output / 'strategic.json').exists():
        raise ValueError('Requires complete fits and an unused strategic result target')
    torch.set_num_threads(1)
    start = time.monotonic()
    suite = make_suite(groups, worlds)
    suite.save(output / 'suite.json.gz')
    source = load_model(output / 'source-advantage.pt')
    q = action_values(suite, {'checkpoint_A': NetworkPolicy(source), 'uniform': UniformLegalPolicy()}, output)
    report = {'groups': groups, 'cases': len(suite.cases), 'worlds_per_case': worlds,
              'exclusion_count': suite.exclusion_count, 'models': {},
              'q_scope': 'Fixed checkpoint-A or uniform continuations; forced histories with uniform compatible hidden worlds',
              'uncertainty': 'Descriptive paired group intervals; fixed case panel is not a random sample of all poker states'}
    for name in fits['models']:
        model = load_model(output / f'{name}.pt')
        wall, cpu = time.monotonic(), time.process_time()
        games = evaluate_games(NetworkPolicy(model), suite)
        cases = evaluate_cases(model, suite, q)
        with gzip.open(output / f'{name}-games.json.gz', 'wt') as stream:
            json.dump(games, stream, allow_nan=False)
        write_json(output / f'{name}-cases.json', cases)
        report['models'][name] = {'games': {k: v for k, v in games.items() if k != 'rows'},
                                  'cases': {c: {k: v for k, v in result.items() if k != 'cases'} for c, result in cases.items()},
                                  'evaluation_wall_seconds': time.monotonic() - wall,
                                  'evaluation_cpu_seconds': time.process_time() - cpu}
        write_json(output / 'strategic.json', report)
        print(f'Strategic {name}: {games["bb_per_100"]["mean"]:.3f} BB/100', flush=True)
    report['total_wall_seconds'] = time.monotonic() - start
    report['complete'] = True
    write_json(output / 'strategic.json', report)
    return report


def conditional_panel_statistics(values: np.ndarray) -> dict:
    """MC uncertainty for an equally weighted fixed case panel, not state sampling."""
    if values.ndim != 2 or values.shape[1] < 2 or not np.isfinite(values).all():
        raise ValueError('Panel statistics require a finite case-by-world array')
    mean = float(values.mean())
    se = float(np.sqrt((values.var(axis=1, ddof=1) / values.shape[1]).sum()) / values.shape[0])
    return {'mean': mean, 'se': se, 'interval': [mean - 1.96 * se, mean + 1.96 * se],
            'fixed_cases': values.shape[0], 'worlds_per_case': values.shape[1]}


def grouped_loss_difference(output: Path, kind: str, budget: int, baseline_budget: int = 256) -> dict:
    data = np.load(output / f'{kind}-replay.npz')
    differences = []
    for replicate in SEEDS:
        high = np.load(output / f'{kind}-seed{replicate}-updates{budget}-replay.npz')['losses']
        low = np.load(output / f'{kind}-seed{replicate}-updates{baseline_budget}-replay.npz')['losses']
        differences.append(high - low)
    selected = data['heldout']
    delta = np.mean(differences, axis=0)[selected]
    weights = data['balanced_weights'][selected]
    groups, inverse = np.unique(data['root_groups'][selected], return_inverse=True)
    numerator = np.bincount(inverse, weights=delta * weights)
    denominator = np.bincount(inverse, weights=weights)
    rng = np.random.default_rng(682103)
    draws = rng.integers(0, len(groups), size=(2000, len(groups)))
    bootstrap = numerator[draws].sum(1) / denominator[draws].sum(1)
    return {'mean': float(numerator.sum() / denominator.sum()),
            'root_cluster_bootstrap_interval': np.quantile(bootstrap, [.025, .975]).tolist(),
            'heldout_root_groups': len(groups), 'bootstrap_replicates': 2000,
            'scope': 'Seeds averaged first; root-group ratio bootstrap, descriptive'}


def historical_projection(targets: np.ndarray, weights: np.ndarray, inverse: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Empirical weighted historical teacher mean and its within-input KL scatter."""
    if not np.isfinite(targets).all() or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError('Historical teacher targets and weights must be finite, with positive weights')
    group_weights = np.bincount(inverse, weights=weights)
    totals = np.zeros((len(group_weights), targets.shape[1]))
    np.add.at(totals, inverse, targets * weights[:, None])
    means = totals / group_weights[:, None]
    divergence = (targets * (np.log(np.clip(targets, 1e-12, None))
                             - np.log(np.clip(means[inverse], 1e-12, None)))).sum(1)
    return means, group_weights, float((weights * divergence).sum() / weights.sum())


def historical_strategy_fidelity(source: Path, output: Path) -> dict:
    raw = read_checkpoint(source)
    if raw['iteration'] != 1600 or raw['metadata']['run_id'] != json.loads((output / 'fits.json').read_text())['source_run_id']:
        raise ValueError('Historical fidelity source does not match the fitted audit')
    samples = unpack_memory(raw['tracks']['hu']['strategy_memory']).samples
    data = np.load(output / 'strategy-replay.npz')
    if not np.array_equal(np.asarray([s.target for s in samples]), data['targets']):
        raise ValueError('Replay record ordering changed')
    selected = data['heldout']
    identities = []
    for sample, keep in zip(samples, selected):
        if keep:
            state = sample.state
            header = repr((state.feature_version, state.cards, state.street, state.legal_mask)).encode()
            identities.append(hashlib.sha256(header + state.numeric_data + state.history_data).hexdigest())
    unique, inverse, counts = np.unique(identities, return_inverse=True, return_counts=True)
    weights = data['balanced_weights'][selected]
    means, group_weights, scatter = historical_projection(data['targets'][selected], weights, inverse)
    np.savez_compressed(output / 'historical-teacher-projection.npz',
                        identities=unique, record_groups=inverse, historical_probabilities=means,
                        group_weights=group_weights, counts=counts, selected_indices=np.flatnonzero(selected))
    report = {'heldout_records': int(selected.sum()), 'exact_observable_inputs': len(unique),
              'singleton_inputs': int((counts == 1).sum()), 'max_records_per_input': int(counts.max()),
              'empirical_historical_scatter_kl': scatter,
              'scope': 'Weighted mean of recorded teachers at identical compact observable inputs; singleton groups do not recover an unseen population mean',
              'models': {}}
    for name, record in json.loads((output / 'fits.json').read_text())['models'].items():
        if record['kind'] != 'strategy':
            continue
        predictions = np.load(output / f'{name}-replay.npz')['predictions'][selected]
        pooled = np.zeros_like(means)
        np.add.at(pooled, inverse, predictions * weights[:, None])
        pooled /= group_weights[:, None]
        maximum_gap = float(np.abs(predictions - pooled[inverse]).max())
        if maximum_gap > 1e-6:
            raise ValueError(f'Identical observable inputs produced inconsistent B predictions: {name}/{maximum_gap}')
        kl = (means * (np.log(np.clip(means, 1e-12, None)) - np.log(np.clip(pooled, 1e-12, None)))).sum(1)
        loss = float((group_weights * kl).sum() / group_weights.sum())
        discrepancy = abs(loss + scatter - record['replay']['heldout']['balanced_loss'])
        if discrepancy > 1e-6:
            raise ValueError(f'Historical KL decomposition mismatch: {name}/{discrepancy}')
        report['models'][name] = {'historical_mean_kl': loss,
                                 'historical_mean_tv': float((group_weights * np.abs(means - pooled).sum(1) / 2).sum() / group_weights.sum()),
                                 'decomposition_error': discrepancy, 'identical_input_max_gap': maximum_gap}
    write_json(output / 'historical-fidelity.json', report)
    return report


def summarize(output: Path) -> dict:
    fits = json.loads((output / 'fits.json').read_text())
    strategic = json.loads((output / 'strategic.json').read_text())
    if not fits.get('complete') or not strategic.get('complete'):
        raise ValueError('Cannot summarize incomplete audit experiments')
    report = {'source_sha256': SOURCE_SHA256, 'budgets': {}, 'costs': {},
              'scope': 'Exploratory paired comparisons; no multiple-testing correction or full-game exploitability claim'}
    case_files = {name: json.loads((output / f'{name}-cases.json').read_text()) for name in fits['models']}
    game_files = {}
    for name in fits['models']:
        with gzip.open(output / f'{name}-games.json.gz', 'rt') as stream:
            game_files[name] = json.load(stream)
    for kind in ('advantage', 'strategy'):
        rows = []
        for budget in BUDGETS:
            names = [f'{kind}-seed{s}-updates{budget}' for s in SEEDS]
            base_names = [f'{kind}-seed{s}-updates256' for s in SEEDS]
            previous_budget = BUDGETS[max(0, BUDGETS.index(budget) - 1)]
            previous_names = [f'{kind}-seed{s}-updates{previous_budget}' for s in SEEDS]
            games = np.asarray([strategic['models'][n]['games']['group_bb'] for n in names])
            bases = np.asarray([strategic['models'][n]['games']['group_bb'] for n in base_names])
            previous_games = np.asarray([strategic['models'][n]['games']['group_bb'] for n in previous_names])
            row = {'updates': budget,
                   'mean_heldout_loss': float(np.mean([fits['models'][n]['replay']['heldout']['balanced_loss'] for n in names])),
                   'mean_heldout_tv': None if kind == 'advantage' else float(np.mean([fits['models'][n]['replay']['heldout']['balanced_tv'] for n in names])),
                   'loss_difference_vs256': grouped_loss_difference(output, kind, budget),
                   'loss_difference_vsprevious': grouped_loss_difference(output, kind, budget, previous_budget),
                   'fit_cpu_seconds': float(np.mean([fits['models'][n]['fit_cpu_seconds'] for n in names])),
                   'fit_wall_seconds': float(np.mean([fits['models'][n]['fit_wall_seconds'] for n in names])),
                   'gain_bb100': statistics((games.mean(0) * 100).tolist()),
                   'paired_gain_vs256_bb100': statistics(((games - bases).mean(0) * 100).tolist()),
                   'previous_budget': previous_budget,
                   'paired_gain_vsprevious_bb100': statistics(((games - previous_games).mean(0) * 100).tolist()),
                   'per_seed_paired_gain_bb100': [statistics((d * 100).tolist()) for d in games - bases],
                   'segments': {}, 'mean_decision_shares': {},
                   'cases': {}}
            row['incremental_fit_cpu_seconds'] = row['fit_cpu_seconds'] - float(np.mean([
                fits['models'][n]['fit_cpu_seconds'] for n in previous_names]))
            for field in ('profile', 'region'):
                for segment in sorted({r[field] for r in game_files[names[0]]['rows']}):
                    candidate_groups, baseline_groups = [], []
                    for nn, destination in ((names, candidate_groups), (base_names, baseline_groups)):
                        for name in nn:
                            records = [r for r in game_files[name]['rows'] if r[field] == segment]
                            destination.append([np.mean([r['bb'] for r in records if r['group'] == group])
                                                for group in range(strategic['groups'])])
                    delta = (np.asarray(candidate_groups) - np.asarray(baseline_groups)).mean(0)
                    row['segments'][f'{field}/{segment}'] = statistics((delta * 100).tolist())
            contexts = sorted({c for name in names for c in strategic['models'][name]['games']['decision_visits']})
            for context in contexts:
                row['mean_decision_shares'][context] = float(np.mean([
                    strategic['models'][name]['games']['decision_visits'].get(context, 0)
                    / sum(strategic['models'][name]['games']['decision_visits'].values()) for name in names]))
            for continuation in ('checkpoint_A', 'uniform'):
                policies = np.asarray([[c['world_policy_ev'] for c in case_files[n][continuation]['cases']] for n in names])
                reference = np.asarray([[c['world_policy_ev'] for c in case_files[n][continuation]['cases']] for n in base_names])
                previous = np.asarray([[c['world_policy_ev'] for c in case_files[n][continuation]['cases']] for n in previous_names])
                delta = (policies - reference).mean(0)
                all_cases = [c for n in names for c in case_files[n][continuation]['cases']]
                reliable = sum(c['reliable_action_pairs'] for c in all_cases)
                row['cases'][continuation] = {
                    'mean_policy_ev': float(policies.mean()),
                    'paired_ev_vs256': conditional_panel_statistics(delta),
                    'paired_ev_vsprevious': conditional_panel_statistics((policies - previous).mean(0)),
                    'mean_crossfit_deviation_gain': float(np.mean([c['crossfit_deviation_gain']['mean'] for c in all_cases])),
                    'reliable_pairs': reliable,
                    'rank_accuracy_on_reliable_pairs': sum(c['correctly_ranked_pairs'] for c in all_cases) / reliable if reliable else None,
                    'branches': {}}
                ids = [c['id'] for c in case_files[names[0]][continuation]['cases']]
                for branch in sorted({i.split('/')[0] for i in ids}):
                    selected = [i for i, name in enumerate(ids) if name.split('/')[0] == branch]
                    row['cases'][continuation]['branches'][branch] = conditional_panel_statistics(delta[selected])
            rows.append(row)
        report['budgets'][kind] = rows
    costs = []
    for name, model in fits['models'].items():
        if model.get('source_model'):
            continue
        stats = pstats.Stats(str(output / f'{name}.prof')).stats
        encoding = sum(value[3] for (file, line, function), value in stats.items()
                       if file.endswith('/ml/train.py') and function == '__init__')
        validation = sum(value[3] for (file, line, function), value in stats.items()
                         if file.endswith('/ml/train.py') and function == 'measure')
        costs.append({'model': name, 'encoding_seconds': encoding, 'validation_seconds': validation,
                      'fit_cpu_seconds': model['fit_cpu_seconds'], 'fit_wall_seconds': model['fit_wall_seconds']})
    report['costs'] = {'fits': costs, 'total_fit_cpu_seconds': sum(c['fit_cpu_seconds'] for c in costs),
                       'total_fit_wall_seconds': sum(c['fit_wall_seconds'] for c in costs),
                       'strategic_wall_seconds': strategic['total_wall_seconds'],
                       'peak_fit_rss_bytes_macos': fits['peak_rss_bytes_macos']}
    report['partitions'] = fits['partitions']
    report['replay_coverage'] = {}
    for kind in ('advantage', 'strategy'):
        data = np.load(output / f'{kind}-replay.npz')
        rows = {}
        for context in sorted(set(data['contexts'])):
            selected = data['contexts'] == context
            training = selected & ~data['heldout']
            heldout = selected & data['heldout']
            w = data['balanced_weights'][training]
            rows[context] = {'records': int(selected.sum()), 'heldout_records': int(heldout.sum()),
                             'training_records': int(training.sum()),
                             'training_objective_share': float(w.sum() / data['balanced_weights'][~data['heldout']].sum()),
                             'training_ess': float(w.sum() ** 2 / np.square(w).sum()) if len(w) else 0.,
                             'root_groups': len(set(data['root_groups'][selected])),
                             'trajectories': len(set(data['trajectories'][selected]))}
        report['replay_coverage'][kind] = rows
    write_json(output / 'summary.json', report)
    return report
