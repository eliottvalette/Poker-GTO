"""Two-track runner and durable atomic resume contracts."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from training.checkpoint import read_checkpoint, write_checkpoint
from training.config import load_config
from training.runner import TrainingRunner


def tiny_config(directory):
    config = load_config('configs/deep_cfr_pilot.json')
    config.update(output_dir=str(directory), workers=1, outer_iterations=2, advantage_epochs=1, average_epochs=1,
                  checkpoint_every=1, evaluation_every=2, max_nodes=20000,
                  strategy_capacity=1000)
    for name in ('3max', 'hu'):
        config[name]['traversals_per_player'] = 2
        config[name]['advantage_capacity'] = 1000
    return config


def deterministic_metric(row):
    result = deepcopy(row)
    result.pop('wall_seconds', None)
    result.pop('checkpoint_path', None)
    for track in result['tracks'].values():
        for key in ('worker_seconds', 'worker_cpu_seconds', 'worker_peak_rss_bytes', 'generation_seconds', 'root_seconds', 'fit_seconds', 'fit_component_seconds'):
            track.pop(key, None)
    return result


class TrainingRunnerTests(unittest.TestCase):
    def test_execution_only_resume_is_explicit_and_audited(self):
        with tempfile.TemporaryDirectory() as directory:
            config=tiny_config(directory)
            runner=TrainingRunner(config);runner.run_iteration()
            path=Path(directory)/'checkpoints/iteration_000001.pt'
            requested=deepcopy(config)
            requested.update(workers=2,trainer_threads=2,generation_batch_size=1)
            with self.assertRaisesRegex(ValueError,'hash mismatch'):
                TrainingRunner.load_checkpoint(path,requested)
            resumed=TrainingRunner.load_checkpoint(path,requested,allow_execution_changes=True)
            self.assertEqual(resumed.config,requested)
            self.assertEqual(resumed.metadata['resume_execution_changes'][0]['after'],
                             {'workers':2,'trainer_threads':2,'generation_batch_size':1})
            for name,solver in runner.solvers.items():
                self.assertEqual(solver.strategy_memory.samples,resumed.solvers[name].strategy_memory.samples)
                self.assertEqual(solver.rng.getstate(),resumed.solvers[name].rng.getstate())
                for k,v in solver.advantage_model.state_dict().items():
                    self.assertTrue(torch.equal(v,resumed.solvers[name].advantage_model.state_dict()[k]))
            for key,value in (('batch_size',128),('advantage_epochs',7),('seed',100)):
                invalid=deepcopy(requested);invalid[key]=value
                with self.assertRaisesRegex(ValueError,'hash mismatch'):
                    TrainingRunner.load_checkpoint(path,invalid,allow_execution_changes=True)

    def test_generation_chunks_preserve_replay_and_models(self):
        with tempfile.TemporaryDirectory() as directory:
            config = tiny_config(Path(directory)/'whole')
            other = deepcopy(config)
            other.update(output_dir=str(Path(directory)/'chunks'), generation_batch_size=1)
            a,b = TrainingRunner(config),TrainingRunner(other)
            a.run_iteration();b.run_iteration()
            for name, solver in a.solvers.items():
                self.assertEqual(solver.advantage_memory.samples,b.solvers[name].advantage_memory.samples)
                self.assertEqual(solver.strategy_memory.samples,b.solvers[name].strategy_memory.samples)
                for key,value in solver.average_model.state_dict().items():
                    self.assertTrue(torch.equal(value,b.solvers[name].average_model.state_dict()[key]))

    def test_independent_average_schedule_and_warm_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            config = tiny_config(directory)
            config['fit_schedule'] = {'average_every': 3, 'advantage_max_updates': 2,
                'average_max_updates': 2, 'advantage_initialization': 'warm',
                'average_initialization': 'warm', 'cache_encoding': True}
            runner = TrainingRunner(config)
            runner.run_iteration()
            original = {n: deepcopy(s.average_model.state_dict()) for n, s in runner.solvers.items()}
            row = runner.run_iteration()
            for name, solver in runner.solvers.items():
                self.assertTrue(row['tracks'][name]['average_policy']['skipped'])
                self.assertEqual(row['tracks'][name]['average_model_iteration'], 1)
                for k,v in original[name].items():
                    self.assertTrue(torch.equal(v, solver.average_model.state_dict()[k]))
            resumed = TrainingRunner.load_checkpoint(Path(directory) / 'checkpoints/iteration_000002.pt')
            expected, actual = runner.run_iteration(), resumed.run_iteration()
            self.assertEqual(deterministic_metric(expected), deterministic_metric(actual))
            for name, solver in runner.solvers.items():
                self.assertEqual(actual['tracks'][name]['average_model_iteration'], 3)
                self.assertLessEqual(actual['tracks'][name]['advantage']['updates_completed'], 2)
                for key, value in solver.average_model.state_dict().items():
                    self.assertTrue(torch.equal(value, resumed.solvers[name].average_model.state_dict()[key]))

    def test_protected_replay_runner_resume_preserves_weighted_fit(self):
        from ml.stratified_memory import ProtectedReplay
        with tempfile.TemporaryDirectory() as directory:
            config = tiny_config(directory)
            config['replay_opening_fraction'] = 0.25
            config['strategy_capacity'] = 20
            for track in ('hu', '3max'):
                config[track]['advantage_capacity'] = 20
                config[track]['root_sampling']['hole_card_sampling'] = 'stratified_recorded_opening'
            runner = TrainingRunner(config)
            runner.run_iteration()
            resumed = TrainingRunner.load_checkpoint(Path(directory) / 'checkpoints/iteration_000001.pt')
            for name, solver in runner.solvers.items():
                for kind in ('advantage_memory', 'strategy_memory'):
                    memory = getattr(solver, kind)
                    self.assertIsInstance(memory, ProtectedReplay)
                    self.assertEqual(memory.samples, getattr(resumed.solvers[name], kind).samples)
                    self.assertAlmostEqual(sum(s.weight for s in memory.samples),
                                           sum(s.weight for s in getattr(resumed.solvers[name], kind).samples))
            first, second = runner.run_iteration(), resumed.run_iteration()
            self.assertEqual(deterministic_metric(first), deterministic_metric(second))
            for name, solver in runner.solvers.items():
                self.assertEqual(solver.strategy_memory.samples, resumed.solvers[name].strategy_memory.samples)
                for key, weight in solver.average_model.state_dict().items():
                    self.assertTrue(torch.equal(weight, resumed.solvers[name].average_model.state_dict()[key]))
            config['traversal_mode'] = 'outcome_sampling'
            config['outcome_epsilon'] = 0.6
            with self.assertRaisesRegex(ValueError, 'external_sampling'):
                TrainingRunner(config)

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_iteration_checkpoint_resume_and_scheduled_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            config = tiny_config(directory)
            for track in ('hu', '3max'):
                config[track]['root_sampling']['hole_card_sampling'] = 'stratified'
            config.update(advantage_epochs=3, average_epochs=1)
            runner = TrainingRunner(config)
            row = runner.run_iteration()
            self.assertEqual(runner.iteration, 1)
            self.assertEqual(set(row['tracks']), {'3max', 'hu'})
            self.assertTrue(all(t['fit_epochs'] == {'advantage': 3, 'average_policy': 1} for t in row['tracks'].values()))
            self.assertTrue(all('evaluation' not in t for t in row['tracks'].values()))
            path = Path(directory) / 'checkpoints/iteration_000001.pt'
            restored = TrainingRunner.load_checkpoint(path)
            for name, solver in runner.solvers.items():
                loaded = restored.solvers[name]
                self.assertEqual(solver.rng.getstate(), loaded.rng.getstate())
                self.assertEqual(solver.strategy_memory.rng.getstate(), loaded.strategy_memory.rng.getstate())
                self.assertEqual(solver.strategy_memory.samples, loaded.strategy_memory.samples)
                self.assertEqual(solver.strategy_memory.used_bytes, loaded.strategy_memory.used_bytes)
                self.assertEqual(runner.samplers[name].rng.getstate(), restored.samplers[name].rng.getstate())
                self.assertEqual(runner.samplers[name].card_state(), restored.samplers[name].card_state())
            a, b = runner.run_iteration(), restored.run_iteration()
            self.assertEqual(deterministic_metric(a), deterministic_metric(b))
            for name in runner.solvers:
                self.assertEqual(set(a['tracks'][name]['evaluation']), {'version', 'bounded_best_response', 'independent_advantage_loss', 'independent_advantage_loss_by_player', 'independent_average_policy_loss', 'scripted_opponents'})
                for key, weight in runner.solvers[name].average_model.state_dict().items():
                    self.assertTrue(torch.equal(weight, restored.solvers[name].average_model.state_dict()[key]))
            lines = (Path(directory) / 'metrics.jsonl').read_text().splitlines()
            self.assertEqual([json.loads(line)['iteration'] for line in lines], [1, 2])
            self.assertTrue((Path(directory) / 'hu/coverage.json').exists())
            outputs = runner.export()
            self.assertEqual(set(outputs), {'hu', '3max'})
            from ml.policy_router import AveragePolicyRouter
            from infoset import observe
            router = AveragePolicyRouter({3: outputs['3max'], 2: outputs['hu']})
            for name, count in (('3max', 3), ('hu', 2)):
                observation = observe(runner.probes[name][0][1])
                self.assertEqual(router.query(observation), router.policies[count].query(observation))
            with self.assertRaisesRegex(KeyError, 'unavailable'):
                AveragePolicyRouter({3: outputs['3max']}).query(observe(runner.probes['hu'][0][1]))

    def test_schema_config_and_corruption_fail_explicitly(self):
        with tempfile.TemporaryDirectory() as directory:
            config = tiny_config(directory)
            runner = TrainingRunner(config)
            path = Path(directory) / 'checkpoint.pt'
            runner.save_checkpoint(path)
            different = deepcopy(config)
            different['seed'] += 1
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                TrainingRunner.load_checkpoint(path, different)
            raw = read_checkpoint(path)
            raw['contract']['feature'] = 99
            write_checkpoint(path, raw)
            with self.assertRaisesRegex(ValueError, 'schema'):
                TrainingRunner.load_checkpoint(path)
            path.write_bytes(b'corrupt checkpoint')
            with self.assertRaisesRegex(ValueError, 'Invalid training checkpoint'):
                TrainingRunner.load_checkpoint(path)

    def test_explicit_budget_increase_preserves_trained_state_and_is_audited(self):
        with tempfile.TemporaryDirectory() as directory:
            config = tiny_config(directory)
            runner = TrainingRunner(config)
            runner.run_iteration()
            path = Path(directory) / 'checkpoint.pt'
            runner.save_checkpoint(path)
            original = path.read_bytes()
            requested = deepcopy(config)
            requested['max_nodes'] *= 2
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                TrainingRunner.load_checkpoint(path, requested)
            restored = TrainingRunner.load_checkpoint(path, requested, allow_budget_increase=True)
            self.assertEqual(restored.iteration, runner.iteration)
            self.assertEqual(restored.metrics, runner.metrics)
            self.assertEqual(restored.config, requested)
            self.assertEqual(path.read_bytes(), original)
            for name, solver in runner.solvers.items():
                current = restored.solvers[name]
                self.assertEqual(current.rng.getstate(), solver.rng.getstate())
                self.assertEqual(current.advantage_memory.samples, solver.advantage_memory.samples)
                self.assertEqual(current.strategy_memory.samples, solver.strategy_memory.samples)
                self.assertEqual(current.advantage_memory.rng.getstate(), solver.advantage_memory.rng.getstate())
                self.assertEqual(current.strategy_memory.rng.getstate(), solver.strategy_memory.rng.getstate())
                self.assertEqual(restored.samplers[name].rng.getstate(), runner.samplers[name].rng.getstate())
                for model in ('advantage_model', 'average_model'):
                    for key, weight in getattr(solver, model).state_dict().items():
                        self.assertTrue(torch.equal(weight, getattr(current, model).state_dict()[key]))
            self.assertEqual(restored.resume_budget_change['before'], {'max_nodes': config['max_nodes']})
            self.assertEqual(restored.resume_budget_change['after'], {'max_nodes': requested['max_nodes']})
            destination = Path(directory) / 'raised.pt'
            restored.save_checkpoint(destination)
            again = TrainingRunner.load_checkpoint(destination, requested)
            self.assertEqual(again.metadata['resume_budget_changes'], [restored.resume_budget_change])
            self.assertIsNone(again.resume_budget_change)
            for key, value in [('max_nodes', 1), ('max_depth', 1), ('seed', config['seed'] + 1), ('learning_rate', 0.001)]:
                invalid = deepcopy(requested)
                invalid[key] = value
                with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                    TrainingRunner.load_checkpoint(path, invalid, allow_budget_increase=True)

    def test_worker_failure_does_not_publish_any_track_or_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = TrainingRunner(tiny_config(directory))
            rngs = {name: sampler.rng.getstate() for name, sampler in runner.samplers.items()}
            with patch('scripts.parallel_cfr.collect_samples', side_effect=RuntimeError('Injected worker failure')):
                with self.assertRaisesRegex(RuntimeError, 'worker failure'):
                    runner.run_iteration()
            self.assertEqual(runner.iteration, 0)
            self.assertEqual(runner.metrics, [])
            self.assertTrue(all(solver.strategy_memory.seen == 0 for solver in runner.solvers.values()))
            self.assertEqual(rngs, {name: sampler.rng.getstate() for name, sampler in runner.samplers.items()})
            self.assertFalse((Path(directory) / 'checkpoints').exists())

    def test_session_reuses_one_executor_across_outer_iterations(self):
        from scripts.parallel_cfr import collect_samples
        with tempfile.TemporaryDirectory() as directory:
            config = tiny_config(directory)
            config['workers'] = 2
            runner = TrainingRunner(config)
            with patch('scripts.parallel_cfr.collect_samples', wraps=collect_samples) as collect:
                runner.run(iterations=2)
            executors = [call.kwargs['executor'] for call in collect.call_args_list]
            self.assertEqual(len(executors), 4)
            self.assertIsNotNone(executors[0])
            self.assertTrue(all(executor is executors[0] for executor in executors))
            self.assertEqual(runner.iteration, 2)
            restored = TrainingRunner.load_checkpoint(Path(directory) / 'checkpoints/iteration_000002.pt')
            self.assertEqual(restored.iteration, 2)

    def test_second_track_failure_rolls_back_first_track(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = TrainingRunner(tiny_config(directory))
            from ml.deep_cfr import DeepCFRSolver
            original = DeepCFRSolver.run_iteration
            def fail_hu(solver, *args, **kwargs):
                if len(solver.players) == 2:
                    raise RuntimeError('Injected HU failure')
                return original(solver, *args, **kwargs)
            with patch.object(DeepCFRSolver, 'run_iteration', fail_hu):
                with self.assertRaisesRegex(RuntimeError, 'HU failure'):
                    runner.run_iteration()
            self.assertEqual(runner.iteration, 0)
            self.assertTrue(all(s.version == 0 for s in runner.solvers.values()))
            self.assertFalse((Path(directory) / 'metrics.jsonl').exists())

    def test_atomic_checkpoint_replace_failure_keeps_previous_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = TrainingRunner(tiny_config(directory))
            path = Path(directory) / 'checkpoint.pt'
            runner.save_checkpoint(path)
            original = path.read_bytes()
            with patch('training.checkpoint.os.replace', side_effect=OSError('Injected replace failure')):
                with self.assertRaises(OSError):
                    runner.save_checkpoint(path)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(Path(directory).glob('.checkpoint.pt.*')), [])
            self.assertEqual(TrainingRunner.load_checkpoint(path).iteration, 0)

    def test_outcome_selection_requires_explicit_epsilon(self):
        from training.config import validate_config
        with tempfile.TemporaryDirectory() as directory:
            config = tiny_config(directory)
            self.assertNotIn('outcome_epsilon', config)
            config['traversal_mode'] = 'outcome_sampling'
            with self.assertRaisesRegex(ValueError, 'fields'):
                validate_config(config)
            config['outcome_epsilon'] = .6
            self.assertEqual(validate_config(config)['traversal_mode'], 'outcome_sampling')
            runner = TrainingRunner(config)
            row = runner.run_iteration()
            self.assertTrue(all(track['epsilon'] == .6 for track in row['tracks'].values()))
            restored = TrainingRunner.load_checkpoint(Path(directory) / 'checkpoints/iteration_000001.pt')
            self.assertTrue(all(s.traversal_mode == 'outcome_sampling' for s in restored.solvers.values()))
