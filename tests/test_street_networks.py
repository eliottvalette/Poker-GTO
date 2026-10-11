"""Street routing, atomic snapshots, independent optimization and durable resume."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import torch
from infoset import observe
from ml.model import STREETS, StreetNetworks, encode_batch
from ml.deep_cfr import FrozenStrategy
from training.evaluation import fixed_roots
from training.runner import TrainingRunner


def street_config(directory, count=2):
    config = json.loads(Path(f'configs/train_{"hu" if count == 2 else "3max"}.json').read_text())
    config.pop('replay_opening_fraction', None)
    config.update(output_dir=str(directory), workers=1, trainer_threads=1,
                  checkpoint_every=100, evaluation_every=100, generation_batch_size=16)
    for track in ('hu', '3max'):
        config[track]['enabled'] = track == ('hu' if count == 2 else '3max')
        config[track]['advantage_capacity'] = 1600
        config[track]['advantage_byte_budget'] = 16 * 1024**2
        config[track]['traversals_per_player'] = 16
    config['strategy_capacity'] = 1600
    config['memory_byte_budget'] = 16 * 1024**2
    config['street_networks'] = {'version': 1, **{
        kind: {s: {'capacity': 400, 'byte_budget': 4 * 1024**2,
                   'every': 1, 'min_new_samples': 2, 'max_updates': 2} for s in STREETS}
        for kind in ('advantage', 'strategy')}}
    return config


class StreetNetworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_independent_parameters_and_mixed_batch_routing(self):
        models = [StreetNetworks(kind) for _ in (2,3) for kind in ('advantage','strategy')]
        parameters = [p for model in models for p in model.parameters()]
        self.assertEqual(len(parameters), len({id(p) for p in parameters}))
        observations = [observe(h) for _,h in fixed_roots(2)]
        batch = encode_batch(observations)
        for model in models:
            with torch.no_grad():
                output = model(batch)
                for i,obs in enumerate(observations):
                    expected = model.specialists[STREETS[obs.street]](encode_batch([obs]))
                    self.assertTrue(torch.allclose(output[i], expected[0], atol=1e-7))

    def test_resume_optimizers_and_frozen_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = TrainingRunner(street_config(directory))
            runner.run_iteration()
            solver = runner.solvers['hu']
            self.assertTrue(solver.advantage_model.trained.all())
            self.assertTrue(solver.average_model.trained.all())
            frozen = FrozenStrategy(solver.snapshot())
            observations = [observe(h) for _,h in fixed_roots(2)]
            old = [frozen(o) for o in observations]
            path = Path(directory)/'checkpoint.pt'
            runner.save_checkpoint(path)
            resumed = TrainingRunner.load_checkpoint(path)
            runner.run_iteration()
            resumed.run_iteration()
            for kind in ('advantage', 'average'):
                a = getattr(runner.solvers['hu'], f'{kind}_model').state_dict()
                b = getattr(resumed.solvers['hu'], f'{kind}_model').state_dict()
                self.assertTrue(all(torch.equal(a[k], b[k]) for k in a))
            self.assertEqual(old, [frozen(o) for o in observations])
            for kind, states in runner.solvers['hu'].specialist_states.items():
                for name,state in states.items():
                    self.assertEqual(state['updates'], 4)
                    self.assertEqual(state['fits'], 2)
                    self.assertEqual(state['last_fit_metrics']['model_version'],2)
                    self.assertIn('heldout_loss',state['last_fit_metrics'])
                    self.assertTrue(all(int(s['step']) == 4 for s in state['optimizer']['state'].values()))

    def test_export_all_streets_both_tracks(self):
        from ml.export_onnx import export_average_policy
        from ml.onnx_policy import StreetOnnxPolicy
        for count in (2,3):
            with self.subTest(count=count), tempfile.TemporaryDirectory() as directory:
                directory = Path(directory)
                config = street_config(directory,count)
                config['hu' if count == 2 else '3max']['traversals_per_player'] = 64
                runner = TrainingRunner(config)
                runner.run_iteration()
                solver = next(iter(runner.solvers.values()))
                path = directory/'average.pt'
                solver.export_average(path)
                track = 'hu' if count == 2 else '3max'
                bundle = directory/'bundle'; bundle.mkdir()
                catalog = export_average_policy(path, bundle/f'average_{track}.onnx', bundle/f'average_{track}.json',
                                                observe(runner.probes[next(iter(runner.solvers))][0][1]))
                policy = StreetOnnxPolicy(catalog, count, lambda name: (bundle/name).read_bytes())
                self.assertEqual(len(policy.sessions),0)
                for _,hand in fixed_roots(count):
                    obs = observe(hand)
                    actual = policy.probabilities(obs)
                    expected = solver.average_model.probabilities(obs)
                    self.assertLess(max(abs(a-b) for a,b in zip(actual,expected)), 1e-6)
                self.assertEqual(len(policy.sessions),4)
                from training.workflow import seal_bundle
                from training.publication import publish_bundle, StorageError
                from tests.test_publication import MemoryStorage
                seal_bundle(bundle, {"kind":"live_training_export", "iteration":1, "tracks":[track]})
                storage = MemoryStorage(); storage.fail = "_river.onnx"
                with self.assertRaises(StorageError):
                    publish_bundle(storage,track,bundle)
                self.assertNotIn(f"{track}/current.json",storage.objects)
                storage.fail = None
                pointer = publish_bundle(storage,track,bundle)
                self.assertEqual(len(storage.objects),6)
                self.assertEqual(storage.events[-2],('put',f'{track}/current.json'))
                self.assertEqual(pointer['iteration'],1)
                from training.evaluation_history import evaluate_published
                from training.poker_evaluation import EvaluationBudget
                history = evaluate_published(storage,track,directory/'street-history',EvaluationBudget(groups=1))
                self.assertEqual(history['status'],'complete')
                self.assertEqual(history['difference']['mean'],0)
                # A new schema has its own iteration clock; migration is explicit.
                old_bundle = directory/'legacy'; old_bundle.mkdir()
                old_manifest = copy.deepcopy(catalog['routes']['PREFLOP']['manifest'])
                old_manifest['iteration'] = 1000
                (old_bundle/f'average_{track}.json').write_text(json.dumps(old_manifest))
                (old_bundle/f'average_{track}.onnx').write_bytes((bundle/catalog['routes']['PREFLOP']['model_file']).read_bytes())
                seal_bundle(old_bundle, {"kind":"live_training_export", "iteration":1000, "tracks":[track]})
                migration_storage = MemoryStorage()
                old_pointer = publish_bundle(migration_storage,track,old_bundle)
                evaluate_published(migration_storage,track,directory/'migrated-history',EvaluationBudget(groups=1))
                with self.assertRaisesRegex(ValueError, 'explicit schema migration'):
                    publish_bundle(migration_storage,track,bundle)
                with self.assertRaisesRegex(ValueError, 'exact current'):
                    publish_bundle(migration_storage,track,bundle,migrate_shared_release='f'*64)
                migrated = publish_bundle(migration_storage,track,bundle,migrate_shared_release=old_pointer['release_id'])
                self.assertEqual(migrated['schema_migration_from'],old_pointer['release_id'])
                self.assertEqual(migrated['iteration'],1)
                compared = evaluate_published(migration_storage,track,directory/'migrated-history',EvaluationBudget(groups=1))
                self.assertEqual(compared['status'],'complete')
                self.assertEqual(compared['reference']['iteration'],1000)
                with self.assertRaisesRegex(ValueError, 'shared-policy writer'):
                    publish_bundle(migration_storage,track,old_bundle)
                from training.checkpoint import export_checkpoint_average
                runner.save_checkpoint(directory/'checkpoint.pt')
                export_checkpoint_average(directory/'checkpoint.pt',directory/'extracted.pt',track)
                from ml.deep_cfr import NeuralAveragePolicy
                extracted=NeuralAveragePolicy(directory/'extracted.pt')
                self.assertTrue(all(torch.equal(v, extracted.model.state_dict()[k])
                                    for k,v in solver.average_model.state_dict().items()))

    def test_river_enumeration_exact_action_mass(self):
        from scripts.strategy_collector_audit import small_root, exact_reference, enumerated_expectation
        root = small_root(3)
        reference = exact_reference(root)["reference"]
        for orientation in (0,1):
            actual = enumerated_expectation(root, "partial_enumeration", orientation=orientation,
                                            enumerate_from_street=3)
            for key,row in reference.items():
                for a,b in zip(actual[key]["action_mass"], row["action_mass"]):
                    self.assertAlmostEqual(a,b,places=11)

    def test_controlled_contexts_are_live_legal_and_cover_streets(self):
        from training.root_sampler import RootSampler
        from training.config import validate_config
        contexts = ["open", "facing_open", "facing_3bet", "facing_4bet", "short", "facing_jam",
                    "flop_check", "flop_bet", "flop_raise", "turn_check", "turn_bet", "turn_raise",
                    "river_check", "river_bet", "river_raise", "river_jam"]
        for count in (2,3):
            config = street_config("/tmp/unused", count)
            track = "hu" if count == 2 else "3max"
            root_config = config[track]["root_sampling"]
            root_config["decision_contexts"] = contexts
            root_config["mixture"] = {"on_policy":0, "synthetic":.5, "stratified":.5}
            validate_config(config)
            sampler = RootSampler(count, 907, root_config)
            seen = set()
            for i in range(64):
                hand = sampler.sample(traverser=i%count)
                self.assertFalse(hand.terminal)
                hand.assert_invariants()
                seen.add(hand.street)
            self.assertEqual(seen,set(STREETS))

    def test_parallel_workers_preserve_complete_street_snapshot(self):
        from ml.deep_cfr import TraversalTask
        from scripts.parallel_cfr import collect_samples
        from scripts.strategy_collector_audit import small_root
        with tempfile.TemporaryDirectory() as directory:
            runner=TrainingRunner(street_config(directory,3))
            solver=runner.solvers["3max"]
            solver.version=1
            solver.street_config["collector_enumerate_from_street"]=3
            tasks=[TraversalTask(i,i%3,small_root(3),901+i,10000,64) for i in range(6)]
            snapshot=solver.snapshot()
            self.assertEqual(collect_samples(snapshot,tasks,1),collect_samples(snapshot,tasks,2))

    def test_grouped_splits_ignore_private_deals_and_trajectory_seed(self):
        from ml.deep_cfr import TraversalTask, generate_samples
        from ml.street_training import grouped_partition
        from scripts.strategy_collector_audit import small_root
        with tempfile.TemporaryDirectory() as directory:
            solver = TrainingRunner(street_config(directory)).solvers['hu']
            first = small_root(2)
            second = first.clone()
            players = list(second.players.values())
            players[0].cards, players[1].cards = players[1].cards, players[0].cards
            results = [generate_samples(solver.snapshot(), TraversalTask(i,root.current_player,root,501+i,10000,64))
                       for i,root in enumerate((first,second))]
            samples = [r.advantages[0] for r in results]
            self.assertEqual(samples[0].root_group,samples[1].root_group)
            self.assertNotEqual(samples[0].trajectory_id,samples[1].trajectory_id)
            self.assertEqual(grouped_partition(samples[0],42),grouped_partition(samples[1],42))

    def test_no_refit_without_new_data_and_reject_corrupt_optimizer(self):
        from ml.street_training import fit_specialists, validate_specialist_states
        with tempfile.TemporaryDirectory() as directory:
            runner = TrainingRunner(street_config(directory))
            solver = runner.solvers['hu']
            with self.assertRaisesRegex(ValueError, 'cold'):
                solver.export_average(Path(directory)/'cold.pt')
            runner.run_iteration()
            solver = runner.solvers['hu']
            original = copy.deepcopy(solver.advantage_model.state_dict())
            model,states,metrics = fit_specialists(solver.advantage_model,solver.advantage_memory,
                solver.specialist_states['advantage'],solver.street_config['advantage'],
                version=solver.version+1,seed=solver.seed,batch_size=64,learning_rate=.0003,cache_encoding=True)
            self.assertTrue(all(m['reason']=='no_new_retained_training_data' for m in metrics.values()))
            self.assertTrue(all(torch.equal(v,model.state_dict()[k]) for k,v in original.items()))
            validate_specialist_states(solver)
            solver.specialist_states['advantage']['PREFLOP']['updates'] += 1
            with self.assertRaisesRegex(ValueError, 'optimizer'):
                validate_specialist_states(solver)

    def test_independent_family_cadence_with_bounded_storage(self):
        with tempfile.TemporaryDirectory() as directory:
            config = street_config(directory)
            config['hu']['traversals_per_player'] = 32
            for settings in config['street_networks']['strategy'].values():
                settings['every'] = 3
            runner = TrainingRunner(config)
            runner.enable_bounded_storage()
            for iteration in range(1,4):
                runner.run_iteration()
                solver = runner.solvers['hu']
                self.assertEqual(bool(solver.average_model.trained.any()), iteration == 3)
            self.assertEqual(len(runner.metrics),1)
            self.assertTrue(solver.average_model.trained.all())
            self.assertTrue(all(s['updates']==2 for s in solver.specialist_states['strategy'].values()))
            runner.save_checkpoint(Path(directory)/'checkpoint.pt')
            resumed = TrainingRunner.load_checkpoint(Path(directory)/'checkpoint.pt')
            self.assertEqual(resumed.runtime_storage,runner.runtime_storage)
            self.assertEqual(resumed.metadata['network_fit'],'independent_streets_persistent_weights_and_adam')
