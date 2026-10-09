"""Operational bounds preserve learned weights, replay and deterministic continuation."""
from pathlib import Path
import tempfile
import unittest
import hashlib
import io
import torch
from unittest.mock import patch
from test_training_runner import tiny_config
from training.runner import TrainingRunner
from training.runtime_storage import RuntimeStorage
from training.checkpoint import read_checkpoint, write_checkpoint, export_checkpoint_average
from training.vps import prepare_runner, export_live, run_continuous


class VPSStorageTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_retention_and_deterministic_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner = TrainingRunner(tiny_config(root/'original'))
            runner.run_iteration()
            initial = root/'initial.pt'
            runner.save_checkpoint(initial)
            source_hash = hashlib.sha256(initial.read_bytes()).hexdigest()
            bounded = TrainingRunner.load_checkpoint(initial)
            bounded.config = {**bounded.config, 'output_dir': str(root/'bounded')}
            bounded.enable_bounded_storage(RuntimeStorage(checkpoints_keep=2,journal_keep=2))
            baseline = TrainingRunner.load_checkpoint(initial)
            baseline.run_iteration(); bounded.run_iteration()
            for name in baseline.solvers:
                a,b=baseline.solvers[name],bounded.solvers[name]
                self.assertEqual(a.strategy_memory.samples,b.strategy_memory.samples)
                self.assertEqual(a.advantage_memory.samples,b.advantage_memory.samples)
                for key,value in a.average_model.state_dict().items():
                    self.assertTrue(torch.equal(value,b.average_model.state_dict()[key]))
            path=root/'bounded/checkpoint.pt';bounded.save_checkpoint(path)
            restored=TrainingRunner.load_checkpoint(path)
            self.assertEqual(restored.runtime_storage,bounded.runtime_storage)
            restored.run_iteration();bounded.run_iteration()
            for name in bounded.solvers:
                self.assertEqual(restored.solvers[name].strategy_memory.samples,bounded.solvers[name].strategy_memory.samples)
            bounded.run_iteration()
            self.assertEqual(len(bounded.metrics),1)
            self.assertTrue(all(len(s.metrics)==1 for s in bounded.solvers.values()))
            self.assertEqual(len(list((root/'bounded/checkpoints').glob('iteration_*.pt'))),2)
            self.assertEqual(len(list((root/'bounded/diagnostics').glob('*.gz'))),2)
            self.assertEqual(hashlib.sha256(initial.read_bytes()).hexdigest(),source_hash)

    def test_old_envelope_and_new_checksum(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);runner=TrainingRunner(tiny_config(root))
            runner.save_checkpoint(root/'new.pt')
            raw=read_checkpoint(root/'new.pt')
            payload=io.BytesIO();torch.save(raw,payload);data=payload.getvalue()
            torch.save({'payload':data,'sha256':hashlib.sha256(data).hexdigest()},root/'old.pt')
            self.assertEqual(TrainingRunner.load_checkpoint(root/'old.pt').iteration,0)
            import zipfile,json
            with zipfile.ZipFile(root/'bad.pt','w') as z:
                z.writestr('payload.pt',data)
                z.writestr('checkpoint.json',json.dumps({'format':'streamed-checkpoint-v1','sha256':'0'*64}))
            with self.assertRaisesRegex(ValueError,'Checksum mismatch'): read_checkpoint(root/'bad.pt')

    def test_live_export_never_reloads_checkpoint_and_rejects_partial_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);config=tiny_config(root)
            config['3max']['enabled']=False
            config['hu']['root_sampling']['tournament_start_players']=2
            runner=TrainingRunner(config);runner.enable_bounded_storage();runner.run_iteration()
            with patch('training.runner.TrainingRunner.load_checkpoint',side_effect=AssertionError('Reload forbidden')):
                bundle=export_live(runner)
            self.assertTrue((bundle/'average_hu.onnx').exists())
            pointer=(root/'exports/current.json').read_bytes()
            runner.run_iteration()
            with patch('training.vps.export_average_policy',side_effect=RuntimeError('Export failed')):
                with self.assertRaisesRegex(RuntimeError,'Export failed'):export_live(runner)
            self.assertEqual((root/'exports/current.json').read_bytes(),pointer)
            self.assertFalse(list((root/'exports').glob('.export-*')))
            export_live(runner,keep=1)
            self.assertEqual(len(list((root/'exports').glob('iteration_*'))),1)
            path=root/'checkpoint.pt';runner.save_checkpoint(path)
            export_checkpoint_average(path,root/'extracted.pt','hu')
            self.assertEqual(TrainingRunner.load_checkpoint(path).iteration,2)

    def test_bounded_continuous_sessions_resume_and_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); config=tiny_config(root/'source')
            config['3max']['enabled']=False
            config['hu']['root_sampling']['tournament_start_players']=2
            original=TrainingRunner(config);original.run_iteration()
            source=root/'source.pt';original.save_checkpoint(source)
            fingerprint=hashlib.sha256(source.read_bytes()).hexdigest()
            run_continuous('hu',root/'service',source_checkpoint=source,total_seconds=.001)
            first=TrainingRunner.load_checkpoint(root/'service/checkpoint.pt')
            self.assertEqual(first.iteration,2)
            self.assertEqual(first.config['workers'],1)
            self.assertEqual(first.config['trainer_threads'],1)
            self.assertEqual(first.config['strategy_capacity'],config['strategy_capacity'])
            run_continuous('hu',root/'service',total_seconds=.001)
            self.assertEqual(TrainingRunner.load_checkpoint(root/'service/checkpoint.pt').iteration,3)
            self.assertTrue((root/'service/exports/current.json').exists())
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),fingerprint)
