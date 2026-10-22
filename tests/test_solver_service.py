"""Production solver routing, persistence, sampling, and publication contracts."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hybrid.knowledge import calculate
from hybrid.knowledge_curriculum import FAMILIES, scenario
from hybrid.state import ComputeBudget
from training.solver_service import DEFAULTS, ENGINE, initialize, step, fit_students, publish, publish_status
from training.publication import StorageError

class Storage:
    def __init__(self): self.data = {}
    def get(self, key):
        if key not in self.data: raise StorageError(404, key)
        return self.data[key]
    def put(self, key, data, **kwargs): self.data[key] = data

class SolverServiceTests(unittest.TestCase):
    def test_completed_allocation_is_persisted_and_published(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            initialize(root, 'hu')
            state = {'cursor': 32000, 'status': 'running', 'streets': {},
                     'failed': {}, 'student': {'accepted': False}}
            (root / 'state.json').write_text(json.dumps(state))
            self.assertEqual(step(root, 'hu')['status'], 'allocation_complete')
            self.assertEqual(json.loads((root / 'state.json').read_text())['status'],
                             'allocation_complete')
            storage = Storage()
            publish_status(storage, root, 'hu')
            self.assertEqual(json.loads(storage.data['solver/hu/status.json'])['status'],
                             'allocation_complete')

    def test_sampled_resume_equals_uninterrupted_and_declares_sampling(self):
        item = scenario(3, 'TURN', FAMILIES[0], 0)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            budget = ComputeBudget(iterations=2, samples=4, max_depth=1, max_nodes=100000, seed=141)
            calculate(p/'resume.json', item.root, item.ranges, budget, split='train', sample_private=True)
            resumed = calculate(p/'resume.json', item.root, item.ranges, budget, split='train', sample_private=True)
            whole = calculate(p/'whole.json', item.root, item.ranges,
                              ComputeBudget(iterations=4, samples=4, max_depth=1, max_nodes=100000, seed=141),
                              split='train', sample_private=True)
            self.assertEqual(resumed.checkpoint, whole.checkpoint)
            self.assertEqual(resumed.strategy_by_hand, whole.strategy_by_hand)
            self.assertEqual(json.loads((p/'whole.json').read_text())['contract']['private_sampling'], 'seeded_compatible_joint')
            with self.assertRaisesRegex(ValueError, 'contract changed'):
                calculate(p/'resume.json', item.root, item.ranges, budget, split='train')

    def test_service_resumes_tables_and_rejects_changed_contract(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            config = {**DEFAULTS, 'roots_per_family_street': 1, 'samples': 2, 'iterations_per_visit': 2}
            first = step(root,'hu',config)
            second = step(root,'hu',config)
            self.assertEqual((first['cursor'], second['cursor']), (1,2))
            self.assertEqual(second['computed'],2)
            self.assertEqual(len(list((root/'tables').glob('*.json'))),2)
            self.assertEqual(fit_students(root,'hu')['status'], 'insufficient_split_coverage')
            with self.assertRaisesRegex(ValueError,'contract changed'):
                initialize(root,'hu',{**config, 'samples':3})
            storage=Storage()
            publish_status(storage, root,'hu')
            status=json.loads(storage.get('solver/hu/status.json'))
            self.assertEqual(status['engine'],ENGINE)
            self.assertNotIn('hu/current.json',storage.data)

    def test_hourly_solver_history_never_overwrites_nn_history(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); initialize(root,'hu')
            storage=Storage()
            row={'at':'2026-10-16T12:00:00+00:00','visits':1}
            with patch('training.solver_service.evaluate',return_value=row) as evaluate:
                publish(storage,root,'hu')
                self.assertEqual(evaluate.call_count,1)
            self.assertIn('solver/hu/history.json',storage.data)
            self.assertNotIn('evaluation/hu/history.json.gz',storage.data)
            self.assertEqual(json.loads(storage.data['solver/hu/current.json'])['student_accepted'],False)

    def test_evaluation_is_read_only_and_students_require_explicit_dataset_updates(self):
        import torch
        from hybrid.distillation import SolverStudents
        from hybrid.solver_experiments import bounded_roots
        from training.solver_service import refine_references, evaluate
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory() as d:
                root=Path(d); initialize(root,'hu')
                refine_references(root,'hu')
                before={p.name:p.read_bytes() for p in (root/'references').glob('*.json')}
                row=evaluate(root,'hu')
                self.assertEqual(len(row['streets']),4)
                self.assertEqual(before,{p.name:p.read_bytes() for p in (root/'references').glob('*.json')})
                paths=[]
                budget=ComputeBudget(iterations=2,samples=16,max_depth=1,max_nodes=100000)
                for split in ('train','validation','test'):
                    for state,ranges in bounded_roots(2,split):
                        p=root/'tables'/f'{split}-{state.street}.json'
                        calculate(p,state,ranges,budget,split=split); paths.append(p)
                learner=SolverStudents(2,seed=141)
                learner.fit(paths,updates=1)
                state,ranges=next(bounded_roots(2,'train'))
                calculate(paths[0],state,ranges,budget,split='train')
                with self.assertRaisesRegex(ValueError,'Frozen distillation dataset changed'):
                    learner.fit(paths,updates=1)
                learner.fit(paths,updates=1,allow_dataset_update=True)
                self.assertTrue(all(n==2 for n in learner.updates.values()))
                learner.save(root/'students.pt')
                loaded=SolverStudents.load(root/'students.pt')
                self.assertEqual(loaded.updates,learner.updates)
        finally:
            torch.set_num_threads(previous)

    def test_depth_limited_river_preserves_continuation_identity(self):
        from hybrid.solver_inputs import continuation_id, encode_roots
        from hybrid.knowledge import root_supervision
        item=scenario(2,'RIVER',FAMILIES[1],0)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'river.json'
            calculate(path,item.root,item.ranges,ComputeBudget(iterations=2,samples=4,max_depth=1,max_nodes=10000),
                      split='train',continuation=item.policy,sample_private=True,depth_limited=True)
            contract,records=root_supervision(path,'strategy')
            self.assertEqual(continuation_id(contract),'passive-continuation-v1')
            encoded=encode_roots([records[0].state],[contract],('terminal-river-v1','passive-continuation-v1'))
            self.assertEqual(float(encoded['solving_context'][0,-1]),1.)
