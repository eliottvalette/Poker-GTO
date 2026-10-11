"""Context allocation correctness, resume, and frozen-validation contracts."""
import copy
from dataclasses import replace
from itertools import combinations, product
import random
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import torch
from actions import ACTION_IDS
from infoset import observe
from poker_game_expresso import HandState
from ml.context_replay import ContextReplay, context_key
from ml.memory import TrainingSample
from training.checkpoint import pack_memory, unpack_memory

CAPS={'first_open':169,'facing_open':1,'facing_3bet':1,'facing_4bet_or_more':1,'facing_jam':1,'other':1}

def sample(holding=(48,49), target=1., group='g', kind='advantage'):
    root=HandState.start({0:25.,1:25.},0,random.Random(2))
    root.actor.cards=holding
    obs=observe(root)
    values=tuple(target if legal else 0. for legal in obs.legal_mask)
    if kind=='strategy': values=tuple(v/sum(values) for v in values)
    return TrainingSample(1,root.current_player,obs,values,2.,kind,0,trajectory_id=group,root_group=group)

class ContextReplayTests(unittest.TestCase):
    def test_exact_expected_action_mass_and_no_double_correction(self):
        from ml.street_replay import StreetReplay
        rows=[sample(target=i+1.,group=str(i)) for i in range(3)]
        rows += [sample((44,45),target=5+i,group='k'+str(i)) for i in range(2)]
        memory=ContextReplay(CAPS,1024**2,'advantage',9)
        for row in rows: memory.add(row)
        expected=[sum(s.weight*s.target[i] for s in rows) for i in range(len(ACTION_IDS))]
        estimates=[]
        for aa,kk in product(rows[:3],rows[3:]):
            memory.memories['first_open/AA'].samples=[aa]
            memory.memories['first_open/KK'].samples=[kk]
            estimates.append([sum(s.weight*s.target[i] for s in memory.samples) for i in range(len(ACTION_IDS))])
        for i,total in enumerate(expected): self.assertAlmostEqual(sum(r[i] for r in estimates)/len(estimates),total)
        street=StreetReplay({s:174 if s=='PREFLOP' else 1 for s in ('PREFLOP','FLOP','TURN','RIVER')},
                            {s:1024**2 for s in ('PREFLOP','FLOP','TURN','RIVER')},'advantage',1,CAPS)
        street.memories['PREFLOP']=memory
        self.assertEqual([s.weight for s in street.samples],[s.weight for s in memory.samples])

    def test_resume_retains_original_weights_and_rng(self):
        memory=ContextReplay(CAPS,1024**2,'advantage',9)
        for i in range(20): memory.add(sample(target=i,group=str(i)))
        restored=unpack_memory(pack_memory(memory))
        for i in range(20,40):
            memory.add(sample(target=i,group=str(i)));restored.add(sample(target=i,group=str(i)))
        self.assertEqual(pack_memory(memory),pack_memory(restored))
        self.assertEqual(memory.memories['first_open/AA'].samples[0].weight,2.)
        self.assertEqual(memory.samples[0].weight,80.)
        malformed=pack_memory(memory);malformed['strata']['first_open/KK']=malformed['strata']['first_open/AA']
        with self.assertRaises(ValueError):unpack_memory(malformed)

    def test_context_checkpoint_and_frozen_bank_resume(self):
        from tests.test_street_networks import street_config
        from training.runner import TrainingRunner
        with tempfile.TemporaryDirectory() as tmp:
            config=street_config(tmp)
            for kind in ('advantage','strategy'):
                config['street_networks'][kind]['PREFLOP']['context_capacities']={**CAPS,'first_open':395}
            runner=TrainingRunner(config)
            runner.run_iteration()
            runner.save_checkpoint(Path(tmp)/'checkpoint.pt')
            restored=TrainingRunner.load_checkpoint(Path(tmp)/'checkpoint.pt')
            self.assertEqual(pack_memory(runner.solvers['hu'].strategy_memory),pack_memory(restored.solvers['hu'].strategy_memory))
            runner.run_iteration();restored.run_iteration()
            self.assertEqual(pack_memory(runner.solvers['hu'].strategy_memory),pack_memory(restored.solvers['hu'].strategy_memory))
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(runner.solvers['hu'].average_model.parameters(),restored.solvers['hu'].average_model.parameters())))

    def test_frozen_targets_stay_fixed_and_are_not_exported(self):
        from ml.fixed_validation import fixed_metrics
        from ml.model import AdvantageNetwork
        torch.set_num_threads(1)
        model=AdvantageNetwork()
        parts={k:[sample(group=f'{k}-{i}') for i in range(64)] for k in ('validation','test')}
        state={}
        before=fixed_metrics(model,state,parts,4,1,'advantage')
        bank=copy.deepcopy(state['validation_bank'])
        later={k:[replace(s,target=tuple(99*v for v in s.target)) for s in rows] for k,rows in parts.items()}
        after=fixed_metrics(model,state,later,5,2,'advantage')
        self.assertEqual(before,after);self.assertEqual(bank,state['validation_bank'])
        self.assertEqual(before['test']['zero_predictor_loss'],1.)
