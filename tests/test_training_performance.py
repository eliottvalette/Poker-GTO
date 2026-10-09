"""Numerical and accounting contracts protected by training performance changes."""
import copy
from dataclasses import asdict, fields, is_dataclass, replace
import sys
import unittest

import torch

from actions import ACTION_IDS
from features.neural import neural_observation, NEURAL_NUMERIC_NAMES
from infoset import observe, HISTORY_WIDTH
from ml.memory import TrainingSample, sample_bytes
from ml.model import encode_batch
from training.evaluation import fixed_roots


def reference_size(sample):
    seen=set()
    def visit(value):
        if is_dataclass(value) or isinstance(value,(tuple,list,dict)):
            if id(value) in seen:
                return 0
            seen.add(id(value))
        total=sys.getsizeof(value)
        if is_dataclass(value):
            total += sys.getsizeof(value.__dict__) if hasattr(value,'__dict__') else 0
            total += sum(visit(getattr(value,f.name)) for f in fields(value))
        elif isinstance(value,(tuple,list)):
            total += sum(visit(v) for v in value)
        elif isinstance(value,dict):
            total += sum(visit(k)+visit(v) for k,v in value.items())
        return total
    return visit(sample)


class PerformanceParityTests(unittest.TestCase):
    def test_checkpoint_fast_packing_preserves_primitive_schema(self):
        from training.checkpoint import _pack_sample
        obs=neural_observation(observe(fixed_roots(2)[0][1]))
        s=TrainingSample(1,obs.hero,obs,tuple(0. for _ in ACTION_IDS),1.,'advantage',0)
        for sample in (s,replace(s,target=list(s.target),state=replace(obs,cards=list(obs.cards)))):
            self.assertEqual(_pack_sample(sample),asdict(sample))
        mutable=replace(s,target=list(s.target),state=replace(obs,cards=list(obs.cards)))
        packed=_pack_sample(mutable)
        packed['target'][0]=7
        packed['state']['cards'][0]=51
        self.assertEqual(mutable.target[0],0.)
        self.assertEqual(mutable.state.cards,list(obs.cards))
        buffered=replace(s,state=replace(obs,numeric_data=bytearray(obs.numeric_data)))
        packed=_pack_sample(buffered)
        self.assertEqual(packed,asdict(buffered))
        packed['state']['numeric_data'][0] ^= 1
        self.assertEqual(buffered.state.numeric_data,bytearray(obs.numeric_data))

    def test_bulk_encoding_matches_scalar_reference(self):
        rows=[neural_observation(observe(root)) for count in (2,3) for _,root in fixed_roots(count)]
        actual=encode_batch(rows)
        histories=[o.history for o in rows]
        expected=torch.zeros(len(rows),max(map(len,histories)),HISTORY_WIDTH)
        for i,h in enumerate(histories):
            expected[i,:len(h)]=torch.tensor(h)
        self.assertTrue(torch.equal(actual['history'],expected))
        self.assertTrue(torch.equal(actual['numeric'],torch.tensor([o.numeric for o in rows],dtype=torch.float32)))
        self.assertTrue(torch.equal(actual['position'],torch.tensor([
            round(o.numeric[NEURAL_NUMERIC_NAMES.index('hero_position')]*2) for o in rows])))
        self.assertTrue(torch.equal(actual['cards'],torch.tensor([o.cards for o in rows])))
        self.assertTrue(torch.equal(actual['mask'],torch.tensor([o.legal_mask for o in rows])))
        original=rows[0].numeric_data
        actual['numeric'][0,0]=999.
        self.assertEqual(rows[0].numeric_data,original)

    def test_fast_accounting_equals_recursive_reference_with_aliases(self):
        for count in (2,3):
            for _,root in fixed_roots(count):
                obs=neural_observation(observe(root))
                s=TrainingSample(10,obs.hero,obs,tuple(0. for _ in ACTION_IDS),1.,'advantage',9)
                for candidate in (s,replace(s,target=obs.legal_mask),replace(s,target=list(s.target))):
                    self.assertEqual(sample_bytes(candidate),reference_size(candidate))

    def test_deepcopy_shares_only_immutable_payloads(self):
        obs=neural_observation(observe(fixed_roots(2)[0][1]))
        sample=TrainingSample(1,obs.hero,obs,tuple(0. for _ in ACTION_IDS),1.,'advantage',0)
        self.assertIs(copy.deepcopy(sample),sample)
        mutable=replace(sample,target=list(sample.target),state=replace(obs,cards=list(obs.cards)))
        copied=copy.deepcopy(mutable)
        copied.target[0]=9.
        copied.state.cards[0]=51
        self.assertEqual(mutable.target[0],0.)
        self.assertEqual(mutable.state.cards, list(obs.cards))
