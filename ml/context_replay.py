"""Preflop context reservoirs with exact inclusion probabilities and bounded storage."""
from __future__ import annotations
from dataclasses import replace
from ml.memory import ReservoirMemory, TrainingSample
from training.metrics import decision_context, opening_hand_class
from features.neural import NEURAL_NUMERIC_NAMES

CONTEXTS = ('first_open', 'facing_open', 'facing_3bet', 'facing_4bet_or_more', 'facing_jam', 'other')
RANKS = '23456789TJQKA'
HAND_CLASSES = tuple(sorted([r+r for r in RANKS] + [RANKS[h]+RANKS[l]+s
                    for h in range(13) for l in range(h) for s in ('s','o')]))


def context_key(sample: TrainingSample) -> str:
    if sample.state.street != 0:
        raise ValueError('Context replay accepts preflop decisions only')
    holding = opening_hand_class(sample.state)
    if holding is not None:
        return 'first_open/' + holding
    values = dict(zip(NEURAL_NUMERIC_NAMES, sample.state.numeric))
    if values['to_call'] > 0 and values['to_call'] >= values['stack_0']:
        return 'facing_jam'
    context = decision_context(sample.state)
    return context if context in CONTEXTS else 'other'


def expanded_capacities(capacities: dict[str, int]) -> dict[str, int]:
    if (set(capacities) != set(CONTEXTS) or any(type(v) is not int or v < 1 for v in capacities.values())
            or capacities['first_open'] < len(HAND_CLASSES)):
        raise ValueError('Explicit positive context capacities and at least 169 opening slots required')
    size, extra = divmod(capacities['first_open'], len(HAND_CLASSES))
    return {**{f'first_open/{h}': size + (i < extra) for i,h in enumerate(HAND_CLASSES)},
            **{k:v for k,v in capacities.items() if k != 'first_open'}}


class ContextReplay:
    """HT-corrected stream totals; normalized fitting remains a ratio estimator.

    Each stratum uses Algorithm R. Original samples retain collector weights;
    corrected views divide by the exact stratum inclusion probability once.
    """
    def __init__(self, capacities: dict[str, int], byte_budget: int, kind: str, seed: int):
        self.capacities = dict(capacities)
        expanded = expanded_capacities(capacities)
        total = sum(expanded.values())
        budgets = {k: byte_budget * v // total for k,v in expanded.items()}
        budgets[next(iter(budgets))] += byte_budget - sum(budgets.values())
        self.memories = {k: ReservoirMemory(v, seed+i, kind, 'hand_chip_delta', budgets[k])
                         for i,(k,v) in enumerate(expanded.items())}

    def add(self, sample: TrainingSample) -> None:
        if sample.traversal_mode != 'external_sampling':
            raise ValueError('Context replay requires external_sampling')
        self.memories[context_key(sample)].add(sample)

    @property
    def kind(self): return next(iter(self.memories.values())).kind
    @property
    def objective(self): return 'hand_chip_delta'
    @property
    def traversal_mode(self): return 'external_sampling' if self.seen else None
    @property
    def capacity(self): return sum(m.capacity for m in self.memories.values())
    @property
    def seen(self): return sum(m.seen for m in self.memories.values())
    @property
    def byte_budget(self): return sum(m.byte_budget for m in self.memories.values())
    @property
    def used_bytes(self): return sum(m.used_bytes for m in self.memories.values())
    @property
    def samples(self):
        return [replace(s, weight=s.weight / min(1., m.capacity/m.seen))
                for m in self.memories.values() for s in m.samples]

    def diagnostics(self):
        return {k: {'generated':m.seen, 'retained':len(m.samples), 'capacity':m.capacity,
                    'inclusion_probability': min(1.,m.capacity/m.seen) if m.seen else None}
                for k,m in self.memories.items()}
