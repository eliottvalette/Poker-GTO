"""Minimum opening coverage and card sensitivity; never an equilibrium certificate."""
from __future__ import annotations

from collections import Counter
from itertools import combinations
import random

from infoset import observe
from poker_game_expresso import HandState
from training.metrics import opening_hand_class


def assess_opening(solver) -> dict:
    count = len(solver.players)
    coverage = {}
    reasons = []
    for name, memory in (('advantage', solver.advantage_memory), ('strategy', solver.strategy_memory)):
        classes = Counter(holding for sample in memory.samples
                          if (holding := opening_hand_class(sample.state)) is not None)
        coverage[name] = {'classes': len(classes), 'minimum_records_per_class': min(classes.values(), default=0)}
        if len(classes) != 169 or min(classes.values(), default=0) < 4:
            reasons.append(f'{name}: require all 169 opening classes with at least four retained records each')
    root = HandState.start({p: 25. if count == 3 else 37.5 for p in range(count)}, 0, random.Random(42))
    rows = []
    for cards in combinations(range(52), 2):
        root.actor.cards = cards
        rows.append(solver.average_model.probabilities(observe(root)))
    # The largest pairwise action difference catches near-constant output without
    # assigning any prescribed action to a particular holding.
    span = max(max(row[i] for row in rows) - min(row[i] for row in rows) for i in range(len(rows[0])))
    if span < .15:
        reasons.append('Opening action probabilities vary by less than 15 percentage points across all 1326 holdings')
    probes = {}
    if not reasons:
        from training.preflop_diagnostics import paired_opening_values
        for label, holding in (('AA', (48, 49)), ('72o', (20, 1))):
            report = paired_opening_values(solver.average_model, count,
                                          25. if count == 3 else 37.5,
                                          holding, samples=128, seed=790019)
            # Conservative screening margin across these multiple action probes.
            # This rejects demonstrated local weaknesses; passing is not a proof
            # of equilibrium or a formal sequential promotion test.
            lower = max(row['paired_gain_vs_mixture']['mean']
                        - 3.5 * row['paired_gain_vs_mixture']['se']
                        for row in report['actions'].values())
            probes[label] = {'largest_conservative_gain_bb': lower,
                             'samples': report['samples'], 'seed': report['seed']}
            if lower > .5:
                reasons.append(f'{label}: a root-action deviation improves the fixed-policy probe by more than 0.5 BB')
    return {'version': 1, 'passed': not reasons, 'coverage': coverage,
            'maximum_action_span': span, 'reasons': reasons, 'probes': probes,
            'scope': 'coverage, card sensitivity and fixed-continuation probes; poker strength remains experimental'}
