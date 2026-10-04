# Hand-terminal chip-EV learning

## Objective and ownership

The supported objective is `hand_chip_delta`, for winner-take-all Expresso.
For each seated player, utility is settled stack minus the stack before posting
this hand's blinds. The engine checks that these utilities sum to zero.
Physical chips use the initial big blind as their unit; increasing blinds never
rescales stacks or creates chips. Observations and browser amounts use the
current big blind. Both the blind level index and physical chips per current
big blind are explicit model features.

`TournamentState` owns persistent stacks, elimination, button rotation, the
three-player to heads-up transition, and scheduling. A caller starts a hand;
`hand_root` clones that existing hand for the solver. Neither sampler starts
another hand. Exact folded-player payoffs can be returned before showdown,
because that player's remaining chips can no longer change.

`ExternalSamplingMCCFR` and the Deep CFR external-sampling traversal branch at
all future traverser decisions in that hand. External sampling is the default
Deep CFR sample generator. Outcome sampling remains an explicit, tested
alternative with epsilon 0.6. Its estimator and average weighting are documented
in [outcome-sampling.md](outcome-sampling.md). No importance weights or targets
are clipped or transformed, and no float64 training-loss workaround was added.

Advantage and average-policy memories/models remain separate. Frozen workers
produce samples for central training. State schema 3, checkpoint schema 3 and
ONNX manifest 2 reject incompatible tournament-objective artifacts. Existing
legacy model files are preserved. Paid second/third places and ICM/$EV are
unsupported, not approximated with this objective. Per-hand cEV is not a claim
of an exact equilibrium of the complete tournament.

## Blind progression

`BlindSchedule` is a validated list of hand-number thresholds and `BlindLevel`s.
The tournament applies changes only at the start of a new hand; an in-progress
hand keeps its level. The configurable default simulation preset starts at
0.5/1 and doubles at hands 11, 21, 31, 41 and 51, holding 16/32 thereafter.
This is a simulation preset, not an official timed Expresso structure. A timed
schedule and a venue-specific schedule remain separate product decisions.
`BlindSchedule.fixed()` remains available for controlled tests.

The TypeScript engine mirrors this contract under Python-generated parity
fixtures. The migration changes data/engine logic, not the table layout.
Progression tests also cover a discovered short-blind defect: a lone actionable
player facing an all-in opponent must not call a phantom full nominal big blind.

## Measured traversal cost

Reproduce without training:

```python
from scripts.hand_diagnostics import run_hand_probe
report = run_hand_probe(trials=64, max_nodes=2000, max_depth=100, epsilon=0.6)
```

[hand-diagnostics.json](hand-diagnostics.json) contains seeds, raw samples,
legal masks, errors/censoring fields and distribution summaries. The run used
one fixed deal per scenario (seed 3), a uniform strategy and 64 action-sampling
seeds for every active player. All 512 external and 512 outcome traversals
completed; none reached a budget or numerical failure. Total elapsed time was
10.05 seconds on the local CPU, including separate external average-policy
passes. These are bounded diagnostics, not a learned-policy benchmark.

External regret traversal metrics (time excludes the separate averaging pass):

| Root | Nodes median / p95 / max | Depth median / max | Time median / p95 |
|---|---:|---:|---:|
| 25 BB three-player | 94 / 359 / 899 | 12 / 18 | 14.24 / 64.11 ms |
| 5 BB three-player | 29 / 71 / 149 | 7 / 16 | 3.76 / 10.43 ms |
| 25 BB heads-up | 218.5 / 625 / 1173 | 11 / 14 | 31.05 / 99.48 ms |

Outcome traversal median times were 0.51, 0.40 and 0.38 ms respectively.
All eight player/root value comparisons and all 39 root action-slot regret
comparisons were within six paired standard errors; illegal slots are exact
zeros. With 64 trials, this broad consistency check does not establish precise
convergence or low variance. The regression suite additionally enumerates
complete outcome paths on bounded river states and compares 1,000-trial
estimates against independent full-tree/external references, including
nonuniform and zero-probability target policies. Uniform diagnostics alone
would not validate general importance correction.

## Target and weight distributions

Absolute regret targets include legal slots only, including genuine zero targets.
At these initial-level roots, one chip equals one current BB.

| Root | External abs regret median / p95 / max | Outcome abs regret median / p95 / max |
|---|---:|---:|
| 25 BB three-player | 7.5 / 24.08 / 42.25 | 17 / 629.69 / 11,760 |
| 5 BB three-player | 1.75 / 5 / 9.13 | 3.5 / 52.5 / 1,323 |
| 25 BB heads-up | 7.5 / 23.52 / 29.93 | 25 / 1,702.26 / 44,100 |

External regret sample weights are one. Outcome regret correction weights
(prefix opponent/sampling reach times inverse sampled-action probability
and the suffix target/sampling ratio) are:

| Root | Correction median / p95 / max | External average weight median / p95 / max | Outcome average weight median / p95 / max |
|---|---:|---:|---:|
| 25 BB three-player | 7 / 147 / 686 | 49 / 1,256.85 / 28,812 | 49 / 2,646 / 57,624 |
| 5 BB three-player | 7 / 56 / 567 | 14 / 166.5 / 2,646 | 14 / 567 / 8,064 |
| 25 BB heads-up | 14 / 392 / 2,016 | 7 / 199 / 2,058 | 7 / 277.2 / 2,016 |

Outcome sampling still has substantially greater target variance. No exploration
change was made: measured external-sampling cost supports using it as the
primary generator. Average-policy importance weights remain variable in both
paths and need monitoring with learned strategies. Observed maxima are not
global bounds. Historical tournament-long targets remain recorded in
`outcome-diagnostics.json`; they are no longer neural training targets.

## Validation and readiness

- `.venv/bin/python -m unittest discover -s tests -p 'test_*.py'`: 93 passed (6.66 s).
- `node tests/run_browser_tests.mjs`: 64 passed; 37 hand and 10 tournament fixtures.
- `cd ui && npx tsc --noEmit`: passed.
- `cd ui && npm run lint`: passed.
- `cd ui && npm run build`: passed, static export.
- `node tests/ui_browser_smoke.mjs` with the newly exported random-weight cEV
  ONNX fixture: passed, no browser errors, 10 interactions, widths 1440/390/320.
  This fixture verifies inference wiring, not policy quality.

The direct Node strip-types invocation cannot resolve the tests' extensionless
TypeScript imports; the repository runner compiles them correctly and passes.
ONNX exporter deprecation and GRU tracing warnings remain visible.

**SMOKE-TRAIN READY.** External sampling is tractable for the measured per-hand
roots. No expensive training was launched. Broader deals, uneven stacks, later
blind levels, learned policies, sample diversity, neural approximation errors
and best-response/NashConv remain unmeasured here. Three-player CFR offers no
general two-player zero-sum Nash-convergence guarantee. Tournament simulation
may still last arbitrarily many hands; the hand boundary removes that horizon
from each learning target without claiming a bounded tournament duration.
