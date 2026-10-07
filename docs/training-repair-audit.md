# Training repair audit

## Implemented contracts

Feature schema 3 canonicalizes private-card order jointly with suit labels. It
compares first-occurrence suit normalization of both private-card orders and
keeps the lexicographically smaller complete card vector. The board breaks
equal-rank ties. The two private-card history tokens use that same order, and
all public card tokens use the selected suit mapping. Numeric features and the
network dimensions are unchanged. Python and browser implementations agree.

Feature-2 average policies remain readable using their explicitly declared
encoding. Feature-3 inference never silently reinterprets feature-2 compact
observations. New training checkpoints use contract 3 and reject old training
state; the active reservoir also rejects old feature records. Existing run
checkpoints and published policies were preserved. Starting a new substantive
run requires an explicitly chosen new output directory; the old configured run
directories are not migrated or overwritten.

HU external sampling now records strategy targets at opponent nodes during
the regret traversal, with unit collection weights. Linear iteration weights
remain in the fitting objective. Three-player training retains the previous
uniform-trajectory importance estimator. Metrics and checkpoints identify this
difference. Outcome sampling and root sampling are unchanged.

## Collector correctness

The comparison uses real engine river roots with 1.5-BB initial stacks, a fixed
deal, and three different strategy profiles. Full action enumeration contains
9 nodes / 4 infosets in HU and 25 nodes / 12 infosets in 3-max. At every infoset,
the reference is the iteration- and own-reach-weighted average. Tests also
enumerate every random choice in the actual production traversals and verify
their expected accumulated action masses against independent reach formulas.

| Collector | HU maximum exact bias | 3-max maximum exact bias |
| --- | ---: | ---: |
| Uniform trajectory with importance correction | 0 | 0 |
| Opponent nodes, Algorithm 2 style | 0 | 0.196331 |

The three-player result is an exact counterexample, not sampling noise. With
traverser `p` and recording player `i`, the inclusion probability contains the
reach of the third player as well as that of `i`. That extra factor changes
across iterations and distorts the average. A better weight ESS does not fix
this bias, so this candidate is not adopted for 3-max.

The Monte Carlo comparison additionally runs 256 trials per player and profile
with the production traversals. Complete per-infoset expectations, sampled
accumulations, coverage, and weight concentration are retained in
[strategy-collector-audit.json](strategy-collector-audit.json). This is a
conditional fixed-root estimator check, not a full-game equilibrium guarantee.
The reference algorithm is [Deep CFR Algorithm 2](https://proceedings.mlr.press/v97/brown19b/brown19b.pdf).

## Frozen replay fitting protocol

The sweep uses `runs/hu/checkpoint.pt` at iteration 412 and
`runs/3max/checkpoint.pt` at iteration 245. Source payload checksums, full
configuration, training/held-out indices, and seeds are retained in the report.
Each track fits its advantage and average-policy models separately. Every model
starts from the original next-iteration seed. Measurements at 2, 5, 10, 20 and
50 epochs share one uninterrupted Adam trajectory, with the same 80/20 split,
learning rate, batch size and uniform-shuffle objective. A regression test
verifies that measurement callbacks do not change the optimizer trajectory.

Feature schema 2 is deliberately retained for this experiment to isolate the
fitting budget. It does not measure the benefit of canonicalization. The
two-epoch average-policy losses reproduce the earlier optimizer comparison.
Every milestone retains model weights, complete outputs and probabilities for
all 1326 ascending-card-ID holdings and their reversed order, plus explicit
AA/KK/AKs/AKo/72o/32o probes. Private-card history tokens change with the cards.
Opening roots are balanced 37.5-BB HU and 25-BB 3-max states.

Complete artifacts are in `runs/fit-budget-audit/`, with the retained two-epoch
timing benchmark in `runs/fit-budget-preflight/`. Reproduce through the
programmatic `run_fit_budget_audit` function in `scripts/fit_budget_audit.py`,
using an explicit unused output directory. No CFR is run during this sweep.

### Fitting results

Each cell reports training / held-out loss. Advantage loss is masked MSE;
average-policy loss is weighted KL. They are not comparable across model kinds.

| Epochs | HU advantage MSE | HU average KL | 3-max advantage MSE | 3-max average KL |
| --- | ---: | ---: | ---: | ---: |
| 2 | 398.250 / 398.933 | 0.466 / 0.290 | 225.846 / 217.333 | 0.274 / 0.826 |
| 5 | 373.928 / 380.940 | 0.394 / 0.258 | 211.078 / 204.081 | 0.210 / 1.080 |
| 10 | 344.906 / 356.087 | 0.344 / 0.263 | 199.516 / 195.471 | 0.175 / 1.303 |
| 20 | 330.356 / 344.506 | 0.278 / 0.252 | 190.790 / 189.063 | 0.140 / 1.590 |
| 50 | 308.220 / 329.598 | 0.203 / 0.240 | 177.405 / 185.109 | 0.104 / 1.992 |

Both advantage fits improve held-out MSE with more updates: 17.4% in HU and
14.8% in 3-max from 2 to 50 epochs. This supports increasing their fit budgets,
but neither trajectory establishes convergence or separates all label noise
from approximation error. Regret-matched probes can jump when predicted
advantages cross zero; they are not monotone quality metrics.

The old HU average fit improves KL but still folds the explicit AA probe 57.4%
at 50 epochs. The old 3-max average fit shows worsening held-out loss at every
measured budget beyond two epochs while its training loss decreases. More
epochs alone therefore cannot repair these policies. Held-out strategy ESS is
only 4.27 in HU and 6.37 in 3-max; these validation losses are dominated by very
few records and do not support a robust global hyperparameter choice.

**No larger global fit budget is adopted.** A follow-up should measure advantage
and strategy budgets separately on the corrected replay. The 3-max collector
still needs a low-variance estimator that passes its exact average reference;
the direct Algorithm 2 candidate is ruled out by the counterexample above.
The root sampler and all other deferred model changes remain untouched.

Complete scalar results and artifact references are in
[fit-budget-audit.json](fit-budget-audit.json). All 20 milestone artifacts were
checked for complete 1326-holding predictions. The two-epoch weights of all
four fits exactly match their independent preflight runs.

## Bounded HU integration pilot

`runs/hu-repair-pilot/` retains three complete feature-3 checkpoints, the exact
configuration, scalar metrics, and both networks' full opening predictions at
every iteration. The pilot uses four traversals per player per iteration,
one CPU worker, the unchanged root mixture and replay capacities, and an
exploratory 20-epoch fit budget. This budget is not a convergence claim.

| Iteration | Strategy records | Weight ESS | Average AA fold | Average 72o fold | Fixed-probe drift |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 791 | 791.0 | 15.08% | 13.14% | — |
| 2 | 1182 | 1050.7 | 2.61% | 0.84% | 0.368 |
| 3 | 1372 | 1129.8 | 0.65% | 0.78% | 0.343 |

Maximum private-card order prediction difference is exactly zero for both
networks across all 1326 opening holdings at every iteration. The final average
policy's largest action spread across holdings is 29.91 percentage points.
Nevertheless AA and 72o still have similar action distributions: approximately
46.4% versus 44.2% all-in, respectively. Drift remains high. This pilot validates
the new collection/encoding pipeline, but **fails poker-quality acceptance**.
It is not published and does not justify scaling either training track.
The retained machine-readable report is [hu-repair-pilot.json](hu-repair-pilot.json).

Use `run_hu_repair_pilot` in `scripts/hu_repair_pilot.py` to reproduce this bounded
experiment with an explicit unused output directory. A substantially larger run
needs its own measured preflight and fit/coverage acceptance criteria.

## Verification

Tests cover combined suit permutations and private-card reversal across both
player counts and all streets, including pairs; complete Python/browser feature
parity; exact traversal expectations and the three-player counterexample;
deterministic fitting measurements; feature-version rejection; old-policy
inference and ONNX export; checkpoint resume; and training/publication workflows
in temporary directories. The browser suite passes 114 tests and TypeScript
checking passes. No published policy, legacy gzip policy, GPU configuration,
network size, root sampler, or weighted-replacement default was changed.
