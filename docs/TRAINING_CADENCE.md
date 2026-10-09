# Training cadence correction

## Scope and outcome

The previous 64-traversal HU / 50-epoch A / 20-epoch B loop was rejected. Checkpoints at `runs/hu/checkpoint.pt` (iteration28) and `runs/3max/checkpoint.pt` (iteration23) remain unchanged; their SHA256 values are retained in the comparison reports. No active three-player training process remained when inspected. No policy was published, no GPU used, and no new 30-minute run was launched.

Current entrypoints use new `runs/hu-batched` and `runs/3max-batched` outputs. These are fresh experiments; the old models are not implicitly loaded. Main settings:512 HU /384 three-player traversals per frozen strategy batch; A fit after each batch; B at batch1 and every third batch; warm weight initialization after the first fit; fresh Adam state each fit; at most4 epochs and800 updates per fitted network. On-policy root rollouts use current A regret matching so B cadence no longer determines their freshness.

## Implemented changes

- Explicit validated `fit_schedule`; fresh and warm initialization remain selectable independently for A/B.
- Update ceilings limit optimizer work even when replay grows. Final loss evaluation still scans the replay; this cost is reported, not hidden.
- Fit-local encoded tensors eliminate repeated decoding across updates; a256MiB tensor storage guard fails explicitly. This is not a bound on whole-process peak RSS.
- Generation arrives in32-task chunks under one frozen snapshot. Replay insertion order, collector semantics and weights are preserved. Failed chunks never publish the staged iteration. The generated-sample byte budget applies per transfer chunk, not to the total discarded stream across the batch.
- Logs report traversals, nodes, generation time, fit time and actual A/B updates. Metrics identify skipped B fits and the source iteration of its weights; skipped fixed-probe drift is unavailable rather than misleadingly zero.
- Checkpoint payload/schema remains compatible with original runs. New config keys are explicit and strict; old configurations retain their former semantics when loaded for audit. Resuming with different settings still fails rather than silently changing the experiment.

## Frozen-generation comparison

Serial CPU, same saved snapshot, nested balanced prefixes of a single sampled task schedule; generation results are discarded after counting. Root preparation time is recorded separately. These are samples under a fixed profile, not additional strategy improvements.

| Track | Traversals | Seconds | A openings / classes | B openings / classes |
|---|---:|---:|---:|---:|
| hu | 256 | 8.56 | 109 / 79 | 100 / 65 |
| hu | 512 | 19.17 | 211 / 122 | 209 / 103 |
| hu | 1024 | 36.30 | 416 / 146 | 422 / 141 |
| 3max | 96 | 3.40 | 28 / 25 | 56 / 25 |
| 3max | 192 | 6.40 | 61 / 49 | 122 / 49 |
| 3max | 384 | 13.19 | 139 / 87 | 278 / 87 |

Larger batches produced more information, not better openings per traversal. No full169-class guarantee is claimed. Three-player strategy counts include partial-enumeration sampling, not independent unique private hands. Terminal roots are still included; no biased rejection or sampler change was introduced.

## Fixed-replay fitting comparison

The retained checkpoint replay and seeded record split are identical within each dataset.200 and800 updates are separate budget measurements, not equal elapsed-time claims. Warm models have already seen historical replay, including records that may have belonged to earlier training splits: warm held-out results do not establish independent generalization. Both modes reset Adam. Fit time includes cache construction, validation and final train/held-out evaluation.

| Track | Dataset | Initialization | Updates | Cache | Seconds | Held-out loss |
|---|---|---|---:|---|---:|---:|
| hu | advantage | fresh | 200 | False | 2.25 | 424.045091 |
| hu | advantage | fresh | 200 | True | 1.98 | 424.045091 |
| hu | advantage | warm | 200 | True | 2.00 | 179.396968 |
| hu | advantage | fresh | 800 | True | 3.34 | 366.529144 |
| hu | advantage | warm | 800 | True | 3.42 | 178.660959 |
| hu | strategy | fresh | 200 | False | 3.15 | 0.513691 |
| hu | strategy | fresh | 200 | True | 2.41 | 0.513691 |
| hu | strategy | warm | 200 | True | 2.22 | 0.236220 |
| hu | strategy | fresh | 800 | True | 3.51 | 0.430070 |
| hu | strategy | warm | 800 | True | 3.40 | 0.235509 |
| 3max | advantage | fresh | 200 | False | 1.91 | 234.527635 |
| 3max | advantage | fresh | 200 | True | 1.76 | 234.527635 |
| 3max | advantage | warm | 200 | True | 1.75 | 104.823294 |
| 3max | advantage | fresh | 800 | True | 3.10 | 199.441858 |
| 3max | advantage | warm | 800 | True | 3.04 | 104.311324 |
| 3max | strategy | fresh | 200 | False | 2.16 | 0.379130 |
| 3max | strategy | fresh | 200 | True | 2.04 | 0.379130 |
| 3max | strategy | warm | 200 | True | 2.00 | 0.279534 |
| 3max | strategy | fresh | 800 | True | 3.41 | 0.343028 |
| 3max | strategy | warm | 800 | True | 3.33 | 0.283204 |

Cached versus uncached fresh fits produce exactly identical reported losses at200 updates. Unit tests also compare final weights, including variable history lengths and masks. Warm starts retain substantially lower frozen-replay errors at these small budgets; this justifies keeping a warm experimental path, not declaring it superior CFR convergence. Increasing warm B updates from200 to800 worsened the three-player held-out result.800 is a compute ceiling, not a demonstrated optimal setting.

## Actual full-config pilots

Two complete batches per track, four worker processes, actual generation/fitting/evaluation and checkpoint reload. The first batch includes worker startup. Short fit tests ran concurrently during part of the pilots, so timings are indicative local measurements, not isolated machine capacity benchmarks.

| Track | Batch | Traversals | Nodes | Wall seconds | Generation | Fit | A updates | B updates | A / B openings |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hu | 1 | 512 | 71192 | 14.98 | 6.11 | 6.60 | 668 | 800 | 212 / 208 |
| hu | 2 | 512 | 7888 | 9.91 | 2.17 | 3.65 | 760 | 0 | 217 / 205 |
| 3max | 1 | 384 | 80013 | 9.41 | 4.54 | 3.47 | 328 | 708 | 133 / 266 |
| 3max | 2 | 384 | 28895 | 9.24 | 3.88 | 2.07 | 384 | 0 | 124 / 248 |

The HU pilot generated413 strategy openings in24.89 measured iteration seconds, versus696 in1900.27 seconds in the historical run. This is a throughput comparison across different strategy trajectories, not a matched playing-strength experiment. Fitting occupied about41% of HU and30% of three-player pilot wall time. The falling HU node count between batches also shows that current learned strategies change traversal shape; more traversals alone do not certify useful strategic information.

## Local algorithmic reference

A deliberately small HU river game with two candidate holdings/player, explicit ranges, exact payouts and independent bounded best response. No neural fitting occurs inside local CFR. Costs are measured separately; these are equal iteration/node comparisons for the local variants, not equal CPU budgets across all engines. Neural-only cost includes materializing its strategy table at all reachable information sets.

| Mode | CFR iterations | Nodes | Seconds | Sum of bounded response gains |
|---|---:|---:|---:|---:|
| neural_only | 0 | 0 | 0.009930 | 0.465054 |
| local_without_prior | 10 | 428 | 0.006570 | 0.190957 |
| local_without_prior | 200 | 7268 | 0.022568 | 0.009548 |
| local_neural_prior | 10 | 428 | 0.008746 | 0.138575 |
| local_neural_prior | 200 | 7268 | 0.024526 | 0.006929 |

Local calculation materially improves this tiny reference over direct NN inference. It does not establish whole-game exploitability or that a learned prior is worth its cost broadly. The independent exhaustive current-profile regret diagnostic is also retained: on its two Hero holdings, A's squared discrepancies are180.97 and4.63, compared with zero-prediction baselines0.270 and0.00081. A estimates historical sampled regrets rather than exactly this current-profile instantaneous quantity, so these are warning diagnostics, not a correctly matched supervised generalization score.

## Validation and reproduction

Regression suites exercised schedule skipping, current-policy roots, bounded updates, exact cached/uncached fitting, variable histories/masks, streaming generation parity, deterministic checkpoint resume and the reviewed publication workflow.31 training/runner/workflow tests passed; a subsequent focused14-test run passed after final metadata changes. Earlier31 training/runner/Deep-CFR tests also passed. No browser code changed in this correction.

```python
from pathlib import Path
from training.cadence_experiment import compare, compare_reference, preflight

compare(Path("runs/hu/checkpoint.pt"), Path("runs/cadence-comparison/hu.json"))
compare(Path("runs/3max/checkpoint.pt"), Path("runs/cadence-comparison/3max.json"), track="3max")
compare_reference(Path("runs/hu/checkpoint.pt"), Path("runs/cadence-comparison/hu-reference.json"))
# preflight requires a deliberately chosen, nonexistent output directory:
# preflight("hu", Path("your-explicit-diagnostic-directory"))
```

Diagnostics were bounded:1024 HU plus384 three-player frozen traversals, ten capped fits per track,200-iteration tiny reference solves, and two actual batches per track. No further training session was started. Original checkpoint hashes were checked unchanged in both comparison reports.

## Limits

This correction establishes substantially better compute allocation and operational controls. It does not establish a strong poker model. Root-distribution conditioning, a calibrated high-coverage curriculum, independent trajectory splits and a statistically powered matched-compute playing-strength comparison have not been validated here. The sampler, action abstraction and regret fallback were intentionally preserved. The former checkpoint remains a bad reference; no new model has been accepted or exported. Large training remains a separate explicit decision.
