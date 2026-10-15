# Range-conditioned solver students

Status: IMPLEMENTED AND BOUNDED-TESTED LOCALLY. All sixteen specialists were fitted on bounded data; strategic quality remains UNACCEPTED. No production training, publication, or activation changes.

| Requirement | Status | Implementation and evidence |
|---|---|---|
| Exact suit-aware public ranges | DONE | `hybrid/solver_inputs.py`: 1,326 exact combination factors per player, padded third seat for HU; common canonicalization with private/public cards |
| Continuation assumptions | DONE | Explicit version vocabulary and decision depth; unknown profiles fail; river explicitly uses terminal continuation |
| Stable supervision semantics | DONE | `hybrid/knowledge.py:root_supervision`: root action advantages under the calculated average profile, in BB; no cumulative-regret imitation |
| Same state, different ranges | DONE | Exact facing-jam regression and 256-update controlled identifiability experiment |
| Diverse bounded curriculum | DONE, limited | `hybrid/range_student_experiments.py`: 224 games, 64 distinct public roots, all streets and both player counts |
| Independent specialist fitting/resume | DONE | `hybrid/distillation.py`: eight independent models per track, new checkpoint schema, all sixteen tested for deterministic optimizer resume |
| Direct/table/NN-assisted evaluation | DONE | Held-out roots, independent chance seed and larger forest, exact river response gains, matched iteration and repeated CPU comparisons |
| Profiling and artifacts | DONE | `runs/range-conditioned-students/`: full supervision, four checkpoints, report, CPU profile, contrast and CPU comparison |
| Production rollout | NOT REQUESTED | Existing OVH Deep CFR training and published artifacts unchanged |

## Input and target contracts

`exact-public-factors-root-v1` represents normalized exact-combination **factors**. Their joint meaning is the product conditioned on mutually exclusive cards; these are not independent posterior marginals. Hero's public range remains a distribution, not a delta at the queried holding. Only the candidate player's own hypothetical cards enter its observation. No simulator opponent holding or future deck is read.

Seat order is actor then clockwise. Suit canonicalization minimizes the visible cards first and the complete sparse factors second, resolving suits absent from the board and private hand. Tests cover all 24 suit permutations, reversed private cards, and renamed seats. Positive range weights that underflow float32 are rejected rather than erased silently.

Every query must match the declared public root, history, board, initial stacks and actor. Inputs include the immutable continuation profile identity and depth. Unknown profiles, untrained streets, and mismatched player counts raise explicit errors. Profile identities must version their behavior: changing a policy while retaining its version is unsupported. Chance seeds are estimator provenance, not strategic input features.

Targets use the root player's action EVs under the computed average profile:

`advantage(a) = (EV(a) - sum_b average_strategy(b) * EV(b)) / current_BB`.

B learns that computed average strategy. This is a distinct root distillation population, not Deep CFR's cumulative-regret objective. An A output is an approximate root action advantage, not a table of cumulative CFR regrets. No change was made to production Deep CFR target semantics.

Internal information-set records remain in the computed tables and remain readable by the original `supervision()` audit function. They are deliberately excluded from this fit: assigning initial priors to downstream observations without their range/reach context would recreate an input ambiguity. The root inference adapter explicitly uses a separately supplied downstream policy; it does not silently invoke a root model at internal nodes. A prior only initializes local solving; explicit CFR still produces the final computed strategy.

The existing state encoder is reused. Each specialist owns its state encoder, exact-range projection, normalized head, optimizer and update count. The no-range ablation uses the same architecture and initialization, masking the entire solving-context input. It is an ablation, not the old production checkpoint.

`solver-root-students-v2` checkpoints persist the input/target/feature/action schemas, continuation vocabulary, track, model and optimizer weights, update counters, seed, dataset hashes and unaccepted status. Old `solver-students-v1` weights are rejected explicitly. Production checkpoint and browser export schemas are unchanged; these experimental checkpoints cannot be published through that path.

## Bounded experiment

- Two tracks; four streets; four training, two validation and two test public roots per street: 64 public roots total.
- Each root has three candidate combinations per player and two different nonuniform prior factors. Non-river games use uniform and passive continuations; river trees settle exactly. This produces 224 finite games.
- All variants of a public root stay in the same split; fitting rejects exact public-root split collisions. Stacks, boards, actor positions and first-open/facing-bet contexts vary. River stacks remain deliberately shallow to bound exhaustive trees.
- Teacher: 64 CFR iterations, at most 64 sampled worlds, depth one outside river, 3,000,000-node cap. Independent comparison: 128 iterations, 128 worlds and a different seed. The latter is a stronger approximation, **not exact non-river truth**.
- Each A/B specialist receives 128 updates, batch 64, Adam 0.0003, one Torch thread. Four roots provide only 48 training records per non-river specialist and 24 per river specialist, including repeated range/profile variants. Record counts are not independent roots.
- Full study: 232.13 seconds under cProfile; peak process RSS 358.48 MiB. HU/3-max generation took 33.96/55.96 seconds. Fitting took 2.65–3.32 seconds per variant/track, including checkpoint writing.
- Each full eight-model checkpoint is 27.04 MB, including Adam. The main study artifacts occupy about 129 MB; contrast and the migrated-demo regression add separate files. These are local experiments, not Supabase uploads.

## Identifiability result

One exact facing-jam spot, identical observable input and candidate holding, two opposite opponent factors:

| Model | Call vs strong range | Call vs weak range | Mean target KL |
|---|---:|---:|---:|
| Exact solver target | 0.25% | 99.75% | — |
| Observation-only student | 50.00% | 50.00% | 0.67567 |
| Range-conditioned student | 1.06% | 98.98% | 0.00437 |

Both variants used 256 updates on these two examples. This establishes that the missing information prevented representing both targets. It is a training-set identifiability demonstration, **not held-out strategic validation**. Independent tests calculate fold/call utilities directly from canonical settlement and verify the action-advantage labels.

## Held-out specialist measurements

Test A MSE and B KL against the stored finite-game targets; baseline means identical model with context masked. Lower is better.

| Track / street | A baseline | A conditioned | B baseline | B conditioned |
|---|---:|---:|---:|---:|
| HU preflop | 2.7730 | 2.7584 | 1.5197 | 1.5195 |
| HU flop | 2.5507 | 2.5311 | 2.5205 | 2.5028 |
| HU turn | 5.1441 | 5.1500 | 1.7016 | 1.6729 |
| HU river | 0.1915 | 0.1883 | 0.3477 | 0.3520 |
| 3-max preflop | 16.5549 | 16.6431 | 1.9996 | 1.9803 |
| 3-max flop | 16.8727 | 16.9687 | 2.4337 | 2.4288 |
| 3-max turn | 8.5802 | 8.6446 | 2.1340 | 2.1312 |
| 3-max river | 0.3159 | 0.3252 | 0.4360 | 0.4270 |

No consistent strategic improvement is established. Against the independent larger-forest action values, mean root-response shortfall was 1.2423/1.2424 BB for HU without/with context, and 2.1582/2.1537 BB for 3-max. Reusing the low-budget calculated table gave 1.0909 and 1.5210 BB. These are one-step deviations under specified continuations, not full-game exploitability or realized tournament gains.

The teacher itself is noisy outside river. Mean low-versus-independent-higher-budget action-EV RMSE:

| Track | Preflop | Flop | Turn | River |
|---|---:|---:|---:|---:|
| HU | 2.171 | 1.947 | 1.594 | 0.035 |
| 3-max | 3.558 | 4.136 | 3.400 | 0.029 |

More fitting cannot repair these target disagreements. Sparse independent roots and finite-forest uncertainty remain major limitations.

## CPU and reuse

The first timing pass showed a cold-run anomaly (8 iterations slower than 32). The final CPU benchmark therefore records five repetitions and uses medians, with 8/16/24/32/64/128-iteration candidates. Each method is selected by its highest feasible iteration count under the same measured CPU ceiling, not by selecting the best test result.

| Exact river example | CPU ceiling | Direct CFR | Root prior + CFR |
|---|---:|---:|---:|
| HU | 11.86 ms | 32 iterations; response gain 0.06425 BB | 24 iterations; 0.08284 BB |
| 3-max | 83.27 ms | 32 iterations; response gain 0.10243 BB | 24 iterations; 0.13512 BB |

This is a discrete cost/quality curve on two examples, not a universal advantage claim or a production wall-clock scheduler. At matched iterations, priors sometimes help slightly; their inference cost does not earn acceptance here. Candidate values reproduce exactly across timing repetitions.

Cached strategies require the exact root/prior and fail on unsupported infosets. Lookup is substantially cheaper than resolving these games, but initial calculation and storage are not free; generation cost is retained separately.

Profiling: solving accounts for approximately 204 of 232 seconds under cProfile; constructing rollout forests and safe observations dominates. Full observation creation during synthetic continuation rollouts is a measured future optimization target. No game-rule or observation-validation shortcut was introduced.

## Corrections and limitations

The first curriculum attempt failed because its postflop bet selector inspected preflop action-ID spelling. It now uses the canonical `SolverAction.category`, and the complete experiment succeeded. Failed partial outputs were replaced at the exact experiment destination before any model fitting; production data was never touched.

All sixteen specialists are IMPLEMENTED and received bounded optimization. None is strategically VALIDATED or ACCEPTED. The correction solves the missing-input contract, not general poker strength. Broader preflop ranges, deeper river trees, multi-step range/reach-conditioned internal-node learning, calibrated real-player continuations, uncertainty-aware model use and large independent root coverage remain outside the demonstrated quality scope.

## Reproduction and next action

```python
from pathlib import Path
from examples.range_conditioned_students import demonstrate

# Refuses an existing destination; no deployment or publication.
report = demonstrate(Path("runs/my-range-student-study"))
```

Next executable action: increase **teacher precision and independent public-root coverage before optimizer budget**. First compare 64/256/1024 worlds on fixed roots with independent seeds, retain per-action disagreement, then select a bounded broader curriculum. A 16/4/4-root-per-street train/validation/test curriculum produces 672 games; linear extrapolation of measured generation alone is roughly 4.5 minutes under the current profiler at 64 worlds, before stronger references. Raising chance samples increases this cost and needs a new preflight. This is a planning estimate, not a production benchmark or authorization for a large training job.

References: [DeepStack](https://arxiv.org/abs/1701.01724) conditions value prediction on public state and player ranges. [Depth-limited solving](https://papers.nips.cc/paper/7993-depth-limited-solving-for-imperfect-information-games) explains why fixed continuation assumptions are not generally safe imperfect-information solving. No safe-resolving guarantee is claimed here.

## Teacher-label stability follow-up

The next precision diagnostic was executed on one held-out turn root per track, three independent seeds per budget, 32 CFR iterations and depth one. Full action-value vectors are retained in `label-stability.json`.

| Turn example | 64-world budget | 256-world budget | 1,024-world budget |
|---|---:|---:|---:|
| HU cross-seed RMS standard deviation | 0.7708 BB | 0.4799 BB | 0.3414 BB |
| 3-max cross-seed RMS standard deviation | 1.9777 BB | 0.9145 BB | 0.4568 BB |

The 18 solves used approximately 14.94 CPU seconds in total. Counts in `actual_worlds` account for per-deal rounding and exhaustive turn-river switching: budget is not assumed to equal actual samples. Exhaustive river cards can still have noisy policy rollout actions. This measures dispersion, not bias or error against ground truth; two roots cannot calibrate all poker contexts.

Result: increasing teacher precision is useful on these examples. Do not spend a larger fitting budget on the same noisy labels. Next: extend this precision check to a fixed independent selection of preflop/flop/turn roots, then preflight a broader labeled curriculum at the selected sample budgets. No large run was launched.

## Regression verification

44 relevant Python tests passed across solver inputs/authority, exact river, chance search, hybrid search/safety, Deep CFR and validated strategy collectors. Added guards cover split leakage before any model mutation and an explicit encoded-dataset RAM budget (256 MiB by default). Deterministic optimizer resume is tested for both tracks, all sixteen specialists. The migrated original solver demonstration also ran end-to-end at two-update smoke budgets; its outputs are retained separately under `legacy-demo-regression/`.

No browser feature/export contract changed in this task, so browser validation results from the prior task are not presented as newly rerun tests.
