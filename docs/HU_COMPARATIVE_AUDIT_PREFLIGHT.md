# HU comparative checkpoint audit: preflight

## Status and available evidence

Inspection date: 2026-10-24. The requested comparison around iterations 1,000,
1,300 and 1,590 cannot currently be performed from the identified artifacts.
No independent strategic evaluation, advantage relabelling or fitting-budget
ablation has run. This document records availability, existing measurements and
the required protocol, not a completed strategic audit.

Read-only SSH inspection of `51.81.202.115` found these complete HU v3 inputs:

| Input | Bytes | Availability at inspection |
| --- | ---: | --- |
| `selfplay-v3/checkpoints/iteration_001595.pt` | 268,159,000 | Numbered checkpoint |
| `selfplay-v3/checkpoints/iteration_001600.pt` | 268,166,872 | Numbered checkpoint; copied locally |
| `selfplay-v3/checkpoint.pt` | 268,155,224 | Resume pointer reported iteration 1,602 |

The process remained active; inventory and diagnostic reads were not an atomic
snapshot. All `.pt` paths under `/var/lib/gto` were inventoried. No HU v3
checkpoints near 1,000 or 1,300 were found. A local `runs/` filename inventory
also found no matching historical v3 inputs. This does not rule out external
backups. Other retained HU experiments have different run identities, sampling
or model contracts and cannot silently substitute for these missing iterations.

The runtime retains two numbered checkpoints and one stable checkpoint. Its 256
diagnostic files covered iterations 1,348–1,603 during the second inspection.
Diagnostics contain summary statistics, not recoverable historical A/B weights.
Exports contain B only. Neither can reconstruct the requested earlier checkpoints.

Local diagnostic artifacts (ignored by Git):

- `runs/hu-comparative-audit/preflight.json`: complete inspected checkpoint-path
  inventory and selected metric fields from three existing diagnostic records.
- `runs/hu-comparative-audit/inputs/iteration_001600.pt`: complete immutable audit
  input copied from the numbered source, including A, B and replay.
- `runs/hu-comparative-audit/inputs/manifest.json`: file SHA256, source path,
  run identity, configuration and envelope-validation scope.

No service, training configuration, publication or remote retention was changed.
The existing working-tree edit to `docs/CONTINUOUS_SELFPLAY.md` was preserved.

## Existing measurements, not independent validation

The first, middle and last retained diagnostic records were inspected. These
three deterministic selections are descriptive samples, not timing confidence
intervals or a complete trend analysis.

| Iteration | Generation s | Fitting A+B s | Iteration wall s | A held-out MSE | B held-out KL |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1,348 | 14.00 | 72.22 | 102.05 | 288.5403 | 0.54325 |
| 1,476 | 28.09 | 75.71 | 120.75 | 281.1770 | 0.52361 |
| 1,603 | 7.97 | 76.28 | 100.42 | 282.5518 | 0.49325 |

All six fits completed 2,048 optimizer updates: 2,048 for A **and** 2,048 for B,
not 2,048 combined. The last fitting share was 75.96% of measured iteration wall
time. The wall timer also includes work outside generation and fitting; it does
not include every operational cost of saving, exporting and publication.

Each network fitted 40,000 records and evaluated 10,000 held-out records from a
50,000-record replay. `ml/train.py` partitions individual records, after applying
the decision-balanced objective. These are changing replay populations with
historical labels. Loss differences do not isolate strategic improvement or the
benefit of 2,048 rather than 256 updates. They are not measurements on new deals.

At iteration 1,603, existing preflop retained-record counters were:

| Context | A replay | B replay |
| --- | ---: | ---: |
| First opening | 25,000 | 25,000 |
| Facing open | 334 | 1,015 |
| Facing 3-bet | 607 | 536 |
| Facing 4-bet or more | 1,335 | 1,283 |
| Checked to | 53 | 175 |

These contexts pool stacks, holdings and histories. They are neither independent
root counts nor effective sample sizes. `decision_context` uses raises in the
observable history; labels are the implementation's categories, not independently
verified annotations of every betting situation. `short_stack` and `jam_fold`
counters overlap other categories and must not be summed into a distribution.
Traversal record frequency also differs from the probability of visiting a
branch in ordinary play because traverser actions are enumerated.

## Required comparative protocol

### 1. Common independent evaluation population

Freeze exact checkpoint identities and configurations before generating a new
suite. Save the full deal manifest, hidden worlds, action RNG streams and complete
per-deal results. Exclude deals from the hourly schedule by comparing canonical
physical deal identities, not merely by choosing a different seed. Reconstruct
the known hourly schedule and inspect retained evaluation provenance for any
other seed offsets before claiming non-overlap.

Use the same deals, seats, stacks and fixed opponent pool for all candidates.
Report paired changes in BB/100 for A-derived and B policies separately, with
uncertainty clustered by independent deal group. Also evaluate all candidates on
the same fixed information sets: matched deals alone do not keep later decision
states identical when policies choose different actions. Keep naturally reached
states and deliberately forced rare-branch probes in separately reported panels.

Do not reuse `evaluate_policy(..., seed_offset=...)` unchanged: its main hands
respect the offset, but its river probe always starts from seed 73,219 and uses
hidden-world seeds 99,001–99,003. That river probe would reuse hourly deals.
A separate audit suite must own every randomness source, including river probes.
Fixed-pool results and small river trees are conditional diagnostics, not full-game
exploitability or evidence of GTO convergence.

### 2. Independently recalculated action advantages

Use an evaluator separate from the training traversal/target collector. For each
fixed information set, force each legal action and roll out frozen continuation
policies on paired hidden worlds; use exact enumeration in small tractable river
subgames. Return per-action values, uncertainty, action ranking, sign stability and
the EV loss of the policy mixture relative to the best tested action.

For a fixed policy sigma, the immediate deviation advantage is
`Q_sigma(I,a) - sum_b sigma(I,b) * Q_sigma(I,b)`. Hidden worlds after a raise must
respect the declared reach-conditioned range; uniform redeterminization is a
different intervention and must be labelled as such. A single sampled opponent
holding must not be treated as a perfect-information oracle for an infoset.

The raw A network is fitted to weighted historical sampled advantages. Its
distance to the current-policy deviation advantage is **not**, by itself, its
supervised approximation error. A true independent historical target audit needs
the generating policies, historical weighting/visitation contract and compatible
worlds. Three sparse checkpoints do not recover every historical teacher. If that
history is unavailable, report current-policy decision loss and historical replay
fit separately, with independent historical A error explicitly unavailable.

### 3. B fidelity to A-generated strategies

Measure weighted KL and total variation against the stored A-generated strategy
targets, stratified by source `model_version`, street and decision context. Report
the exact decision-balanced training objective and ordinary reach-weighted results
separately. Existing record-split metrics are diagnostic replay fit only.

Independent generalization requires fresh common infosets and the historical
teacher mixture with the correct iteration/reach weights. For cohorts whose A
weights exist, regenerate targets and validate provenance (`model_version =
iteration - 1`). Report missing historical mass. A comparison of B to current A,
or to an unweighted average of only three A checkpoints, cannot establish fidelity
to the historical target. Disagreement with current A can be averaging lag.

### 4. Visited and rare situations

For each checkpoint distinguish fresh self-play visitation, generated traversal
records, retained replay, and weighted fitting exposure. Report unique root groups
and trajectories, raw counts, normalized probabilities, effective sample sizes,
record ages and missing strata. Cross street, effective stack, position, raise
depth, legal-action set and public-board texture. Measure both actual-play branch
frequency and performance on fixed forced branches; a branch becoming rare is not
automatically a strategic error. Aggregate windows of independent roots to avoid
interpreting one iteration's traversal-size fluctuation as a trend.

### 5. Benefit and cost of 2,048 updates

On each frozen full replay, fit fresh copies with budgets 256, 512, 1,024 and 2,048.
Hold initialization, replay partition, sampling stream, architecture, learning
rate and objective fixed within each seed. Use three paired seeds. Evaluate every
budget on the same independent strategic suite as well as supervised validation.
Do not compare unrelated production checkpoints as a causal update-budget test.

The existing `fit` API can reproduce matched starts through explicit seeds and
`max_updates`. It cannot combine `decision_balance=True` with explicit grouped
partitions, nor request measurements at update milestones. Any addition must
preserve the production sampling/objective contract and verify matched-prefix
weights. Do not silently replace grouped validation with a record split.

Record encoding, optimizer, validation, evaluation and total wall/CPU time
separately, plus peak RSS. Report incremental strategic gain and supervised error
reduction per additional second. Fixed-replay ablation isolates fitting; an
equal-wall-time end-to-end branch experiment is additionally needed to establish
whether longer fits outweigh fewer fresh traversals over a training run.

## Compute preflight and remaining dependency

Three inputs of the observed size require approximately 0.75 GiB before audit
outputs. Load checkpoints sequentially and run CPU fits one at a time; a local
load/fit pilot must establish peak memory and timing before a full run.
The four-budget, three-seed, two-network, three-checkpoint ablation is 72 fits and
69,120 optimizer updates if every cap is reached. Repeated encoding and independent
evaluation add cost. Production timing above is not an isolated local benchmark
and cannot safely determine the full audit duration.

The immediate dependency is the location of external HU v3 archives near 1,000
and 1,300. Without them, the historical three-checkpoint comparison is not
recoverable from the inspected storage. A current-checkpoint diagnosis or a future
comparison with explicitly retained milestones would be a different experiment.

Methodological reference: [Deep CFR, algorithms 1–2 and the historical strategy
average](https://proceedings.mlr.press/v97/brown19b/brown19b.pdf). The repository's
decision balancing is an additional approximation objective and should not be
represented as the paper's unchanged stream-weighted objective.
