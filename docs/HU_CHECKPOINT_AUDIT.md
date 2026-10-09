# HU checkpoint audit — iteration 28

## Verdict

The checkpoint is structurally valid and reloadable, but its average policy is not accepted as a credible poker reference. Do not promote it based on iteration count or the narrow river best-response probe. No production source, training configuration, checkpoint or publication was changed by this audit. The concurrent three-player run was not interrupted.

## Reproduction and provenance

- Checkpoint: `runs/hu/checkpoint.pt`, SHA256 `5c0128677eed460beaf2989665673861022a1e2c2627186adfe5873d7a9abfa0`.
- Full runner reload validates envelope checksum, schema, configuration hash, model state, replay and sampler state. Source checkpoint unchanged after analysis: `True`.
- Run `PYTHONPATH=. .venv/bin/python -u runs/hu/audit/audit_checkpoint.py` from the repository root.
- Full outputs: `runs/hu/audit/report.json`; cProfile binaries and text summaries beside it.
- One CPU thread; full advantage replay inference; 24 named probes and all 1,326 combinations at two fixed roots; one diagnostic fitting epoch on 2,048 sampled records per model, on discarded model copies. No production training or export.
- Final audit execution: 17.21s including 14.09s checkpoint reconstruction. An initial 14.65s pass exposed lazy optimizer imports; the final pass warms imports before profiling and adds exhaustive preflop combo probes. Total diagnostic process work was under one minute. Concurrent training and profiling overhead limit absolute timing comparisons.

## Data coverage

| Dataset | Generated | Retained | Opening records | Opening classes |
|---|---:|---:|---:|---:|
| Advantage | 36,567 | 36,567 | 736 | 159 / 169 |
| Average strategy | 66,053 | 40,696 | 696 | 158 / 169 |

All opening records survived retention. The 10,000-record protected allocation is mostly unused; protection cannot create missing supervision. The other strategy stratum retained 40,000 of 65,357 records, with explicit inverse-inclusion weighting.

| Holding | Advantage train / held-out | Strategy train / held-out |
|---|---:|---:|
| AA | 5 / 1 | 1 / 2 |
| KK | 1 / 0 | 0 / 1 |
| AKs | 3 / 0 | 1 / 0 |
| AKo | 6 / 1 | 9 / 0 |
| 72o | 3 / 2 | 5 / 1 |
| 32o | 6 / 1 | 2 / 2 |

Counts pool all stacks, blind levels and contexts. They are not supervision at one consistent poker situation. Splits reproduce the final fit's seeded shuffle exactly. They are record-level splits, not independent public-root/trajectory validation.

There were 1,792 traversal roots, including 360 terminal roots (20.1%): 16 on-policy, 136 stratified and 208 synthetic. Terminal roots legitimately yield no decision records. There were therefore 1,432 nonterminal roots, matching 736 advantage plus 696 strategy opening records.

`stratified_recorded_opening` targets the correct recorded actor, but uses separately shuffled 1,326-combo cycles per player/position/dataset, reset for each frozen strategy iteration. With only 32 traversals/player/iteration, cycles remain far from complete. This mode does not guarantee all 169 classes in a small batch. Do not remove the per-profile reset without checking estimator semantics.

## Weight quality and fit

Average-policy effective sample size: 32,857.9 / 40,696. Largest weight share: 0.00435%; top ten: 0.0435%. Advantage ESS: 29,549.8 / 36,567. Extreme importance-weight concentration is not the current observed bottleneck.

Final weighted advantage MSE: train175.14, held-out180.39. Final average-policy KL: train0.2054, held-out0.2379. Targets and replay change each iteration, so the time series is not a frozen-data convergence study.

All legal predicted regrets are nonpositive on 40/736 retained openings (5.43%) and 4,348/36,567 retained advantage observations (11.89%). The normal uniform fallback still occurs, but it does not explain every bad opening. No fallback was modified.

The three AA strategy opening targets have iteration-weighted mean fold probability90.03%, across different contexts. This is evidence of poor historical supervision as well as scarce supervision; the average network is not learning from a trusted poker teacher. KK has no training-partition opening at all.

## Fixed-context policy probes

HU SB first decision, equal starting stacks, blinds0.5/1, first hand, no previous voluntary action. Results below average every exact combination of each class; all combinations within a class agree in these preflop roots.

| Holding | Average fold at5 BB | Average fold at37.5 BB |
|---|---:|---:|
| AA | 42.20% | 30.10% |
| KK | 14.85% | 31.99% |
| AKs | 21.22% | 32.24% |
| AKo | 23.97% | 37.74% |
| 72o | 7.61% | 15.56% |
| 32o | 10.07% | 13.24% |

The premium/weak-hand pattern is not credible. The current advantage-derived policy and historical average differ: at37.5 BB AA has zero fold under current regret matching, while at5 BB it folds75.77%. Thus neither more average fitting alone nor a single favorable AA probe establishes a repaired strategic learner.

Private-card reversal produced exactly zero average-policy difference on all24 named probes. The all-combo preflop checks found zero within-class suit variation. These are preflop checks, not a new exhaustive postflop symmetry certificate.

## Stability and narrow best response

Fixed-probe L1 drift: iteration20=0.243,25=0.293,28=0.378. It is not converged on these probes. The existing bounded river response gain improves from0.701 BB at iteration5 to0.0102 BB at iteration25, but this is only a36-node, four-hidden-deal river diagnostic. No iteration28 BR was recorded by the training schedule. It does not validate the opening policy or general exploitability. Existing scripted evaluations contain only three fixed hands per opponent and cannot establish playing strength.

## Performance

Measured iteration wall total: 1900.27s; fitting: 1847.89s (97.24%); generation: 16.17s (0.85%). Session console duration also includes checkpoint/metrics work outside the measured iteration timer.

Final iteration:125.22s total,122.44s fitting,0.67s generation. More worker processes would target a very small measured fraction. Larger fits alone would further reduce new supervision per wall-clock budget.

Warmed one-epoch cProfile on2,048 records/model:

| Component | Advantage | Average policy |
|---|---:|---:|
| Whole diagnostic fit, including validation | 0.248s | 0.238s |
| `encode_batch` cumulative | 0.122s | 0.119s |
| Backward cumulative | 0.045s | 0.036s |
| Adam step cumulative | 0.006s | 0.007s |

Encoding reconstructs tensors and decodes replay histories on repeated batch visits. It is a measured optimization candidate, not evidence of an already implemented speedup. These short fits include final validation and are not a direct extrapolation to50/20-epoch full replay runs.

## Recommended next experiment

Keep this checkpoint as an audit baseline, not a promoted policy. After the three-player run ends, measure a larger generation batch before another production fit, with an explicit target for nonterminal opening coverage in both datasets. Compare matched fixed-replay budgets separately for advantage and average, using grouped independent-root validation and action probes. Optimize repeated encoding only with numerical and resume parity checks. Retain corrected collectors, invariance and weighting semantics; do not force premium-hand action labels or change regret-matching fallback.

No new training budget, configuration change, source optimization or export was performed in this audit. General playing strength remains unvalidated.
