# Strategy-chain audit — 2026-10-14

Status: COMPLETE for the bounded diagnosis. Production training, checkpoints, policies and configuration were not changed. Findings do not establish a successful repair.

## Sources and scope

- Actual OVH checkpoint extracts: HU #2873, run `792568bdb59040adbb1d7275e036793a`; 3-max #2494, run `682e4a7359b84ceeb281e50835973d04`.
- Matching sealed ONNX exports, complete preflop A/B replay, all specialist states excluding optimizer tensors, and recent detailed diagnostics (HU #2904, 3-max #2512).
- Artifacts: `runs/strategy-chain-audit/`. Importable reproduction: `training.chain_audit.audit(source, destination, samples=128)`. Exact extraction and follow-up probe scripts are retained beside the results.
- Public roots: HU SB 37.5 BB each; 3-max BTN 25 BB each. Hero's cards from the copied range were not supplied, so that exact blocker-conditioned posterior cannot be reconstructed.
- The Overview 3-max public root hashes to **test**, HU to **train**, under the actual production grouped split. Do not move the test root into training to manufacture acceptance.

## 1. Export and collection checks

All eight Average routes match checkpoint inference on 32 frozen test observations per route. Maximum probability discrepancy: **7.49e-7**. There is no evidence of an export or routing mismatch on these probes.

Nine collector/context-replay regression tests pass, including exact action masses and inverse-inclusion correction. This validates the tested estimator mechanics, not all possible root-distribution semantics or convergence in the full game. Historical producer snapshots are not retained for every replay iteration; their exact probabilities cannot be reconstructed retrospectively from the current A alone.

## 2. Direct failure in A

For 3-max BTN 25 BB, all **1,326** holdings produce identical raw regrets and identical regret-matched policies:

| Action | Probability |
|---|---:|
| Fold | 26.797% |
| Raise 4 BB | 24.476% |
| All-in | 48.727% |
| Other actions | 0% |

Every one of the head's 64 ReLU units is inactive. The largest preactivation across all holdings is -0.74988. Output is exactly the final-layer bias, with zero holding-dependent spread. The head therefore provides no gradient to its encoder for these inputs. Positive output biases mean this failure is **not** detected by the all-nonpositive-regrets fallback metric.

Across the 8,000 retained 3-max opening records, 77.11% have an entirely inactive head. Opening regret MSE is 124.67 on training against 124.32 for predicting zero, and 88.08 on test against 87.84 for predicting zero. This is not explained solely by an unseen Overview root.

HU does not show this complete head collapse. However, KK at the fixed 37.5 BB opening produces all-nonpositive regrets and the uniform legal fallback (including 14.29% fold). AA calls 100%; 72o folds 100%. This is a separate, context-dependent approximation failure.

## 3. Retention is no longer the same as effective supervision

Each opening class has 47/48 retained examples across stacks and public roots. AA has 34 A training records in HU and 38 in 3-max; B has 33 and 39 respectively. This repairs the previous zero/one-record class coverage but does not provide dense conditional coverage.

| Track | Opening slots / preflop replay | Opening weight in A | Opening weight in B |
|---|---:|---:|---:|
| HU | 8,000 / 20,000 | 7.83% | 4.13% |
| 3-max | 8,000 / 20,000 | 4.18% | **0.29%** |

The inverse-inclusion correction restores the source stream's weighted objective. Protected storage does not force the optimizer to prioritize openings. In 3-max B, the diagnostic context `facing_4bet_or_more` carries 74.13% of weighted loss mass. This classification counts previous raises and is not itself the reservoir's exclusive jam stratum.

Preflop B validation ESS is 59.23 across 2,380 retained records in 3-max. Context protection alone has not solved weighted optimization imbalance.

## 4. B is also an imperfect historical-average approximation

On retained opening states, decompose weighted KL into disagreement between historical targets for identical encoded states, and error between the empirical weighted-average target and B:

| Track | Historical-target disagreement | B vs empirical average | Total KL |
|---|---:|---:|---:|
| HU | 0.1264 | 0.5029 | 0.6293 |
| 3-max | 0.0291 | 0.5809 | 0.6100 |

There are 5,528 unique HU states and 6,917 unique 3-max states among 8,000 records. These are replay fidelity measurements, not a complete historical average at every unobserved information set. Most measured opening error is not removed by aggregating repeated-state historical targets.

At the fixed 3-max Overview root, B folds AA 17.79% and jams AA 38.54%, versus 33.82% jam for 72o. A's poor producing strategies and B's approximation error are separate problems.

## 5. Independent fixed-continuation action values

A first 128-world probe was followed by **1,024 worlds per track/model/holding**, seed 810719, using compatible uniform private deals and common random numbers across root actions. Every later player follows the specified frozen A or B policy. Terminal payouts use the canonical engine. These are best responses against specified continuations, not equilibrium values or cumulative historical regret targets.

| Track/model/holding | Mixed-policy EV | Best sampled action EV | Paired gain, BB/hand (descriptive 95%) |
|---|---:|---:|---:|
| HU A / KK | 8.55 | Raise 4 BB: 12.43 | +3.88 [3.18, 4.59] |
| 3-max A / AA | 11.16 | Call: 17.55 | +6.39 [5.27, 7.50] |
| 3-max A / 72o | -3.69 | Fold: 0 | +3.69 [2.78, 4.60] |
| 3-max B / AA | 14.11 | Jam: 17.70 | +3.59 [2.95, 4.23] |
| 3-max B / 72o | -3.17 | Fold: 0 | +3.17 [2.56, 3.79] |

Choosing the largest sampled EV introduces selection optimism; these intervals are descriptive and do not correct multiple action comparisons. All action estimates are retained. Do not compare A-continuation EVs directly with B-continuation EVs as if opponents were unchanged.

Independent tiny river games still favor B over A in bounded response gain (HU A 0.3333 vs B 0.0238 BB; 3-max A 0.4288 vs B 0.0318). The river checks do not validate preflop strength or full-game exploitability.

## 6. Failed repair probe retained

Equal frozen replay, grouped splits, seed 812014, batch 256, 128 optimizer updates, fresh Adam for both offline branches; same starting A weights. Compare current ReLU against replacing only the head activation with LeakyReLU(0.01).

- ReLU branch remains holding-independent at the fixed 3-max root; opening test MSE 88.08.
- Leaky branch regains output variation but opening test MSE worsens to 157.62; AA fold rises to 63.92%.
- Neither is accepted as a repair. Changing an activation on already-trained weights is not sufficient evidence for a new production architecture. These diagnostic branches were not exported or deployed.

## Budgets and next action

Remote checkpoint extraction: one CPU thread, reduced scheduling priority, sequential, 24.73 s / 25.65 s; maximum process RSS approximately 1.14 / 1.32 GiB. Existing training services stayed active. Local initial audits took 10.94 s HU and 14.69 s 3-max; the 12 strengthened rollout probes took 82.03 s total. Both 128-update fitting branches took under two seconds combined, excluding coverage/probe computation. No expensive production fitting was launched.

Next executable engineering action: test an explicit context-balanced supervised objective that preserves the weighted target within each information set, together with an initialization/activation design that retains gradients for opening inputs. Compare on the same untouched root groups, include exact-state target fidelity, head inactivity, A/B opening discrimination, and fixed-continuation EV. Preserve the validated collectors and the sixteen-specialist structure. Do not clip weights, silently change CFR semantics, train on test roots, or increase epochs as the sole correction. A successful candidate must repair both the inactive-head failure and the effective opening-supervision deficit before another production switch.
