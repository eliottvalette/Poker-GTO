# Phase 2 Experiments

> Historical evidence: the old models and `artifacts/` were explicitly deleted during training finalization. Numerical results below describe previous bounded experiments, not currently deployed models. See [TRAINING.md](TRAINING.md) for the current configuration, empty catalog and launch/export workflow.

Status: bounded experiments complete; general playing strength remains unvalidated. See the roadmap for the complete implementation contract.

Historical Phase 1 reports were explicitly deleted; they are not present evidence. New measurements will be recorded independently.

## Baseline v1

`artifacts/phase2/evaluation/baseline.json`: 24 independent public roots, HU/3-max, all four streets, 3 stack regions (3–45 BB), asymmetric stacks and quarter-chip off-tree raises. Candidate 8/32 paired samples; independent reference 128 samples. Uniform continuation action-EV MSE averaged 9.173 chip² across the 48 root/budget rows, with no budget failures. Repeated budgets share roots and must not be treated as 48 independent roots for inference. References are the stated passive/aggressive synthetic teachers, not equilibrium or human play.

## Initial session and reference validation

Four new substantive tests pass: complete multi-street HU/3-max hands with actual 2.25 raise, analytic joint hand/profile posterior, atomic impossible observation, hidden/future-card isolation, accepted finite-game reference tables. HU and 3-max 500-iteration reference response gains are 0.003819 and 0.019239 on the original bounded river cases. These small references are deliberately scoped, not claims of full-game quality.

A routing defect was identified: supplied behavior could be bypassed by automatic small-river CFR. Explicit `response` mode now preserves exploitative assumptions. A reference hierarchy retains exact validated information-set tables and an explicitly approximate, observable-equity-based conservative fallback. Profile parameters describe synthetic populations, not calibrated humans.

## Bounded quality loop, cycle 1

The hand-conditioned teacher comparison uses the same 24 frozen test roots and independent simulation streams for references. At 32 samples, uniform MSE=10.126, conservative=3.901, known-profile=1.420 chip². Root-clustered intervals and per-root action losses are in `hand-conditioned.json`. Do not confuse knowledge of the synthetic teacher with calibrated inference about real humans.

Independent-pilot adaptive allocation reduced some action-selection losses but increased EV MSE to 2.009. It remains opt-in. A 480-hand matched experiment has zero workflow failures; paired realized-chip intervals are retained in `matched-hands.json`. Sixteen tournaments per count provide only wide descriptive uncertainty; the trained objective remains per-hand chips.

All continuation variants remain disabled. Diversity and suit-aware ranges alone did not produce useful value accuracy within the 20/50-epoch bounded fit. Behavior models are separately keyed by profile and count, with independent trajectory validation, OOD tests and exact optimizer resume. Fitting invalidates a previous acceptance gate until it is recomputed.

Actual browser smoke: HU 8 and three-player 12 observed actions reach settlement with persistent posteriors; three exact Python fixtures have zero measured EV/strategy difference. Pure compatible-deal sampling cache preserves seeded output and reduces measured 1000-sample time from 75.73 to 2.89 ms.

## Final cycle and acceptance measurements

Status: completed bounded experiments. Artifact paths in this section are relative to `artifacts/phase2/`. Physical chip squared errors are not normalized tournament-win objectives.

| Comparison | Budget / independent units | Measured result | Decision |
|---|---|---|---|
| Hand-conditioned reference assumptions | 24 held-out roots; 32 candidate / 128 reference worlds | EV MSE uniform 10.1258, conservative 3.9014, known profile 1.4202; action loss 0.9965 / 0.4941 / 0.2870 | Replace exclusive uniform default with explicit computed fallback; disclose assumptions |
| Root-paired MSE improvement | Same 24 roots | Conservative minus uniform −6.2244, normal 95% CI [−11.0279, −1.4209]; known profile −8.7056 [−13.7793, −3.6319] | Measured controlled EV improvement, not general playing strength |
| Wide/polarized ranges | Additional full-range variant of 24 roots; 32/128 worlds | MSE 5.1262 / 4.0449 / 4.6580; action loss 0.6467 / 0.3438 / 0.2412 | Benefit depends on distribution and metric |
| Adaptive allocation | Independent pilot; 32 total worlds | MSE 2.0088 vs fixed 1.4202; action loss 0.1898 vs 0.2870; 17,813 vs 19,491 nodes | Retain opt-in; no unconditional improvement claim |
| Learned behavior online | Same fixed held-out roots and 32 worlds | MSE 5.8661; action loss 0.5742; explicit known-profile reference MSE 1.4202 | Do not promote NN as default |
| Posterior reconstruction | 384 conditional-action rows, including off-tree interpolation | Mean TV uniform 0.1019 vs learned 0.1051 | Learned prediction acceptance does not imply better posteriors |
| Sampled-world robustness | HU/3-max × 4 seeds × 16/64/256 worlds, 50 CFR iterations; independent exhaustive-turn 150-iteration reference | HU MSE 0.01219 / 0.00773 / 0.00057; three-player 0.02486 / 0.02791 / 0.00507 | Larger forests help here overall, not monotonically |
| Finite river references | 4 roots per count; 5/50/500 iterations, published NN prior at 50 | Mean summed bounded response gain 0.39525 / 0.06544 / 0.02468; NN prior at 50: 0.06173 | Store only tables passing declared 0.05 gain threshold; no full-game guarantee |

### Realized chips and tournaments

`evaluation/matched-hands.json` and `matched-hands-32-samples.json` each contain 480 complete hands: 16 matched private-deal seeds × 5 profiles × 2 player counts × 3 candidates. Priors are explicitly narrow and include the generated deal; opponent identities are controlled assumptions. Maximum 80 decisions per query sequence and 50,000 search nodes per decision; no workflow failures. Root worlds are paired between candidates, but divergent actions lead to different subsequent histories.

At 8 samples, reference-search minus uniform-search was −0.2449 chips/hand, 95% normal seed-clustered CI [−0.6064, 0.1167]; known-profile search difference −0.1761 [−0.5789, 0.2267]. At 32 samples the differences were −0.1658 [−0.5017, 0.1702] and +0.1766 [−0.0665, 0.4196]. **Every realized-improvement interval contains zero.** Increasing compute is not a substitute for reliable opponent assumptions. `realized-paired-summary.json` preserves the grouped calculation.

Sixteen actual tournaments per player count were also executed with conservative versus aggressive policy play: observed win fractions 0.50 HU and 0.1875 three-player. These are descriptive checks of the tournament lifecycle, not a full-hybrid tournament-strength experiment. Normal intervals can extend outside [0,1] at this sample size and are labeled as such; do not interpret their endpoints as possible probabilities.

### Learning, coverage and failed models

`behavior/report.json`: two profiles × two counts × widths 32/96; 12 complete varied-stack training trajectories, independent six-hand validation and four-hand push/fold OOD sets. Epoch checkpoints 20/50 plus one exact-resume step. Labels are full conditional teacher distributions, not invented one-hot actions. Log loss, Brier score, reliability bins, OOD metrics, labels, optimizer state and ONNX parity are retained. Six variants pass the narrow prediction gate, both aggressive three-player variants fail. Downstream EV/posterior results above prevent default promotion.

`values/report.json`: 24 independent public roots per count, up to two jointly feasible queried holdings per root, 64 independent rollout samples per label. Compare 169-class and suit-aware exact-combination features at widths 32/96, epochs 20/50. Held-out MSE remains roughly 6–10 chip², above the declared 0.01 raw-value ceiling. `values-aggressive/report.json` repeats with 12 roots per count under an aggressive continuation profile; MSE roughly 13–21 chip². All 16 variants are rejected, including models that improve over a weak mean/table baseline. Label generation took about 26.91 and 14.60 seconds respectively; the conservative model fitting sweep took approximately 4.2 seconds. No rejected weights are enabled online.

`evaluation/value-exact-comparison.json` additionally compares class169 predictions, 64 solved-policy rollouts and exact terminal expectations under independent 500-iteration river policies. A solved policy's terminal expectation is exact for that policy, not an equilibrium certificate. The models were trained on another continuation profile, so this is an explicit transfer/OOD diagnostic. A failed raw-value gate precludes live action-selection promotion; no learned-leaf benefit is fabricated.

`fixed-replay/`: 26 advantage / 34 strategy examples, bounded 20/50/100 epoch studies. Advantage held-out MSE 0.6590 / 0.6084 / 0.6906; average KL 0.000547 / 0.000675 / 0.000576. This tiny fit checks the workflow, not representative poker supervision. The original all-nonpositive regret fallback is unchanged.

`retention/report.json`: 32 seeds, 1,000-example stream with 1% openings, total capacity 100, opening quota 10. Ordinary reservoirs lose every opening in 25% of seeds; protected reservoirs in 0%. The inverse-inclusion total-error interval contains zero (mean 41.625, CI [−102.184, 185.434]). The exact constant-target correction and deterministic saved continuation also pass tests. `protected-training/report.json` subsequently runs the actual two-track TrainingRunner for two iterations with quota 25%, capacity 20 and recorded-opening stratification; uninterrupted/resumed metrics and weights agree. This is optional allocation, not a new strategy collector or evidence of sufficient per-class coverage.

### Runtime measurements

| Component | Measured local time | Scope |
|---|---|---|
| Build 1,326-combo range | 1.129 ms | Ten repetitions |
| Bayesian factor update | 0.016 ms | Narrow range, 100 repetitions |
| Exact turn equity | 0.870 ms | Narrow reference |
| MC turn equity | 3.526 ms | 128 samples |
| Hand clone | 0.0077 ms | 1,000 repetitions |
| HU CFR build + 100 iterations | 18.933 ms | 3,668 visited nodes |
| Joint chance sample | 0.0129 ms | Narrow reference |
| Neural behavior inference | 0.698 ms | Small model, 20 repetitions |
| Replay generation | 0.485 ms | Small reference |
| Full three-player compatible sampling | 75.726 → 2.890 ms / 1,000 deals | Cached immutable cumulative distributions; same seeded stream |
| Static worker exact HU river / 3-max river / turn | 5.6 / 14.9 / 69.5 ms | Matching Python fixture values/probabilities, maximum error 0 |
| Actual broad browser rollout | 67.7 ms HU / 101.1 ms 3-max | 16 worlds, 557 / 665 nodes; cache/hardware dependent |

Source reports: `evaluation/components.json`, `sampling-profile.json`, learning reports and `browser-static-smoke-initial.json`. The final warning-enabled browser rerun is `browser-static-smoke.json`; execution times vary between runs. Component microbenchmarks must not be extrapolated to dense multiway CFR. Continuation inference latency is measured separately in the exact-value comparison. Worker numbers measure execution, not a guaranteed end-to-end UI service level.

### Defects corrected during the loop

- Explicit response mode prevents the automatic small-river CFR path from ignoring exploitative behavior inputs.
- Jointly infeasible factor holdings are excluded from continuation label generation using compatible marginal support.
- Fitting invalidates model acceptance; the post-resume epoch is validated again rather than retaining a stale gate.
- Tournament experiments now use the canonical RNG injection API.
- Browser range replay accounts for actions before the first Hero turn and multiple intervening bots; clone-only renders no longer reset the session. Request IDs, transition guards and cancellation discard stale worker results.
- Deleted legacy diagnostic artifact paths became explicit checkpoint arguments rather than guessed fallback locations.
- Large-profile evidence sampling with zero sampled likelihood raises an insufficient-evidence error; it does not silently delete a possibly feasible hypothesis.
- Protected replay is integrated behind an explicit config key with allocation validation and exact checkpoint resume; default reservoir schema and published artifacts stay unchanged.

## Statistical limits

The same public roots are repeated across budgets and candidates. Inference clusters roots or private-deal seeds rather than treating actions or hands under different policies as independent. Normal 95% intervals are approximate, not small-sample guarantees. Finite reference-world error remains in the EV metrics. No real-human calibration, unrestricted exploitability estimate or statistically established general playing-strength improvement is claimed.

The independent exact-profile value probes contained 16 holding/root rows: HU learned MSE 4.0272 versus 64-rollout MSE 0.0000113; three-player 4.1688 versus 0.000949. Mean inference cost was 0.610 / 0.502 ms, versus 73.322 / 113.337 ms for 64 rollouts. Faster prediction did not compensate for transfer error, so learned leaves remain disabled. Final browser rollout results also explicitly warn when the top-action gap is unresolved under the approximate paired 95% sampling threshold.
