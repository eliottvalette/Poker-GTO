# First training audit

## Verdict

The training/checkpoint/export pipeline runs, but the first policies are not yet
credible poker reference policies. Do not scale the same training configuration
or start opponent modeling/local resolving on the strength of these results.
The observed weakness exists in the models, not just the range visualization.

The published policies match completed runs: 3-max iteration 245 and HU iteration
412. CPU PyTorch/ONNX differences on the fixed probes are below 6e-8. The browser
also displayed the same AA probabilities as Python. No checkpoint or published
policy was replaced by this audit or its fitting comparisons.

## Policy probes

At balanced initial preflop roots, compare the same public state across all 1326
hero holdings. The following figures use the average policy, not the advantage
network, with ascending card-ID ordering matching the Overview combo query.

| Published policy | Fold AA | Fold 72o | Largest single-action spread across all combos |
| --- | ---: | ---: | ---: |
| 3-max, BTN, 25 BB each | 4.38% | 5.80% | 7.22 percentage points |
| HU, SB/button, 37.5 BB each | 37.25% | 35.94% | 6.05 percentage points |

Outputs are not literally identical. Their weak hand discrimination and large
HU AA fold frequency are unhealthy sanity-check results. Reversing hero-card
order also changes predictions: up to 4.03 points on the tested 3-max 72o
holding. Suit normalization exists, but private-card order is not normalized.
This is avoidable representation sensitivity; changing that encoding requires
an explicit feature/model migration rather than altering inputs to old models.

## Why the average-policy training is unstable

The current average collector samples every action uniformly along one complete
trajectory. It stores a strategy with weight `own_reach / sample_reach`.
Training multiplies that weight by the outer iteration. Long, rarely sampled
prefixes can therefore have enormous relative influence.

Measured on the retained replay:

| Diagnostic | 3-max | HU |
| --- | ---: | ---: |
| Retained advantage samples | 30000 | 20000 |
| Retained strategy samples | 10000 | 10000 |
| Strategy weight ESS | 52.14 | 34.68 |
| Largest strategy sample share | 7.80% | 14.90% |
| Top 10 strategy sample share | 36.82% | 33.16% |
| Strategy training-partition ESS | 50.07 | 83.53 |
| Strategy held-out-partition ESS | 6.37 | 4.27 |

ESS is `(sum(w))^2 / sum(w^2)`, a weight-concentration diagnostic, not a count of
independent poker observations. Correlation between traversal samples makes it
insufficient as an uncertainty estimate. The held-out losses in particular have
very little effective support. The largest full-replay HU record happens to be
in the held-out partition, not the training partition.

There is also a state-coverage problem. The advantage memories contain only 518
3-max BTN opening decisions and 274 HU SB opening decisions, across all stacks,
blind levels and hand numbers. Thus each exact opening hand/context has sparse
advantage supervision. In strategy replay, 3001 BTN openings carry just 0.14%
of total 3-max loss weight; 4386 HU SB openings carry 1.36%. A high count of
preflop records does not imply a high contribution to the weighted loss.

The current 3-max advantage model itself folds AA about 48% in the tested opening
root. Consequently the poor average model is not the only issue: generation is
also driven by a weak advantage approximation. High regret MSE alone cannot
separate underfitting from Monte Carlo label noise; the failed poker probes and
coverage/concentration measurements are the stronger evidence.

Both networks are fitted from scratch for only two epochs per outer iteration.
Later iterations are not additional epochs on the previous network weights.
On the last 20 iterations, fixed-probe policy drift median remains about 0.179
in 3-max and 0.219 in HU. This does not establish convergence.

The original [Deep CFR Algorithm 2](https://proceedings.mlr.press/v97/brown19b/brown19b.pdf)
collects strategy records at opponent nodes during external sampling, with
iteration weighting. Its collection path differs from the current separate
uniform-trajectory, reach-corrected collector. That difference merits a controlled
variance/correctness comparison, particularly in HU; multiplayer averaging needs
its own justification. The current estimator's concentration is measured, not
proof that its expectation is wrong.

## Controlled optimizer comparison

`fit(..., sampling="weighted_replacement")` now provides an experimental option:
select minibatch records with probability proportional to their iteration/reach
weight, then average their unweighted pointwise losses. Its expected gradient
estimates the same weighted objective as uniform sampling with weighted losses.
Weights and targets are never clipped or replaced. Default training remains
`uniform_shuffle`.

Two offline fits per track used the same actual replay, split, initialization,
batch size and two-epoch budget. Model seed was the next-iteration initialization
seed; these are diagnostic fits, not reproductions of the published weights.

| Held-out weighted KL | Uniform shuffle | Weighted replacement |
| --- | ---: | ---: |
| 3-max | 0.826 | 1.251 |
| HU | 0.290 | 0.235 |

The candidate is not adopted: it worsens 3-max validation and still folds HU AA
about 47%. Changing minibatch sampling alone does not cure the problem. Default
fits now report replay/training/held-out ESS and dominant weight shares so the
problem remains observable on later runs.

## Can the gz policy pretrain the new models?

`policy/avg_policy.json.gz` contains 709056 packed-state records with five actions,
169 hand classes, coarse public-state buckets and visit counts. It lacks exact
postflop boards, full public action histories, discrete raise-size targets and a
player-count field. Its former game/action contract differs from the current one.

It does contain a useful coarse hand signal. Aggregated preflop BTN records fold
AA about 23.2%, KK about 21.9% and 32o about 56.4%. Those averages mix betting
contexts; they are not opening frequencies or proof of a high-quality reference.

A legitimate reuse is pretraining card representations with an auxiliary
fold/passive/aggressive classification objective on preflop hand classes. That
prior can initialize card embeddings in each fresh fit, with explicit source hash
and initialization metadata. It should be compared against random initialization.
Initializing only iteration zero would lose the prior at the next fresh fit.

Do not turn the old generic RAISE into an arbitrary modern raise size, fabricate
histories/boards to fill the encoder, infer regret targets from probabilities or
pretend the gz supplies HU-specific policies. Direct 13-action distillation is
not defined by this data. Its packed format must stay outside the active solver
and browser state paths. Pretraining may help hand representation; it cannot
repair the current average replay weighting/coverage by itself.

## Next learning work, in order

1. Compare strategy collectors with a small exact-reference game and fixed real
   roots; measure weighted ESS, state coverage and per-infoset average correctness.
2. Add fixed opening-hand probes and replay-weight diagnostics to run acceptance.
   Existing fit metrics now include ESS; retain exact probes for AA/KK/AK/72/32.
3. Compare fitting budget, deterministic preflop rank descriptors and a versioned
   card-prior initialization using fixed replay and held-out cases. Address
   irrelevant card ordering in the encoder contract.
4. Run another bounded pilot only after those comparisons are healthy. Preserve
   the per-hand objective, external-sampling default, modest architecture and
   separate 3-max/HU tracks.

No importance clipping, difficult-sample dropping, GPU compute, large network,
old packed solver state or online resolver was introduced.

## UI changes

Overview is now a preflop reference matrix with compact track/position buttons,
proportional cells, container-relative hand labels and a selected-hand detail
panel. Custom stacks, board inputs, deal seeds and betting continuations stay in
Cas précis. All views still autoload the same published 3-max/HU policies.
Rem-based controls scale on wide screens; there is no browser-zoom dependency.

Measured in the 3024x1542 browser viewport: base text 19.56px, hand labels 24px,
cells approximately 95x82px, zero select fields in Overview and no horizontal
page overflow. The screenshot is `docs/assets/policy-overview.jpg`.

## Retained evidence

- `first-policy-probes.json`: full probe probabilities and all-combo spread.
- `first-policy-replay-audit.json`: replay coverage, weighted mass, ESS and current
  advantage-model probes.
- `first-training-metrics-audit.json`: complete scalar metric trajectories.
- `first-policy-card-order-audit.json`: holding-order sensitivity.
- `policy-fit-comparison.json`: both controlled fitting outcomes.
- `legacy-pretraining-audit.json`: old gz record count and selected aggregated
  preflop hand-class distributions.
