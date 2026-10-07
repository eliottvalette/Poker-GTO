# Three-player average collector audit

The partial-enumeration collector passes the exact own-reach reference and reduces observed weight concentration on bounded preflop roots. It remains an importance estimator with potentially large weights; this is not a full-game convergence or quality claim.

## Estimator and implementation

`Traversal.average_partial` samples the recorded player's actions from its current strategy, enumerates every action of one opponent, and uniformly samples actions of the remaining opponent. At a recorded-player decision it emits the current strategy with weight `1 / Q`, where `Q` contains only the remaining opponent's sampled prefix probabilities. Thus expected action mass is `own_reach * strategy`; own zero-probability branches need not be visited. Chance remains sampled through the root's normal shuffled deal.

The audit runs both opponent orientations and gives each one half weight. Each orientation separately estimates the same mass; the factor avoids doubling the mass relative to the old collector. Production can also select one orientation independently of the sampled deal/trajectory, but this audit validates the merged implementation. Budgets fail explicitly; no clipping or fabricated leaves are introduced.

The [Deep CFR paper](https://proceedings.mlr.press/v97/brown19b/brown19b.pdf) defines a two-player zero-sum setting and collects opponent strategy nodes during external sampling. The partial collector here is an independently checked extension, not an algorithm claimed by that paper.

## Exact validation

The existing fixed-deal three-player river tree has 25 nodes and 12 infosets. Three linearly weighted iterations use changing policies. The normalized expected average has zero observed bias for partial enumeration, versus `0.196331` maximum absolute bias for direct opponent-node collection.

Independent enumeration of the actual traversal's RNG branches checks **unnormalized action mass**, not merely a duplicate reach formula:

| Policy family | First orientation maximum error | Second orientation maximum error |
| --- | ---: | ---: |
| Changing positive strategies | 1.33e-15 | 1.78e-15 |
| Changing deterministic strategies, zero-probability actions | 0 | 0 |

HU exact checks and original collector comparisons also pass. Normalizing finite random numerator/denominator estimates can still introduce finite-sample ratio bias; zero exact inclusion bias does not remove that distinction.

## Bounded preflop diagnostics

Three fixed-deal roots use equal stacks of 1.5, 5, and 15 BB. Each collector runs 12 trials per player for each of three changing synthetic policies, with linear iteration weights. Partial enumeration uses both orientations. These are paired root/policy budgets, **not equal node budgets**. All raw emitted records, targets, weights, traversal node counts, timings, per-trial totals, and exact rows are retained in `three-player-collector-audit.json`. Each preflop record references a lossless observation key by its index into `preflop_infosets`.

| Stack | Collector | Records | Weight ESS | ESS / 1,000 nodes | Largest weight share | Nodes |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1.5 BB | Uniform importance | 135 | 56.5 | 124.4 | 5.24% | 454 |
| 1.5 BB | Partial enumeration | 600 | 343.6 | 167.6 | 1.37% | 2,051 |
| 5 BB | Uniform importance | 162 | 3.6 | 6.8 | 42.33% | 534 |
| 5 BB | Partial enumeration | 1,726 | 290.1 | 52.9 | 3.05% | 5,486 |
| 15 BB | Uniform importance | 181 | 4.1 | 7.0 | 39.24% | 592 |
| 15 BB | Partial enumeration | 4,043 | 393.4 | 32.3 | 1.94% | 12,181 |

Direct opponent-node collection has much flatter weights (ESS fractions 88–91%), but its failed exact-reference check rules out treating that as a valid improvement. Partial enumeration's 15-BB ESS fraction is still only 9.7%. Its node-normalized ESS improves by approximately 4.6 times there, while node work rises approximately 20.6 times for the paired collection budget. ESS measures weight concentration only; branch-correlated samples are not independent observations. No confidence interval or general variance guarantee follows from these small fixed-deal diagnostics.

## Integration recommendation

Use partial enumeration for the three-player average memory after exact tests pass; retain HU Algorithm 2 collection. Start with both orientations and half weights, account for their full node cost, and monitor concentration by street/player/root alongside fit metrics. Replay-generation metadata must identify the collector. Preserve the old collector as an explicit comparison, not a silent fallback. Benchmark learned policies and deeper roots before claiming the variance problem is solved or starting a large three-player run.

Reproduction: call `scripts.three_player_collector_audit.run_audit(trials=12)` programmatically and serialize its returned report. Focused regression command: `.venv/bin/python -m unittest tests.test_strategy_collectors -v`.
