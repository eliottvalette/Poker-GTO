# HU coverage preflight and feature-v4 fit calibration

This is a bounded measurement run, not a validated poker policy. Two CFR iterations used four workers, 32 traversals per player, batch size 64, two fit epochs per model, the existing root mixture, stratified private cards and unchanged max_nodes=20000. No medium pilot was launched by this experiment.

## Measured replay and coverage

| Iteration | Generation seconds | Advantage records | Advantage opening records retained | Opening hand classes | Exact SB combos generated, cumulative |
|---|---:|---:|---:|---:|---:|
| 1 | 0.876 | 1523 | 31 | 31 | 31 |
| 2 | 0.763 | 3079 | 54 | 50 | 54 |

The final strategy replay has 5765 records, including 48 opening decisions and 41 hand classes. Exact combos count raw traverser-targeted SB roots across both seat IDs; canonical replay card IDs cannot recover exact raw suits. Sample sizes average 4020 bytes for advantage and 4008 bytes for strategy; worker peak RSS is about 191 MiB. Two preflight checkpoints and full 1326-combo probes are retained in `runs/hu-coverage-preflight`.

## Frozen replay calibration

The immutable source is `runs/hu-coverage-preflight/checkpoints/iteration_000002.pt`, SHA256 `08fc9228fbe18e37cdb8c880dfd1ed1930c98280a995a5e990b266bf75eda03e`.

Each model uses one fresh initialization and one continuous Adam trajectory with fixed record-level split and minibatch RNG seed, measured at epochs 2/5/10/20/50. The exact training/held-out indices are retained. Batch 256 was faster than 64 in both one-epoch throughput measurements; this comparison includes validation and changes the SGD update count, so it is not an optimization-equivalence claim.

| Epoch | Advantage held-out MSE | Average held-out KL |
|---|---:|---:|
| 2 | 283.843 | 0.262544 |
| 5 | 276.121 | 0.185661 |
| 10 | 246.256 | 0.139512 |
| 20 | 201.336 | 0.101140 |
| 50 | 125.360 | 0.059418 |

Held-out effective sample sizes are 554.9 and 1035.6. The exploratory earliest-within-5%-of-minimum rule selects 50 epochs independently for both networks. Both minima occur at the last measurement: convergence is not established. Record-level splitting can retain correlations between samples from the same traversal, so these losses do not establish independent-game generalization.

All ten model weights and all 1326 exact-combo outputs/probabilities, including reversed-card predictions, are retained in `runs/hu-coverage-calibration`. Card reversal is exactly invariant at every milestone. At epoch 50 the average policy has AA fold 1.68%, AA jam 42.37%, 72o jam 31.20%; this is more differentiation but does not establish strategic correctness. The advantage policy jams AA 49.91% and 72o 47.24%, but folds AA 0% versus 72o 45.83%: similar jam frequencies conceal a large difference in the complete action distribution. Its class-level total-variation separations are 0.473 for AA/72o and 0.391 for KK/32o; the average network reaches only 0.112 and 0.055, below the provisional 0.15 gate.

## Medium-run feasibility

The unchanged targets are at least six CFR iterations, 3000 retained SB opening advantage records, 1000 distinct raw SB combos generated, and 150/169 retained hand classes. Observed opening density implies 593 traversals/player/iteration and advantage capacity 213820 with 25% capacity margin. Total traversal budget is 7116. The 3000-opening occupancy reference gives roughly 1187 unique exact combos under independent uniform dealing, not a coverage guarantee.

| Strategy capacity | Expected strategy openings retained | Nominal seconds | Conservative scenario seconds | Estimated six checkpoint disk bytes |
|---|---:|---:|---:|---:|
| 10000 | 83 | 1548 | 3095 | 2.45 billion |
| 20000 | 167 | 1671 | 3342 | 2.67 billion |
| 50000 | 416 | 2040 | 4080 | 3.33 billion |

For the unchanged 10000 strategy capacity, nominal components are generation 97s, fit 1412s, root generation 6s and checkpoints 32s. Checkpoint time is extrapolated from a measured 32.76 MB complete serialization/atomic write taking 0.429s. Evaluation and probe overhead are outside this component model. Fit timing includes calibration milestone/probe overhead, and scales linearly with replay size; larger replay throughput and later-policy tree growth remain unmeasured. The conservative scenario doubles component times; it is not a proven upper bound. RAM is estimated at 4.4 GiB for strategy capacity 10000, including staged replay and worker allowances, below the 24 GiB machine limit.

The 10000/20000 candidates fit 30 minutes only nominally. Neither has convincing conservative headroom; 50000 exceeds the nominal budget. Small average replay capacities also retain very few opening targets. This experiment does not relax coverage, iteration or fit requirements to force a run into the time limit.

Validated candidate configurations are `runs/hu-coverage-preflight/medium-strategy-{10000,20000,50000}-config.json`; their common destination is the explicitly unlaunched `runs/hu-coverage-medium`. They use aggregate generation budget 657535505 bytes, per-traversal sample budget 122725000 bytes and advantage replay budget 1284523650 bytes, derived from observed bytes with explicit headroom. The original maximum node/depth limits remain unchanged and fail explicitly rather than discarding samples.

Machine-readable reports: `docs/hu-coverage-preflight.json`, `docs/hu-coverage-calibration.json`, `docs/hu-coverage-medium-estimate.json`. Source implementation: `scripts/hu_coverage_experiment.py`; selection-rule tests: `tests/test_hu_coverage_experiment.py`.

## Provisional acceptance protocol for a separately approved medium run

Record all 1326 exact-combo predictions at the same fixed root after each iteration. Aggregate exact combos into holding classes with their true multiplicities. Require at least 3000 retained SB opening advantage records, at least 1000 unique raw SB combinations generated, and at least 150 retained advantage hand classes. Require an exactly zero card-order probability gap, TV(AA,72o) >= 0.15 and TV(KK,32o) >= 0.15, with AA folding less than 72o and KK folding less than 32o. TV is one half of the L1 distance between class-average action distributions.

Collect at least three comparable fixed-root policy drift and bounded-best-response measurements, including early and late iterations. Require late fixed-probe drift below early drift and late bounded-BR gain below early gain. Keep the bounded-BR hidden-deal/root set identical. Missing measurements leave a gate unvalidated; a two-iteration preflight cannot pass these trend gates. These are provisional engineering gates, not an EV or equilibrium guarantee. Calibration probe gate values are retained for every fit milestone in the calibration JSON; this assesses card signal, not CFR progress.

**Execution decision:** no medium run was launched. The 30-minute ceiling lacks conservative headroom for the unchanged targets and measured 50/50-epoch fit budgets. The 50000 strategy replay alternative improves opening coverage but already exceeds the nominal runtime budget.
