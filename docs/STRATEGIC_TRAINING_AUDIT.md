# Strategic training audit — 2026-10-10

Status: **COMPLETE for the investigation and bounded experiments below. Production correction is not implemented or validated.** No training settings, running services, checkpoints or published policies were modified. Experimental weights are diagnostic artifacts only.

## Conclusion

The measured failure is a combination of poorly allocated supervision, approximation error in both learning stages, and severe strategy-estimator variance in 3-max. It is not explained by a display bug, a universally incorrect CFR payoff, or simply too few epochs.

The current scripts train Deep CFR: A determines traversal strategies, actual terminal chip calculations produce regret targets, and B approximates historical strategies. They do not train against independent hybrid local-solver reference solutions. Overview directly displays B. Having a functional hybrid engine therefore does not establish the quality of this exported policy.

The strongest findings are:

1. Protected openings occupy 20% of memory but only **0.78% of HU B's objective and 0.107% of 3-max B's objective**.
2. B demonstrably misrepresents some retained targets: on matched HU AA opening observations, historical target fold mass is **1.86%**, while B predicts **32.26%**.
3. A also produces strategically weak choices under independently sampled continuations. Improving B's fidelity alone is insufficient.
4. The 3-max B objective has **ESS 123.9 out of 50,000 records**. Its largest record is a river observation from iteration 280, carrying **8.48%** of total objective weight.
5. A controlled change improving B's replay loss **worsened 3-max 72o game outcomes** against unchanged opponents. This candidate is rejected as a production fix.

## 1. Exact sources and reproduction scope

The server's deployment and local base commit both resolve to `deccb334d7e84c56f49af39e369421fd92e1e792`. Audit scripts and the earlier UI changes are uncommitted local work outside the training implementation under inspection.

The copies came from `/var/lib/gto/{hu,3max}/checkpoint.pt`, each read through an open file descriptor. These are completed export-cycle checkpoints, **not necessarily the newest file selected by `resume.json`**. Subsequent inspection found active training services and resume checkpoints HU 850 / 3-max 1025. The earlier metrics-only inspection concerned HU 843 / 3-max 1014. Do not merge their measurements as if they were one snapshot.

| Audit input | HU | 3-max |
| --- | --- | --- |
| Checkpoint iteration | 825 | 994 |
| B's last fitted iteration | 825 | 993 |
| SHA-256 | `84bd5c1e0d0b192f22066a86b1d99fe3970146e14968d03e3ceb57dda665a384` | `ce73b9f1735bf1198f11a98b56082030b567b8dcc5a8105b8c8c57f0693b7a86` |
| A / B retained records | 50,000 / 50,000 | 50,000 / 50,000 |
| A / B protected openings | 10,000 / 10,000 | 10,000 / 10,000 |

The checked envelope, full configuration, metadata, per-record predictions, targets, weights, masks, numerical context, and fitting partitions are retained in `runs/strategic-training-audit/`. Input copies total approximately 565 MB. Analysis ran locally with one PyTorch thread; the VPS only served file reads. No GPU or new dependencies were used.

Primary source owners:

- `training/root_sampler.py:137`: evolving on-policy root distribution; `:162`: exploration stack/blind distribution.
- `cfr_solver.py:113`: external-sampling regrets; `:137`: partial-enumeration average collector.
- `ml/stratified_memory.py:73`: inverse-inclusion correction.
- `ml/train.py:118`: split, optimizer and bounded fitting; `:153`: fresh Adam.
- `ml/deep_cfr.py:335`: warm model weights and independent A/B cadence.
- `training/runner.py:103`: Deep CFR owner; `training/publication.py:158`: artifact publication, without a poker-quality promotion gate.
- `ui/src/components/PolicyOverview.tsx`: direct average-policy inference.

## 2. Coverage and objective allocation

`ProtectedReplay` uses independent opening/other reservoirs and divides retained weights by inclusion probability. This correctly restores the sampled stream's expected weighted loss total. A quota is therefore a variance/coverage mechanism, **not a 20% training-objective allocation**. Removing the correction without declaring a different objective would not be a neutral fix.

| Dataset | Opening objective mass | Overview-geometry records | Overview-geometry objective mass |
| --- | ---: | ---: | ---: |
| HU A | 1.5109% | 1,518 | 0.2378% |
| HU B | 0.7819% | 1,539 | 0.1252% |
| 3-max A | 3.1671% | 2,049 | 0.6997% |
| 3-max B | 0.1073% | 2,063 | 0.0237% |

Overview geometry means an unopened first decision, equal initial stacks of 37.5 BB in HU or 25 BB in 3-max, and physical big blind 1. It does not require identical hand-number features, so exact full neural-input grouping is reported separately.

| B records at Overview geometry | AA | KK | AKs | 22 | 72o |
| --- | ---: | ---: | ---: | ---: | ---: |
| HU | 7 | 13 | **0** | 4 | 17 |
| 3-max | 7 | 8 | 9 | 13 | 22 |

Full opening replay is also fragmented: HU has 8,223 distinct A inputs and 8,250 distinct B inputs among 10,000 openings; 3-max has 7,602 and 7,394. Sharing weights across nearby states is essential, but these counts do not establish that useful generalization has occurred.

### Root distribution

A deterministic probe independently drew 1,200 exploration roots per source and player count, seed 42, using `_exploration` directly. It excludes on-policy roots and does not estimate the full replay distribution.

| Source | Terminal | Minimum initial stack <2 BB | Minimum initial stack >=20 BB |
| --- | ---: | ---: | ---: |
| HU synthetic | 508 | 785 | 58 |
| HU stratified | 400 | 700 | 100 |
| 3-max synthetic | 0 | 917 | 40 |
| 3-max stratified | 0 | 800 | 100 |

Each exploration source has 25% of the configured mixture. The construction distributes 75 physical chips over six blind levels up to 32, often with strongly unequal stacks. This explains why nominal traversal counts do not measure 25–37.5 BB opening supervision. In 3-max, a very short third stack does not imply that the other players' side-pot game is equally short. Actual retained openings still include substantial deeper-stack coverage: approximately 47% of HU openings have minimum initial stacks at least 20 BB. The defect is not that every root is shallow; it is the mismatch between coverage counts, conditional coverage, and effective learning influence.

## 3. B compression error and A target quality are separate

For HU AA at Overview geometry, the seven B observations have weighted target fold probability 0.01862 and weighted predicted fold probability 0.32261. The UI-style exhaustive six-combo query predicts 0.32352. Current A folds AA at 0% there. This demonstrates a real compression/generalization defect in B; latest A and historical average are still different objects and should not generally agree.

However, the same historical B records assign substantial weak-hand aggression: HU 72o has mean target raise-3BB probability about 57%; 3-max 72o has mean target jam probability about 31%. Those records are samples of past A strategies, not independent poker reference labels.

A's aggregate opening MSE is 159.67 versus 167.90 for an all-zero predictor in HU, and 149.41 versus 146.64 in 3-max. This is concerning but **not proof that A learned nothing**: instantaneous regret targets are noisy, and the relevant conditional mean can be much smaller than individual realizations.

A also disagrees with some empirical historical target signs: among 14 retained 3-max 72o observations at Overview geometry, weighted jam target is -4.285 physical chips while A predicts +0.554; fold target is +2.001 versus predicted +0.210. The small conditional sample count prevents treating its empirical mean as exact ground truth. Independent rollout evidence below establishes the decision weakness without relying on that mean alone.

An exact within-replay decomposition verifies this distinction. Among HU A inputs represented at least five times, weighted target variance is 185.06 and excess squared error relative to each input's empirical mean is 30.97. Historical variation and chance noise dominate the raw loss in those repeated inputs. Empirical means from small groups are themselves uncertain; this decomposition is an in-sample diagnostic, not an independent optimal predictor.

Uniform fallback is active on only 2.45% of HU retained opening A observations and 2.18% in 3-max. It is not the universal explanation of current opening failures. The existing all-nonpositive regret-matching rule was left unchanged.

### Independent sampled decisions under frozen policies

For each checkpoint, A and B were evaluated separately. Each candidate root action was forced, then every seat followed that same frozen policy. Uniform compatible private deals, exact terminal settlements, 512 worlds per holding, seed 39101, paired chance/action streams. All values below are **BB per hand**, not BB/100.

| Policy / 72o root | Mixed-policy EV | Gain from folding instead | Approximate paired 95% interval |
| --- | ---: | ---: | --- |
| HU A | -2.685 | +2.185 | [1.363, 3.007] |
| HU B | -3.367 | +2.867 | [1.332, 4.402] |
| 3-max A | -4.195 | +4.195 | [3.109, 5.280] |
| 3-max B | -1.492 | +1.492 | [0.995, 1.990] |

This establishes exploitable one-decision behavior in these controlled cases. It does not rank A versus B generally: each row uses different continuation opponents. It is not full-game exploitability or a solved preflop equilibrium. The noisy 22 mixture alone is not sufficient evidence of a defect; several measured differences for 22 remain uncertain.

A approximates **historical weighted regrets**, not today's instantaneous action advantages. Therefore these current-policy rollouts diagnose strategic usefulness; they must not be mislabeled supervised regression ground truth for historical A.

## 4. 3-max variance is a separate severe problem

The partial-enumeration collector samples the recorded player's actions, enumerates one opponent and samples the other uniformly. The latter's prefix probability enters a reciprocal correction, `1/Q`; both orientations receive half weight. Exact small-tree tests validate the expected action masses. They do not bound long-history variance.

At checkpoint 994:

- Overall strategy ESS: **123.88 / 50,000**.
- Largest record share: **8.475%**; ten largest: **14.602%**.
- Largest record: iteration 280, river, initial stacks 28.5/24/22.5 BB, corrected sample weight 17,748,089.1 before iteration multiplication. Its full history and predictions are retained.
- River strategy ESS: **8.89**.
- The largest record is in the reconstructed held-out partition, not in the optimizer's training partition. Training ESS is **704.28**, and the largest training record contributes **2.074%** of training weight.
- A frozen 625-batch gradient pass gives B gradient norm median **1.244**, maximum **125.997**. HU B's corresponding median/max are **1.739 / 4.999**.

Removing the largest record *only for an ESS sensitivity calculation* raises ESS to 941.6; removing ten raises it to 1,976.9. No records were removed from fitting or production. Clipping importance weights would change the estimator and is not justified by this sensitivity result.

## 5. Bounded fitting ablations and rejected candidates

For each track and A/B, compare identical original replay and split with 800 updates, batch 64, learning rate 0.0003, one CPU thread:

- unchanged snapshot;
- warm weights, original objective;
- warm weights, openings assigned 20% of aggregate objective mass;
- fresh initialization, original objective;
- warm weights, original objective, gradient norm clipped to 1.

The opening-allocation arm explicitly changes the global approximation objective. Its per-opening multiplier is derived from actual effective weights; it does not claim to preserve the original weighted objective. No generated poker labels or hand-specific action rules were added.

Opening loss in the current replay held-out partition:

| Arm | HU A MSE | HU B KL | 3-max A MSE | 3-max B KL |
| --- | ---: | ---: | ---: | ---: |
| Unchanged | 156.479 | 0.69075 | 153.244 | 0.49500 |
| Warm, original | 156.409 | 0.69061 | 153.138 | 0.50869 |
| Warm, opening mass 20% | 154.627 | 0.64833 | 153.680 | 0.41842 |
| Fresh, original | 157.445 | 0.68959 | 155.111 | 0.68464 |
| Warm, norm clip 1 | 156.475 | 0.69578 | 154.104 | 0.48008 |

Interpretation:

- Repeating the current update recipe adds little opening-fit progress in this fixed replay.
- Allocating more objective mass to openings helps B fit its targets, especially 3-max; it does not solve A.
- Fresh initialization at this same small budget is not an accepted remedy: AA fold becomes 85.1% in HU A and 72.3% in 3-max A despite improved non-opening losses.
- Gradient clipping is not a demonstrated general cure. It modestly improves 3-max B's opening KL, but worsens several other diagnostics; clipped HU A folds AA at 24.2% versus 0% initially.
- These experiments do not determine the optimal architecture, optimizer state policy, or long-run convergence. In particular, they do not test persistent Adam moments across an entire new training run.

### Better fit did not consistently mean better decisions

The opening-mass candidate was compared with unchanged B against **unchanged B opponents**, using matched complete deals, independent seed 49101 and 1,024 deals per holding. Hero's entire hand uses the selected candidate. Exact paired outcomes are retained.

| Holding | HU candidate gain, BB/hand | 3-max candidate gain, BB/hand |
| --- | --- | --- |
| AA | +0.749 [0.268, 1.231] | +0.051 [-0.561, 0.663] |
| 22 | -0.052 [-0.593, 0.490] | -0.193 [-0.686, 0.300] |
| 72o | +0.044 [-0.518, 0.605] | **-1.104 [-1.714, -0.493]** |

The tested candidate is **not accepted for production**. Better reproduction of an imperfect historical policy can worsen strategic behavior. These are selected diagnostic roots and descriptive intervals, not multiple-testing-adjusted evidence of general strength.

## 6. Measurement and theory limitations

### Fitting semantics

The configuration says four epochs, but the 800-update cap at 40,000 training records and batch 64 allows only 1.28 passes. The metric correctly reports one completed epoch. Warm starts retain weights, but `ml/train.py:153` recreates Adam; optimizer moments do not persist. Whether preserving them is better remains unmeasured.

Splits are created from replay positions with a fixed seed, not independent trajectories. A seed-42 growth experiment shows that 1,627/4,000 held-out positions after growth from 10k to 20k records were previously training positions; for 20k to 50k, it is 3,163/10,000. Warm-start held-out loss is therefore a replay-fit statistic, not untouched longitudinal validation. At full fixed capacity, stable slots reduce crossings but do not remove prior exposure or correlated contexts. B at checkpoint 994 was last fitted at 993, so its reconstructed current partition is not literally its last training partition.

### What the literature actually supports

- [Deep CFR, Brown et al.](https://proceedings.mlr.press/v97/brown19b/brown19b.pdf), Algorithms 1–2, §5 and Theorem 1: approximation error and memory quality matter; iteration count alone does not remove a fitting-error floor. The paper used substantial fitting, gradient clipping and fresh initialization in its experiments. Its empirical initialization result is not a universal mandate for this workload. A models historical instantaneous regrets; B adds a separate averaging approximation. The convergence setting is two-player zero-sum. Our audit therefore separates replay fit, strategic deviations and theoretical conditions rather than treating a lower scalar loss as convergence.
- [MCCFR, Lanctot et al.](https://papers.nips.cc/paper_files/paper/2009/file/00411460f7c92d2124a67ea0f4cb5f85-Paper.pdf): appropriate sampling/corrections preserve regret-update expectations. Correct expectation does not imply low finite-sample variance. This supports exact collector tests and separate variance measurements.
- [Single Deep CFR, Steinberger](https://arxiv.org/pdf/1901.07621): eliminating B through retained historical strategies separates advantage approximation from average-policy compression. This motivates a bounded exact historical-opening table as a diagnostic, not an automatic migration or claim that current A is sufficient. A historical strategy sampled for a trajectory must be used consistently; independently resampling snapshots at every action is not the same construction.
- [VR-MCCFR, Schmid et al.](https://mlanctot.info/files/papers/aaai19-vrmccfr.pdf): corrected baselines can reduce variance of sampled value/regret estimators. That is relevant to noisy A supervision; it is not, by itself, a remedy for the 3-max strategy-memory reciprocal-prefix weights.
- [Pluribus, Brown and Sandholm](https://noambrown.github.io/papers/19-Science-Superhuman.pdf): successful multiplayer poker computation does not confer the generic two-player zero-sum guarantee on multiplayer CFR. 3-max acceptance must rest on correct estimates and measured robustness.

### Evolving roots change the averaging interpretation

Half the configured roots come from tournaments advanced under current A. Their public-root frequencies change over time without an explicit root-proposal correction. At a public root sampled with frequency `q_t(r)`, collected strategy masses are proportional to `t q_t(r) pi_i^t(I) sigma_t(I)`, rather than the fixed-root `t pi_i^t(I) sigma_t(I)` expression. A constant root frequency cancels; a changing one generally does not.

A two-iteration algebraic example makes the distinction explicit: always-fold then always-jam, own reach 1, linear weights 1 and 2. Equal root sampling gives fold 1/3. Root frequencies 0.9 and 0.1 instead give fold 0.9/(0.9+0.2)=81.8%. This is a local derivation, **not a measured attribution of present poker errors**. A stationary-root controlled comparison is required before calling this the dominant cause or applying guessed importance weights.

### What remains unproven

- No general poker-strength or equilibrium claim follows from these roots.
- No evidence yet makes model size the primary cause.
- No complete set of historical A snapshots survives locally in these checkpoints; producing-policy fidelity for every old record could not be independently reconstructed. Existing exact collector tests and stored targets are the available evidence.
- Opening action-EV probes do not validate every later-street decision or all Overview positions.
- Existing hourly evaluations measure B against a synthetic pool. They neither validate all A regrets nor evaluate live hybrid search. Publication currently checks artifact compatibility and freshness, not a statistically established champion policy.

## 7. Recommended correction order and acceptance gates

1. **Make quality measurements trustworthy before promoting newer policies.** Keep a fixed untouched public-root/trajectory suite, distinguish fitting loss from poker EV, and keep latest export separate from a quality-accepted reference. Gate on paired decision/gain evidence and conditional coverage, not iteration count. This is not a recommendation to hard-code AA/72o actions.
2. **Specify the training distribution explicitly.** Run a bounded stationary-root comparison covering the stack contexts actually served. Measure generated/retained examples and effective mass per class/context for A and B independently. Preserve the conditional estimator or explicitly document changed weighting. Do not infer completeness from 50,000 records or a 10,000-opening quota.
3. **Validate A's strategic signal.** Use exact tractable games and repeated frozen-profile estimates; compare historical conditional target means, estimator uncertainty and action deviations separately. Reduce conditional target noise before buying many more redundant fits. Acceptance requires smaller decision errors on independent deals, not merely lower raw MSE.
4. **Address 3-max estimator variance explicitly.** Compare candidate collection/sampling schemes against the same exact references and repeated-seed variance, then measure per-street/context ESS and gradients. Neither removing high-weight records nor switching to the biased opponent-node collector is acceptable as an undocumented fix.
5. **Then repair B compression.** Keep exact empirical historical strategies for affordable repeated states as an audit reference, and fit against representative conditional targets. Evaluate downstream play with fixed opponents. The 20%-mass ablation here is rejected as a ready-to-use configuration despite its improved KL.
6. **Only after those gates, choose optimizer budget/state and capacity.** Current tests do not justify a giant model, a return to systematic 50 epochs, or fresh initialization by default. Compare improvements at equal CPU and report independent quality.

No expensive new training is required to execute the first bounded comparisons. A full stationary-root/collector redesign is a separate implementation experiment, not secretly applied by this audit.

## 8. Validation, budgets and retained artifacts

| Experiment | Work | Measured local duration |
| --- | --- | --- |
| Full replay + three-stack A/B matrices | 200,000 records; every exact opening combo and reversal | HU 16.29 s; 3-max 17.54 s |
| Original / mass20 / fresh fits | 12 fits × 800 SGD updates | approximately 2.4–2.8 s per fit, excluding probes/load |
| Gradient clipping fits | 4 fits × 800 updates | per-model times retained in clipped-fit reports |
| Frozen gradient distributions | 4 × 625 batches; no optimizer updates | HU 13.10 s; 3-max 13.16 s |
| Frozen A/B action deviations | 43,008 forced-action continuations | 38.75 s |
| Matched refit evaluation | 12,288 complete policy hands | HU 6.73 s; 3-max 7.36 s |
| Root distribution probe | 4,800 exploration roots | 0.14 s |

Each experiment stayed below a few minutes. Aggregate fitting: 12,800 updates, far below the authorized 30-minute experimental training ceiling. A first diagnostic attempt was stopped after detecting repeated NPZ decompression inside the grouping loop; arrays were loaded once before the successful run. A Python-version-incompatible convenience hash call was replaced by streaming SHA-256. These diagnostic corrections did not alter any measured production algorithm.

26 relevant existing tests passed: `tests.test_strategy_collectors`, `tests.test_hybrid_coverage`, `tests.test_root_sampler`, `tests.test_shared_advantage`, `tests.test_hand_solver_boundary`. Three opening-diagnostic tests also passed (combo multiplicity, physical fold utilities, reproducibility and invalid-card rejection), and diagnostic modules compile. The per-input loss decomposition was independently checked to sum back to measured opening loss for both models/tracks. All source checkpoint hashes are retained and inputs remain immutable.

Importable reproduction:

```python
from pathlib import Path
from scripts.audit_training_checkpoint import (
    audit_checkpoint, frozen_replay_study, gradient_audit,
    clipped_fit_study, evaluate_refit,
)

root = Path("runs/strategic-training-audit")
for track in ("hu", "3max"):
    audit_checkpoint(root / f"{track}.pt", track, root)
    frozen_replay_study(root / f"{track}.pt", track, root, updates=800)
    gradient_audit(root / f"{track}.pt", track, root)
    clipped_fit_study(root / f"{track}.pt", track, root)
    evaluate_refit(root, track, samples=1024)
```

Artifacts include complete checkpoint inputs, configuration/provenance and hashes, complete per-record NPZ results, all 169-class matrices over all 1,326 combos at each queried stack, all repeated-input group summaries, all fitted diagnostic model weights, all gradient-batch measurements, action-EV reports, paired complete-hand outcomes, and sampling/split diagnostics. The earlier published-policy audit remains separate in `docs/PUBLISHED_PREFLOP_AUDIT.md`.

## Investigation ledger

All eight investigation questions are addressed: source provenance; theoretical estimator/root assumptions; weight and gradient concentration; A/B hand sensitivity; B target fidelity; independent frozen-profile decision tests; bounded fitting comparisons; and ranked corrections. Historical producer reconstruction, a stationary-root training comparison, persistent-optimizer experiments, and general playing-strength validation remain explicitly unmeasured. They are not represented as completed fixes.
