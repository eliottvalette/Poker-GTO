# Street network experiments

Architecture implementation: COMPLETE. All sixteen specialists received bounded CFR supervision and optimization. General strategic quality: NOT VALIDATED. No experiment was activated or uploaded.

## Protocol and independent groups

The authoritative cohort is `runs/street-network-migration/canonical-groups-{hu,3max}-seed{101,202}`. Each run uses five frozen-profile cycles, 64 traversals/player/cycle, 64 optimizer updates per eligible specialist, one CPU worker and one PyTorch thread. Diagnostic schedules fit both families every cycle with eight new samples required. Each street reservoir retains at most 2,000 records; each family is bounded to 8,000 records / 64 MiB of conservative sample accounting.

This diagnostic schedule is deliberately different from the new substantial-run configuration (A every cycle, B every third cycle, 128 new samples required). A schedule regression verifies that B stays cold for the first two cycles and trains on the third.

Public-root grouping hashes public history, initial stacks, blinds and board, with suit canonicalization and button-relative seat identities. Private cards and traversal seeds are excluded. The deterministic hash split is 80% training / 10% validation / 10% test by root group. All descendants of a root stay together. Test partitions are measured, never optimized. Related trajectories still share a root and are not independent statistical replications.

Earlier `*-seed*` and `recorded-context-*` cohorts remain as preliminary evidence. The first exposed incorrect stratification of the initial rather than recorded actor; the second used physical suit/seat identities in group hashes. Both were corrected. Final claims and tables below use canonical groups. Lifecycle profiling uses the recorded-context cohort solely to measure computation, not held-out generalization.

## All sixteen specialists — seed 101

A loss is iteration/reach-weighted legal-action regret MSE in the existing utility units; B loss is weighted target-to-model KL. These losses measure reproduction of recorded targets, not optimal poker decisions. The shared baseline is evaluated on the exact same grouped test records; it was not trained on the same experiment distribution.

| Track | Model | Generated | Retained | Retained trajectories | Weight ESS | Updates | Test loss | Shared test loss |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| hu | A PREFLOP | 614 | 614 | 205 | 501.2 | 320 | 84.1355 | 367.6355 |
| hu | A FLOP | 1356 | 1356 | 230 | 1057.1 | 320 | 198.0019 | 330.5912 |
| hu | A TURN | 2365 | 2000 | 259 | 1510.2 | 320 | 203.4839 | 479.7394 |
| hu | A RIVER | 3408 | 2000 | 281 | 1508.7 | 320 | 348.4892 | 419.6050 |
| hu | B PREFLOP | 895 | 895 | 222 | 690.4 | 320 | 0.2993 | 0.7944 |
| hu | B FLOP | 2388 | 2000 | 224 | 1481.7 | 320 | 0.2666 | 0.7779 |
| hu | B TURN | 4112 | 2000 | 243 | 1462.2 | 320 | 0.2057 | 1.0018 |
| hu | B RIVER | 5578 | 2000 | 266 | 1463.4 | 320 | 0.1495 | 1.0777 |
| 3max | A PREFLOP | 1070 | 1070 | 340 | 889.1 | 320 | 697.4172 | 565.7416 |
| 3max | A FLOP | 1172 | 1172 | 304 | 906.2 | 320 | 630.3447 | 736.1396 |
| 3max | A TURN | 1455 | 1455 | 295 | 1097.3 | 320 | 283.3768 | 457.5319 |
| 3max | A RIVER | 1717 | 1717 | 389 | 1267.3 | 320 | 511.6989 | 711.3193 |
| 3max | B PREFLOP | 3408 | 2000 | 329 | 461.9 | 320 | 0.2290 | 0.8315 |
| 3max | B FLOP | 4435 | 2000 | 362 | 315.8 | 320 | 0.1697 | 0.5851 |
| 3max | B TURN | 5762 | 2000 | 384 | 182.1 | 320 | 0.1029 | 0.6026 |
| 3max | B RIVER | 65122 | 2000 | 261 | 83.3 | 320 | 0.1000 | 0.7528 |

Each listed model is IMPLEMENTED and TRAINED ON BOUNDED DATA. None is strategically VALIDATED or accepted for production. Seed 202 also trains every route; its complete per-model losses, coverage and optimizer counts are retained in its report.

Generated/retained coverage includes street, position, hand class, canonical exact combo, effective stack, SPR, opening/re-raise/jam context and board texture. Generated samples additionally retain physical-combination counters. Canonical replay identities are explicitly labeled as canonical; they are not physical-suit counts. Context-specific train/validation/test losses and regret-matched strategy errors are in `canonical-context-{hu,3max}.json`.

Opening coverage is still insufficient in these tiny experiments: seed-101 HU retained only one A opening AA and zero B opening AA. The controlled context sampler and protected street allocation make coverage measurable and adjustable; five cycles do not establish conditional 169-class coverage. A large total of postflop records is not evidence of preflop sufficiency.

## Stability across independent seeds

| Track / seed | Cycles | Traversals | Nodes | Cycle wall total (s) | Fit time (s) | Checkpoint (MB) | Final probe drift L1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| hu/101 | 5 | 640 | 37922 | 14.01 | 5.73 | 40.73 | 0.1393 |
| hu/202 | 5 | 640 | 50534 | 16.72 | 5.51 | 41.34 | 0.0803 |
| 3max/101 | 5 | 960 | 351143 | 50.49 | 6.39 | 47.43 | 0.0897 |
| 3max/202 | 5 | 960 | 255621 | 37.80 | 6.34 | 45.51 | 0.0888 |

The final retained all-nonpositive A fractions (PF/flop/turn/river) are:
- hu/101: 19.71% / 8.04% / 5.45% / 0.00%.
- hu/202: 1.45% / 0.78% / 0.00% / 0.00%.
- 3max/101: 0.00% / 0.00% / 5.29% / 0.00%.
- 3max/202: 12.09% / 0.39% / 0.00% / 0.44%.

## Frozen-replay optimization

Identical seed-101 replay and grouped partitions; compare unchanged weights, 64 or 256 additional updates with persistent Adam, and 64 with reset Adam. Each candidate starts from the same checkpoint. New-data eligibility is explicitly bypassed only in this diagnostic, never in the training loop.

| Track / family | Budget | Validation losses PF / flop / turn / river |
|---|---|---|
| hu/advantage | unchanged | 237.0549 / 344.2762 / 373.4261 / 317.5322 |
| hu/advantage | 64_updates_persistent_adam | 236.7148 / 334.7753 / 358.9385 / 295.5628 |
| hu/advantage | 256_updates_persistent_adam | 240.3675 / 306.2509 / 355.9372 / 239.2143 |
| hu/advantage | 64_updates_reset_adam | 236.4551 / 336.0613 / 361.8872 / 301.5924 |
| hu/strategy | unchanged | 0.3675 / 0.1969 / 0.1191 / 0.1362 |
| hu/strategy | 64_updates_persistent_adam | 0.3861 / 0.1937 / 0.1159 / 0.1413 |
| hu/strategy | 256_updates_persistent_adam | 0.3800 / 0.1913 / 0.1134 / 0.1449 |
| hu/strategy | 64_updates_reset_adam | 0.3866 / 0.1946 / 0.1161 / 0.1406 |
| 3max/advantage | unchanged | 337.7011 / 520.1353 / 358.1621 / 601.5435 |
| 3max/advantage | 64_updates_persistent_adam | 338.4386 / 506.6608 / 355.4648 / 587.9739 |
| 3max/advantage | 256_updates_persistent_adam | 347.3980 / 472.2671 / 350.4446 / 536.9692 |
| 3max/advantage | 64_updates_reset_adam | 338.3386 / 508.0362 / 355.6408 / 589.3449 |
| 3max/strategy | unchanged | 0.1836 / 0.1301 / 0.0826 / 0.2441 |
| 3max/strategy | 64_updates_persistent_adam | 0.1858 / 0.1360 / 0.0759 / 0.2602 |
| 3max/strategy | 256_updates_persistent_adam | 0.1986 / 0.1539 / 0.0774 / 0.3131 |
| 3max/strategy | 64_updates_reset_adam | 0.1854 / 0.1395 / 0.0765 / 0.2689 |

Increasing the fit budget improves several postflop A losses but worsens preflop A and several B losses. Persistent versus reset Adam is mixed; persistence is retained for reproducible incremental optimization, not claimed as universally more accurate. The conservative default stays at 64 updates per eligible model. No fixed 20/30/50-epoch schedule is used.

## Independent decision diagnostics

Preflop: all 1,326 exact holdings, card-reversal invariance, and six detailed holdings at 25 BB. Conditional action EV uses 128 paired worlds and fixed shared-B continuations for every candidate root policy. These continuations are a controlled comparator, not a certified strategic reference. Full values and uncertainty are in `canonical-strategic-probes.json`.

Important failure: HU A folds AA about 72.7% and 72o about 72.0% in seed 101. B folds AA about 5.5% and 72o about 4.6%. Thus B retains history while the current A can regress severely; neither is accepted. 3-max A/B almost never fold 72o and lose more conditional EV than the shared baseline. No hand-strength rules or regret fallback changes were used to conceal these outcomes.

River: exact compatible private-deal enumeration and exact terminal utilities, with a 200-iteration CFR reference. The response gains below are exact bounded responses in this tiny conditional game, not full-game exploitability or multiplayer Nash guarantees.

| Track | Shared A response gain | Specialist A | Shared B | Specialist B |
|---|---:|---:|---:|---:|
| hu | 0.3000 | 0.0000 | 0.0492 | 0.0888 |
| 3max | 0.1074 | 0.0000 | 0.4389 | 0.0679 |

Flop and turn: independent 128-world and 512-world calculations through `hybrid.search`, explicit two-combo opponent factors and identical fixed shared-B continuations. The larger sample is a noisy reference, not exact truth. Per-action standard errors and budget warnings remain in `canonical-postflop-probes.json`.

| Track / street | 128-vs-512 action EV RMSE | Shared A EV | Specialist A EV | Shared B EV | Specialist B EV |
|---|---:|---:|---:|---:|---:|
| hu/FLOP | 0.649 | 12.445 | 11.773 | 11.365 | 11.853 |
| hu/TURN | 1.670 | 3.809 | -0.800 | 2.477 | -1.491 |
| 3max/FLOP | 1.552 | 14.494 | 16.354 | 15.145 | 16.340 |
| 3max/TURN | 2.458 | 5.387 | 6.349 | 4.490 | 4.332 |

Results are mixed, including worse HU turn decisions. Loss reduction, falling drift and a zero response gain in one tiny river fixture do not establish strong play.

## 3-max collector variance

`collector-profile.json`: 96 tasks, fixed frozen profile and task seeds. Original corrected partial enumeration is retained for comparison. The optional extension enumerates both opponents from a declared public street onward; own actions remain strategy-sampled and earlier prefix corrections remain intact. No clipping or sample rejection.

| Mode | Seconds | Nodes | River records | River weight ESS | Largest river weight |
|---|---:|---:|---:|---:|---:|
| Original | 1.770 | 11,728 | 498 | 67.58 | 3.67% |
| Enumerate from river | 3.179 | 23,663 | 3,217 | 138.80 | 2.65% |
| Enumerate from turn | 8.581 | 69,594 | 9,569 | 547.83 | 0.19% |

Exact RNG-branch validation gives maximum per-infoset action-mass errors 1.33e-15 (original) and 8.88e-16 (river enumeration). The new 3-max configuration explicitly chooses river enumeration. It doubles raw river ESS in this measurement but does not cure prefix-weight concentration: retained river B ESS is only 83.3 and 85.8 out of 2,000 in the final two seeds. Correlated samples must not be mistaken for independent trajectories. Turn enumeration remains an explicit, more expensive experiment option.

## CPU, RAM and storage

`lifecycle-isolated/`: one further real iteration from a bounded checkpoint, separate process per configuration. RSS is sampled every 20 ms and sums child RSS (shared pages may be double-counted). Replay is 8,000/family, not the 50,000/family substantial-run capacity.

| Track / workers | Load seconds / peak MB | Iteration seconds / peak MB | Save seconds / peak MB | Export seconds / peak MB |
|---|---|---|---|---|
| HU / 1 | 2.199 / 400 | 3.602 / 393 | 0.408 / 553 | 0.177 / 515 |
| HU / 2 | 2.171 / 401 | 3.430 / 791 | 0.406 / 553 | 0.171 / 515 |
| 3-max / 1 | 2.537 / 441 | 12.875 / 431 | 0.497 / 639 | 0.201 / 568 |
| 3-max / 2 | 2.628 / 441 | 9.200 / 876 | 0.524 / 593 | 0.242 / 561 |

Final model hashes match exactly between one and two workers for each track. New-run defaults therefore use HU 1 worker and 3-max 2 workers, one PyTorch thread. These local results do not promise the same speed on OVH or prove full-capacity RSS fits a particular VPS.

cProfile artifacts retain generation, observation encoding, inference, child-state cloning, replay/fit and optimizer handling. Profiled HU: 4.144 s total, 1.921 s collection, 1.255 s fitting. Profiled 3-max: 15.972 s total, 12.982 s collection, including 10.844 s partial collection and 4.325 s frozen inference (nested timings overlap). GRU inference, observation encoding and game transitions dominate 3-max; no engine rewrite or GPU was introduced.

A pre-encoded singleton routing microbenchmark measured 109.77 ms/1,000 calls for mixed-batch dispatch versus 87.01 ms for direct street dispatch. Frozen inference now uses the direct route. Mixed batch dispatch remains tested.

`onnx-performance.json`: four Average graphs total 296,328 bytes/track, plus about 19.6 KB catalog. Median warm Python ONNX queries are 0.063–0.083 ms; first queries including lazy session construction are 0.78–2.01 ms. Browser timing is environment-dependent and is not inferred from these Python numbers.

## End-to-end validation and failures corrected

- All eight ONNX routes match PyTorch within 1e-6, with variable public-history length.
- Real Chrome/WebWorker: 65 observed-action posterior transitions, all eight routes, maximum absolute Python/browser error 3.182e-7. Overview renders 169 classes for both tracks; Specific Spot queries all four streets in HU and 3-max; Test Live uses the same actual worker.
- Local publication transport tests explicit shared-to-street counter reset, stale expected-pointer rejection and refusal of an old shared writer after migration. It uploads four models and catalog before updating the pointer; injected river-upload failure preserves the previous pointer. No Supabase write was made. Old pinned references remain readable.
- Parallel worker serialization initially omitted specialist layout/collector fields. Complete snapshot serialization and a real two-worker equality regression fixed it.
- Worker likelihood cache initially used unsorted candidate cards. Sorted exact-combo keys fixed reversed-order posterior failures.
- Local UI build initially read production storage settings. Browser acceptance uses an explicit local policy source, not production weights. Both production-config and local-config static builds pass.
- Checkpoint tests cover disjoint parameters, cold exports, no-new-data skips, separate A/B cadence, corrupt optimizer rejection, bounded storage, full optimizer/RNG resume and old-schema compatibility.

Full test counts and final source reconciliation are recorded in STREET_NETWORK_MIGRATION.md.

## Next substantial experiment — not executed

Use separate new directories and the versioned street configurations; do not resume a shared checkpoint into these models. First run 20 complete cycles/track: HU 512 traversals/cycle (10,240 total), 3-max 384 (7,680 total), A <=64 updates/street/cycle, B <=64 every third cycle. Keep a 10-minute wall guard per track initially and stop after complete cycles. The measured small-replay throughput suggests minutes, but full-capacity encoding, evaluation and changing trees can increase cost; measure before expanding.

Require generated and retained coverage by context, independent train/test roots, opening hand-class counts, A regret accuracy and all-nonpositive rate, B KL/ESS, independent action EV and bounded response tests. Compare seeds. Reject a run that merely fits historical errors better. Increase informative independent roots before increasing update caps. The tiny cohort lacks AA/KK/AKs conditional supervision and does not justify automatically activating any model.

Supabase continues to retain bounded releases and compact hourly results; detailed per-street replay, optimizer and coverage data stay local. A later deployment must explicitly start the new schema in a new run directory and preserve the currently active shared checkpoint.

Evidence index: `runs/street-network-migration/evidence-index.json` contains source and report hashes. Detailed local evidence occupies approximately 1.9 GB, including multiple immutable experimental checkpoints; it is ignored by Git and is not sent to Supabase. Final regression logs are retained alongside the reports.
