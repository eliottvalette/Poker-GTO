# Phase 2 roadmap

> Historical evidence: the old models and `artifacts/` were explicitly deleted during training finalization. Numerical results below describe previous bounded experiments, not currently deployed models. See [TRAINING.md](TRAINING.md) for the current configuration, empty catalog and launch/export workflow.

Implementation status: **COMPLETE for the bounded, explicitly scoped workflows below.** General poker strength: **NOT YET VALIDATED**. Rejected learned models remain disabled. Completion means operational software and executed acceptance, not convergence of NLHE or calibrated human behavior.

## Preservation contract

Work started from the local uncommitted tree. Canonical rules, HU and three-player collectors, published policies, entrypoint scripts and default training configuration/checkpoints remain. No deployment or long/GPU/cloud training was performed. Phase 1 reports and `artifacts/hybrid-validation` had been explicitly deleted before this phase. They cannot be preserved as available historical evidence; they were not recreated or presented as new measurements. All new evidence is under `artifacts/phase2`.

## Requirement reconciliation

Every numbered implementation phase is retained. Paths below are repository-relative. Detailed numerical results, budgets and limitations are in the experiments and acceptance documents.

| Phase | Status | Completion criterion and actual implementation | Source owners | Validation artifacts / tests | Limitations / next executable action |
|---|---|---|---|---|---|
| 0 Execution and tracking | DONE | Four persistent documents, bounded experiments, retained checkpoints, no replacement migration | `docs/HYBRID_BOT_PHASE2_*.md` | All phase2 reports carry budgets; final test/build logs | Future long compute needs authorization |
| 1 Current-tree audit | DONE | Inspected existing rules, solver, learning, features, browser and tests; preserved correct systems | `hybrid/`, `training/`, `ml/`, `ui/src/` | Original baseline and full regression | Deleted Phase 1 reports unavailable; do not recreate as historical evidence |
| 2 Strategic baseline | DONE | Versioned 24-root suite across counts/streets/stacks, independent seeds, broader polarized ranges, heterogeneous pool, paired EV and full-hand evaluation | `quality.py`, `matches.py`, `reference_experiments.py` | `evaluation/baseline.json`, frozen `test-roots.json`, matched reports | Synthetic populations and finite MC references; expand independent root count before strength claims |
| 3 Persistent ranges | DONE | Atomic action/board updates, explicit human likelihoods, public/private separation, compatible mixtures, next-hand transition | `session.py`, `beliefs.py`, `ranges.py` | `test_hybrid_session.py`, complete-hand fixtures/demo | Product factors within a profile cannot express arbitrary correlations; large evidence uses bounded sampling |
| 4 Computed references | DONE | Independent-BR-gated finite tables, exact local CFR, explicit conservative computed fallback, HU/3-max live routing | `reference.py`, `river_solver.py`, `decision_engine.py` | `reference/`, `reference-ablation/` (472 rows) | Sparse table coverage; fallback is a synthetic equity-based policy, not a balanced equilibrium |
| 5 Reaction models | DONE | Hand/public-state/profile-conditioned policies; separate learned profile/count models; grouped trajectories, calibration, OOD and EV tests | `reference.py`, `behavior.py`, `phase2_learning.py`, `learning.py` | `behavior/report.json`, `evaluation/posterior-calibration.json`, `learned-behavior.json` | Narrow prediction gates do not establish downstream benefit; keep explicit profiles default |
| 6 Behavior uncertainty | DONE | Latent profile posterior retains hand/profile correlation; private conditioning; reference/exploitative modes and EV sensitivity | `session.py` | Analytic posterior/session tests; behavior-sensitivity reports | Browser uses explicitly selected profiles; latent mixtures are Python-only; model uncertainty is not a confidence interval |
| 7 Search quality | DONE | Paired worlds, exact settlements, opt-in independent-pilot allocation, ranking uncertainty, independent forest evaluation | `search.py`, `leaf_evaluation.py`, `decision_engine.py`, `robustness.py` | `adaptive.json`, `forest-generalization.json`, information-safety tests | Adaptive MSE worsened; leave opt-in. Local solving is belief-conditioned, not safe resolving |
| 8 Offline coverage | DONE | Separate generated/retained weighted coverage, preserved recorded-player stratification, optional protected replay with inverse inclusion weights and full runner resume | `training/metrics.py`, `ml/stratified_memory.py`, `ml/deep_cfr.py`, `training/{config,checkpoint,runner}.py`, `hybrid/replay.py` | `retention/report.json`, `fixed-replay/`, `protected-training/report.json`, runner resume regression | Protects opening context, not every hand class; normalized weighted fitting is a finite-sample ratio estimator |
| 9 Continuation learning | DONE | Diverse independent-root labels, class169 vs suit-aware combo1326, capacity sweep, table/mean/rollout/exact comparisons, invalidated/revalidated gates | `continuation.py`, `value_features.py`, `phase2_learning.py` | `values/`, `values-aggressive/`, `evaluation/value-exact-comparison.json`, gate tests | All models rejected; no learned-leaf speedup claimed; expand independent labels only after diagnosing errors |
| 10 Browser integration | DONE | Actual existing pages/tournament table, persistent action/reveal replay, exact raises, profiles/modes/ranges/EVs, worker cancellation and stale-result protection | `ui/src/lib/poker/{behavior,beliefs,hybrid,hybrid.worker}.ts`, existing analysis/table components | 134 browser tests, actual Chrome `browser-static-smoke.json` | Browser latent mixtures unavailable; explicit broad-range evidence can fail instead of inventing a posterior |
| 11 Profiling | DONE | Component timings, worker timings, learning timings, cached compatible proposals with identical random stream | `ranges.py`, `experiments.py`, experiment reports | `sampling-profile.json`, `components.json`, 512-draw numerical identity test | Narrow microbenchmarks are not broad-range latency guarantees |
| 12 Quality loop | DONE | Baseline → conditioned policies → adaptive/learned rejection → larger matched budgets and wider ranges; retained ablations | `quality.py`, `matches.py`, `robustness.py` | Paired improvements and realized-chip intervals, two 480-hand runs | No statistically reliable realized-win improvement; frozen suite was retained rather than optimized in isolation |
| 13 A–H acceptance | DONE | Complete workflows, learning lifecycle, browser, model assumptions and explicit failures exercised | Tests, demo and smoke script | Acceptance matrix in `HYBRID_BOT_PHASE2_ACCEPTANCE.md` | Exact cases deliberately bounded; general strategic quality unvalidated |
| 14 Completeness reconciliation | DONE | All requirements mapped to code/evidence; implementation separated from quality | This document and acceptance report | 195 Python tests; 134 browser tests; static export | No GTO or safe-resolving claim |
| 15 Deliverables | DONE | Source/tests/fixtures/reports/provenance/optimizer checkpoints/exports, Python and browser demonstrations, further compute proposal | `examples/phase2_demo.py`, `tests/phase2_browser_smoke.mjs`, phase2 artifacts/docs | Reproduction recipes in acceptance report | Experimental artifacts are local and were not published |
| 16 Final validation | DONE | Decisions remain computed from ranges/search and exact payouts; optional models never silently promoted | Existing canonical engines plus hybrid owners above | Full regressions, actual static worker, build, published-policy hashes | Next work should be evidence-driven strategic evaluation, not another architecture migration |

## Acceptance ledger

A HU complete hand: DONE. B Three-player complete hand/transition: DONE. C Behavior sensitivity and analytic posterior: DONE. D Bounded self-play learning/resume/export/load: DONE. E Quality versus computation: DONE. F Real static-browser complete hands: DONE. G Matched strategic evaluation: DONE, **playing-strength improvement not established**. H Explicit failure modes: DONE.

## Next executable action

For inspection, run `examples.phase2_demo.demonstrate(2)` or `(3)` and open the existing browser analysis. No implementation milestone remains pending in this bounded delivery. The next research action is a larger independent matched-deal evaluation under the proposal in the acceptance document; do not enable rejected value models or assume a larger training run fixes strategic assumptions.

## Final corrections recorded

The initial roadmap checkpoints were provisional. Frozen replay, retention, published-policy ablations, independent-world tests, actual static-browser cancellation and tournament-table analysis were subsequently executed. Protected replay was then connected as an explicit optional `TrainingRunner` allocation and validated with exact two-track resume. Historical statements that these checks remained pending are superseded by their retained reports, not silently treated as evidence.

## Test Live range inspection extension

Status: DONE. Opponent seats open a 169-class posterior matrix on hover, keyboard focus or tap. `RangeInspection.tsx` displays class probability mass and exact-combo details; `SeatInspection.tsx` portals the panel outside the table clipping area. The existing belief worker emits separate exact observer-conditioned marginals via `observerMarginals`; these display results never replace the public factors used by search. Snapshot keys prevent previous-hand/action posteriors from appearing current. Source observation uses only Hero cards, public board and the same posterior factors used by analysis. A three-player marginal uses inclusion/exclusion over the other unknown hand, including folded players. Independent exhaustive-product and hidden-card-invariance tests pass. The browser suite now contains 136 passing tests. A static Chrome run used real pointer movement over both opponent seats, verified 169 cells and exact-combo selection after bot actions, and retained screenshots plus `artifacts/phase2/evaluation/range-hover-smoke.json`.

### Range visualization refinement

Implemented `range-display.ts` and two inspector views: blocker/multiplicity-normalized Range, and profile-based Reactions restricted to the acting opponent. Added fixed orange/neutral/cyan legend, impossible-class hatching and mutually exclusive postflop composition. 137 browser tests pass, including uniform-combo normalization, conditional reaction mass and invariance to simulator hidden cards. No model training or policy publication was performed. Active published models remain 3-max iteration245 and HU iteration412; Test Live opponents remain uniform legal bots.

## Training finalization and cleanup

DONE: main HU/three-player configs now use recorded-player stratification, 20% protected openings, independent50/20 fit budgets and explicit bounded session settings. Training only resumes its own local checkpoint by default; resuming a published checkpoint requires an explicit programmatic option. Test Live opponent selection now drives actual actions and Bayesian likelihoods, including ONNX inference in the worker. Unknown or unavailable policies fail explicitly. Empty catalogs clear old loaded models. See `TRAINING.md` for the operational commands and limitations.

Explicit cleanup removed `artifacts/`, old `policy/` weights/releases/legacy CSV/GZ, `ui/public/policy/releases`, old run outputs if present, and generated UI build caches/static export. The catalog contains no active or exported models. Historical numerical reports above are not current model-quality claims. No large production training was launched and no diagnostic checkpoint was promoted. Final validation:196 Python tests,140 browser tests, production build, actual ONNX opponent/likelihood test before deletion, and empty-catalog browser test afterward.


## Training cadence correction

The earlier training-finalization settings are superseded; they passed software checks but failed compute-efficiency and opening-coverage audits. Old run checkpoints are retained.

| Requirement | Status | Owner / evidence | Limitation |
|---|---|---|---|
| Stop old production compute | DONE | No active `train-3.py`; durable iteration-23 checkpoint retained | No process was killed after it had already stopped |
| Independent A/B fitting cadence and bounded updates | DONE | `ml/deep_cfr.py`, `training/{config,runner}.py`; skip/resume tests | A still defines strategy updates; frozen traversals are not new CFR iterations |
| Reuse encoded replay without changing numerical results | DONE | `ml/train.py`; cached/uncached and variable-history parity tests | Explicit tensor-memory guard; no GPU |
| Larger generation without buffering the whole sample stream | DONE | Ordered bounded chunks; checkpoint/replay parity tests | Node and per-chunk byte guards remain explicit |
| Compare frozen batch sizes and fresh/warm budgets | DONE | `training/cadence_experiment.py`, `runs/cadence-comparison/{hu,3max}.json` | Frozen replay comparisons do not certify generalization |
| Exact local reference and regret diagnostic | DONE | `runs/cadence-comparison/hu-reference.json` | Small river only; instantaneous versus historical regret targets differ |
| Full-config pilots and resume | DONE | Two batches per track under `runs/cadence-comparison/*-preflight` | Efficiency validated, poker strength not established |
| Coverage-distribution redesign | TODO | Existing sampling probabilities and collectors unchanged | Requires unbiased reference validation before changing root allocation |
| Independent matched-compute playing-strength acceptance | TODO | No models promoted | Requires held-out scenarios; short pilots do not establish strong play |

Next executable action: inspect the cadence report and retained pilot metrics before authorizing a longer run. The delivered cadence correction is functional; the two research acceptance items above are explicitly not claimed complete. See `TRAINING_CADENCE.md`.

## CPU performance investigation

DONE for the measured workloads: eight-worker-bounded worker/thread/minibatch sweeps, cProfile and native PyTorch profiles, source optimizations, full-iteration model/replay parity and execution-only resume support. Evidence and next actions are maintained in `TRAINING_PERFORMANCE.md`; raw runs are under `runs/performance`. Existing production checkpoints are immutable benchmark inputs. No learning budgets or poker distributions are changed as performance shortcuts.

Performance acceptance:64 relevant regression/feature/ONNX tests passed; original production checkpoint hashes unchanged; exact model/replay parity verified across optimization/configuration variants. Selected execution settings are HU8 workers / three-player4 workers / trainer1 thread / generation chunk128. Existing checkpoints resume without changing learning settings; execution changes are explicitly logged. See `TRAINING_PERFORMANCE.md` for raw ranges and scope limitations.
