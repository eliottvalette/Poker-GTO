# Context replay and validation correction

Status: IMPLEMENTED AND DEPLOYED. Strategic improvement remains unvalidated; the first production candidates were rejected by the activation gate.

| Requirement | Status | Evidence / next action |
|---|---|---|
| Explicit preflop context and opening-hand-class allocation | DONE | ContextReplay; fixed total capacity/bytes, original collector retained |
| Inclusion-weight correctness and deterministic resume | DONE | Exhaustive reservoir selections and checkpoint regression |
| Actual train/validation/test coverage | DONE | Per-split context/class counts and ESS |
| Frozen A/B supervision validation | DONE | Frozen historical targets, zero-regret/uniform baselines; separate from strategic references |
| Bounded matched comparison | DONE | Same stream/checkpoint, allocation-only comparison and independent poker probes |
| Separate candidate publication from active policy | DONE | Candidate evaluation before activation; explicit decisions retained |
| OVH end-to-end activation | DONE | Fresh runs bootstrapped, saved/resumed/exported, both services running without restarts |
| UI and Supabase compatibility | DONE | Real candidate uploads and evaluations; both tracks verified in Chrome against live Storage |

No claim of poker quality follows from lower replay loss. Protected strata estimate the original generated stream's loss total using inverse inclusion probability; the normalized fit is a finite-sample ratio estimator. No weight clipping is introduced. Root distribution changes, if selected, must be documented separately.

## Implemented contracts

- Each A/B preflop memory retains 20,000 slots: first open 8,000; facing open 4,000; facing 3-bet 2,000; facing 4-bet or more 1,500; facing jam 2,000; other 2,500. The first-opening allocation is divided into 169 independent class reservoirs, 47 or 48 slots each. Exact combinations retain their original collector weights; class multiplicities are not silently changed in the fitted objective.
- Total capacity remains 50,000 per family; byte allocations still total 512 MiB per family. The original root distribution, card sampler and CFR collectors are unchanged, isolating the allocation correction. No GPU, larger network or epoch increase.
- Version-6 checkpoints declare street-replay-v2 and nested preflop-context-v1 reservoirs; older contracts remain readable. Fresh output paths are `runs/hu-contexts` and `runs/3max-contexts`. All sixteen specialists keep persistent weights/Adam, at most 64 updates per eligible fit, A every cycle and B every third cycle.
- Split diagnostics record retained contexts, opening classes, root counts, sample counts and ESS separately. Once each held-out split has 64 records, each specialist freezes up to 128 validation and 128 test records with exact targets, weights and provenance in its checkpoint. Banks never enter training or neural exports. Historical losses are accompanied by zero-regret/uniform baselines; A reports all-nonpositive predictions. These are historical-supervision tests, not current equilibrium regret ground truth.
- Every live export additionally computes independent bounded river strategy/reference diagnostics for A (regret-matched) and B. They remain in the local sealed bundle. Their narrow game assumptions are explicit.
- OVH publishes to `candidate.json`; `current.json` continues selecting the live policy. The hourly evaluator records candidate performance against the unchanged pinned initial reference, then compares against the active policy on a separate seed stream (offset 3,000,007). Only a complete comparison with a positive lower descriptive 95% bound activates the candidate. Rejected/uncertain candidates remain visible in history. This controlled benchmark gate is not a formal sequential test or general poker certification.
- Pruning protects both active and candidate histories; model storage remains bounded. UI labels candidate results separately from the active iteration without adding another dashboard.

## Bounded evidence

`runs/context-replay-comparison/{hu,3max}.json`: same existing specialist checkpoint, same frozen source stream, 400-slot preflop budget and 64 persistent-Adam updates per model, seed 451. Reduced allocations preserve the same six contexts.

| Track/model | Openings uniform → contexts | Fixed test loss uniform → contexts |
|---|---|---|
| HU A | 27 → 34 | 77.91 → 77.58 |
| HU B | 27 → 39 | 0.3265 → 0.3173 |
| 3-max A | 15 → 41 | 717.63 → 671.14 |
| 3-max B | 10 → 38 | 0.2158 → 0.2438 |

Matched 16-group poker comparisons on a separate seed stream: HU -2.88 BB/100 [-15.69,9.94]; 3-max +4.08 [-7.61,15.77]. No demonstrated playing-strength advantage. The allocation repair is accepted for demonstrably improved retention and verified probability semantics; the resulting policies are NOT accepted as stronger on this evidence. Production candidates must pass their own gate. No unsuccessful neural result is hidden.

`runs/context-replay-pilot/`: three complete production-config cycles for each track, followed by checkpoint resume and all-street ONNX export. HU 37.83 s, 3-max 53.07 s including save/resume/export on the local machine, using one generation worker. All sixteen specialists fitted; all sixteen frozen banks became available by cycle 3. Separate small river diagnostics completed for A/B on both tracks; they do not establish preflop quality.

Validation: 244 Python tests, 158 browser tests and production static build passed. Exact enumeration verifies corrected per-action totals and absence of double correction. Checkpoints preserve reservoirs/RNG/weights; frozen targets remain unchanged across new replay data. Candidate tests preserve the active pointer on uncertain results and exercise a qualified activation. A pre-existing workflow test was updated to supply the now-required run provenance when comparing direct and checkpoint-extracted exports.

## Production verification — 2026-10-11 17:07 UTC

The running release is `25dacf5e3c4759f27daf3b5201ba16fc60c2a493`, a verified source archive from the tested working tree (base commit `2670d068b40cfd824308667d9187dbaa1884af42`, explicitly dirty). No Git commit/push or Vercel deployment was performed. Commit the complete tested source changes together before the next normal CI deployment.

- HU run: `792568bdb59040adbb1d7275e036793a`; 3-max run: `682e4a7359b84ceeb281e50835973d04`.
- Canonical service paths remain `/var/lib/gto/hu` and `/var/lib/gto/3max`. Previous runs are archived under their explicit `*-before-contexts` paths. They are not resumed by the live services.
- Each new run completed three bootstrap cycles, version-6 save/resume checks and validated four-street ONNX export before switching services. Remote bootstrap totals: HU 125.68 s, 3-max 146.88 s, including export. Generation uses one HU worker and two 3-max workers.
- Both services subsequently reached iteration 9, remained active with zero restarts, and continued normal hourly checkpoint/export cycles. Publication/evaluation timers remained active. Host memory at inspection: 2,490 MiB used, 5,261 MiB available, no swap; this is a point-in-time reading, not a maximum-memory benchmark.
- Both iteration-3 candidate bundles were uploaded and read back from Supabase. Their evaluations completed with 128 independent groups. The independent activation comparison was HU -19.82 BB/100 [-49.99, 10.36], 3-max -37.04 [-67.37, -6.71]. Neither candidate was accepted. Active policies remain HU #730 and 3-max #766; they are not certified strong policies.
- Real Chrome checks against the production Storage data showed `Candidate · Iteration 3 · Active #730` for HU and `Candidate · Iteration 3 · Active #766` for 3-max, each in its new run series, with no alert. This checks the local production UI build; public Vercel delivery still follows the user's push workflow.

Next action: inspect the next hourly candidates' fixed A/B validation, conditional coverage and independent activation comparisons. Do not interpret additional iterations, improved retention, or a single successful gate as proof of general poker strength. Position/stack coverage remains available in aggregate diagnostics; the new split-specific summary currently resolves context/opening class/root counts and ESS, not every position-by-stack cross-tabulation.

