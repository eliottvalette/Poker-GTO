# Street network migration

Implementation: COMPLETE. Bounded training: completed for all sixteen specialists and two seeds. Strategic validation: NOT ESTABLISHED; experimental weights are not accepted production policies.

No OVH process, active checkpoint, published policy, Supabase object or deployment was changed. Existing unrelated local UI/audit changes are preserved. All experiment outputs live under `runs/street-network-migration/` and are unpublished.

## Requirements and acceptance ledger

DONE below means implemented and exercised, including unsuccessful quality measurements. It does not mean the associated model has strong poker performance.

| Requirement | Status | Owning modules | Evidence / remaining limit |
|---|---|---|---|
| Inspect current pipeline and preserve contracts | DONE | ml/, training/, hybrid/, ui/ | Legacy shared schema and existing collectors remain supported; full regression |
| Sixteen independent models | DONE | ml/model.py, ml/deep_cfr.py | Four A and four B per track, disjoint parameter identity tests; all sixteen fitted |
| Deterministic public-street routing | DONE | ml/deep_cfr.py, scripts/parallel_cfr.py | Atomic four-A frozen snapshot; mixed-street tests and real 1/2-worker equality |
| Street replay and explicit RAM allocation | DONE | ml/street_replay.py, ml/memory.py, training/config.py | Four Algorithm-R partitions/family; inclusion correction for aggregate diagnostics; strict capacity/budget validation |
| Coverage and actual supervision | DONE | training/root_sampler.py, training/metrics.py | Sixteen legal context recipes, recorded-actor stratification, generated/retained metrics and grouped test errors. Small runs still lack conditional opening examples |
| 3-max variance investigation | DONE | cfr_solver.py, scripts/strategy_collector_audit.py | Original retained; optional river enumeration passes exact action masses. Retained ESS remains poor; no clipping |
| Incremental independent fitting | DONE | ml/street_training.py, ml/train.py | Persistent Adam/weights, separate thresholds/cadences/caps; no-new-data skip, corrupt optimizer rejection, deterministic resume |
| Grouped fixed-replay studies | DONE | training/street_experiments.py | 64/256-update and reset-Adam comparisons; canonical public-root train/validation/test groups; contextual losses |
| Closed-loop HU and 3-max | DONE | training/runner.py, hybrid references | Five cycles × two seeds × two tracks; all sixteen receive data. Mixed independent EV results, no general quality acceptance |
| Versioned checkpoint migration | DONE | training/checkpoint.py, training/runner.py | New v5 contract, full model/optimizer/replay/root/RNG states; old v4 readable; no implicit weight conversion |
| Eight ONNX routes | DONE | ml/export_onnx.py, ml/onnx_policy.py | All routes <1e-6 PyTorch discrepancy; cold export rejection; lazy session loading |
| Supabase and compact history compatibility | DONE | training/publication.py, training/evaluation_history.py | Complete catalog/graphs before pointer; injected upload failure test; legacy baseline handling and bounded retention. No remote activation |
| Existing browser pages and range calculations | DONE | ui/src/lib/onnx-policy.ts, poker/hybrid.worker.ts | Real Chrome: Overview both tracks, Specific Spot all streets, Test Live worker; 65 posterior transitions, 3.182e-7 parity error |
| Performance profiling | DONE | training/vps_benchmark.py, performance_experiment.py, street_experiments.py | cProfile, isolated 1/2-worker load/iterate/save/export RSS, routing and ONNX benchmarks. Full-capacity VPS RSS not measured |
| Regression and reproduction | DONE | tests/test_street_networks.py, tests/street_browser_smoke.mjs, examples/street_network_demo.py | Full Python/browser suites and static builds; importable bounded example and browser fixture server |
| Final model-quality reconciliation | DONE | STREET_NETWORK_EXPERIMENTS.md | Every model: implemented, bounded-trained, not strategically validated. Explicit failed preflop/turn results and next measured budget |

Next executable engineering action: review the retained quality failures and launch an explicitly chosen new-run budget if desired. No production launch or model promotion is pending implicitly.

## Architecture and authority

Each player-count track owns `StreetNetworks("advantage")` and `StreetNetworks("strategy")`. Each bundle contains `PREFLOP`, `FLOP`, `TURN`, `RIVER` modules with independent instances of the existing card/numeric/history encoder and head. There are no shared trainable tensors across streets or tracks and no physical-seat-specific models. Existing exact cards, deterministic descriptors, hero-centric normalization, card-order invariance, full history, feature schema and canonical actions are retained.

CFR reads one immutable snapshot of all four A models. Public street selects the acting player's specialist. All tasks finish before staged model/replay changes become the next frozen profile. Worker transport carries the complete snapshot contract, including player count, layout and optional collector allocation. B models approximate the weighted historical strategies; they are not used as a substitute for exact payouts or hybrid search.

Cold A networks have explicit zero regret outputs and use the existing uniform legal regret-matching fallback. Cold B networks are uniform internally and are visibly marked untrained. Exporting a trained policy bundle requires all four B specialists to have fitted; no missing street is replaced with another model. Unavailable published models continue to leave the algorithmic hybrid engine usable.

## Replay, weighting and roots

`StreetReplay` partitions records by actual decision street. Each partition has its own Algorithm-R reservoir, RNG, capacity, byte accounting and model objective. Within one street, constant inclusion probability cancels in the normalized weighted loss. Aggregate diagnostics use inverse-inclusion correction; data from a populous river cannot consume preflop optimizer updates.

Sample iteration, producing profile version, reach/collector weight, traversal mode, trajectory identity and canonical public-root group persist. Legacy records remain readable for audit without inventing missing trajectory provenance. New street records require it. Grouped fitting excludes validation/test roots from training; a pooled cross-street fit is explicitly rejected.

The optional controlled sampler constructs legal, card-independent prefixes through the canonical engine: first open, facing open/3-bet/4-bet, short stacks, facing jam, and check/bet/raise situations on every postflop street plus river jams. It varies stacks, board runouts and seats. It stratifies the actual recorded HU root actor, or the traverser in 3-max, with separate context streams. Fresh uniform card permutations start per frozen profile; incomplete old permutations are not carried into a changed strategy.

This is an explicit stationary distribution of subgame roots. It is **not** an unbiased estimate of natural full-game root visitation. Prefixes contain no card-strength action rules. CFR sampling and collector weighting inside each selected subgame remain intact. Small experiments do not cover all hand/context combinations; their missing examples are reported rather than inferred from total replay size.

The 3-max option enumerates both opponents from a configured street onward. Earlier sampled-opponent prefix correction is retained, so expected per-infoset action mass remains correct. The original validated partial collector remains available. Raw ESS gains and extra CPU cost are measured separately; retained 3-max variance remains a material limitation.

## Exact default schedules and allocations

`configs/train_hu_streets.json` and `configs/train_3max_streets.json` are independent track configurations. All schedules are individually editable per family and street, with strict positive-integer validation.

| Setting | HU | 3-max |
|---|---:|---:|
| Traversals / frozen cycle | 512 (256/player) | 384 (128/player) |
| Workers / trainer threads | 1 / 1 | 2 / 1 |
| A fit cadence | Every cycle if eligible | Every cycle if eligible |
| B fit cadence | Every third cycle if eligible | Every third cycle if eligible |
| New records required per specialist | 128 | 128 |
| Optimizer update cap per eligible specialist | 64 | 64 |
| Batch size / Adam LR | 64 / 0.0003 | 64 / 0.0003 |
| Replay capacity per family: PF/flop/turn/river | 20k / 10k / 10k / 10k | 20k / 10k / 10k / 10k |
| Accounted replay byte budget per family | 512 MiB | 512 MiB |
| Extra opponent enumeration | None | From river |

Each specialist retains its last fit training/validation metrics even across skipped cycles. Each specialist requires new retained training examples and nonempty independent training/validation groups. A missing group leaves that specialist unchanged with an explicit reason. Adam moments and steps persist. Historical/new examples use the existing iteration/reach-weighted replay objective; new examples are not silently oversampled. Fit logs expose actual updates and per-street budgets, not a misleading fixed epoch count.

A full A-eligible cycle has at most 256 updates per track; a cycle where B is also eligible has at most 512. It does not trigger sixteen fits merely because sixteen networks exist. The inactive track is not constructed or optimized in a dedicated run.

The 512 MiB replay budget is sample accounting, not a process RSS cap. Staged copies, encoded batches, optimizer moments, serialization, workers and Python allocators need extra memory. The bounded RSS experiment uses smaller replay and cannot certify full-capacity usage on an 8 GB VPS.

## Checkpoints, entrypoints and operational storage

- New checkpoint contract v5 declares `independent_streets_v1`, replay v1 and optimizer persistence. Per-track payloads contain four A/four B state dictionaries, trained flags, Adam states/steps, schedule/fit counters, coverage, sampler/card cycles, frozen version and Python/Torch/solver RNG states. Configuration, feature/action versions and source fingerprints are retained.
- Legacy v4 single-network checkpoints remain readable and resumable with their declared configuration. Changing a legacy checkpoint to the specialist configuration fails explicitly. No trained-weight copying or automatic migration occurs.
- Existing `train-hu.py` and `train-3.py` now select the street configurations and new `runs/hu-streets` / `runs/3max-streets` directories. Old `*-batched` checkpoints are untouched.
- New local entrypoint runs enable the existing bounded runtime storage: one current metric in the checkpoint, three rotating iteration checkpoints, and a bounded compressed diagnostic journal (256 records / 256 MiB). Experiment examples deliberately retain named evidence separately.
- Existing VPS resume loads its own checkpoint configuration. No service or production configuration was edited. A future switch requires explicitly starting a new v5 run directory, not overwriting the current v4 state.
- Migrate uses the new candidate directories; it does not activate experimental outputs on import or training completion. Cold exports fail with the missing streets listed.

## Exports, Supabase and browser

A track exports four small Average ONNX graphs and a catalog. Every route declares player count, street, `AVERAGE`, fitted model version, feature/action schemas, model hash, checkpoint source and coverage metadata. `checkpoint_source` is the SHA256 of the small self-contained Average inference checkpoint used for export; it is not the full replay checkpoint hash. All four routes must name the same source and distinct model files.

Publication seals and verifies all files, uploads immutable artifacts, reads them back, then advances the pointer. A partial river upload cannot activate an incomplete bundle. There is no new Supabase table required. A new street run starts its iteration clock at one, so replacing a higher-iteration shared release requires the explicit `publish_bundle(..., migrate_shared_release=expected_current_release_sha)` operation. It validates the old/new schemas and exact expected pointer; this operation was tested only in memory and never executed remotely. An old shared writer cannot subsequently overwrite a street catalog, regardless of its larger iteration number. Stop the old publisher before explicitly activating the new schema. Default remote retention stays at 24 releases with a pruning grace period; compact hourly history and local trace retention remain unchanged. Four graphs total about 296 KB/track in this architecture. Replay, Adam states and large coverage tables are never added to hourly browser history.

Python and TypeScript use identical public-street routing. Browser sessions are loaded on first use and released on policy refresh. Failed lazy downloads surface an error and may be retried; no alternate street or old artifact is silently substituted. Old explicitly declared single-model manifests remain supported. A published bundle swap stays atomic, even when individual street model versions differ because of independent schedules.

Overview uses preflop; Specific Spot and Test Live switch naturally by street; Bayesian range likelihoods use the hypothetical actor holding and public street through the same worker loader. Cache identities include bundle iteration and sorted private-card combinations. Existing compact controls, hover ranges, labels and layout remain unchanged. No new KPI panels, nested settings or explanatory tooltips were introduced.

## Validation and reproduction

Final suite results: 238 Python tests passed; the nine specialist contract tests passed again after adding persisted last-fit metrics. 156 browser tests passed. Production-config and local-config static builds passed. Actual Chrome ran all eight model routes, both Overview matrices, all Specific Spot streets and Test Live worker posterior updates.

Primary tests: `tests/test_street_networks.py`, existing runner/checkpoint/collector/publication/history/hybrid suites, `tests/run_browser_tests.mjs`, `tests/street_browser_smoke.mjs`. No tests rely only on tensor shape: they compare state dictionaries after resume, worker samples, exact action masses, ONNX probabilities and Bayesian posteriors.

Importable bounded example (does not publish):

```python
from pathlib import Path
from examples.street_network_demo import run_demo

result = run_demo(Path("runs/my-explicit-street-hu-study"), 2)
# For 3-max, use player_count=3 and a distinct explicit output directory.
```

Reproduce the quantitative cohort and studies with `training.street_experiments.run_closed_loop`, `fixed_replay_study`, `strategic_probes`, `postflop_probes`, `context_diagnostics`, and `browser_fixtures`. Their budgets and required explicit shared-baseline artifacts are documented in STREET_NETWORK_EXPERIMENTS.md. Fixed-replay diagnostics do not alter source checkpoints.

For real browser fixture execution, build with `NEXT_PUBLIC_POLICY_SOURCE=local npm run build` in `ui/`, then serve through the importable `tests.serve_street_fixtures.serve(Path("ui/out"), Path("runs/street-network-migration"))`. Start an isolated Chrome debug session on port 9436 and run:

```sh
POKER_UI_ORIGIN=http://127.0.0.1:3137 node tests/street_browser_smoke.mjs
```

This localhost fixture server maps only local experimental policy bundles; it does not publish them. An ordinary production build still uses the configured storage source.

## Remaining quality limitations

The migration is implemented; poker strength is not established. Seed-101 HU A folds AA about 72.7%, while B folds AA about 5.5%; this discrepancy exposes an unstable current regret model despite a less extreme historical average. 3-max weak-hand over-aggression and low retained strategy ESS remain. Tiny opening datasets contain missing AA/KK/AKs supervision. Some postflop conditional EVs improve, others regress. The exact small river fixtures do not certify general equilibrium quality.

All sixteen models are therefore experimental. The next substantial budget and acceptance checks are in STREET_NETWORK_EXPERIMENTS.md. No large training, automatic acceptance, production export, commit, push or deployment was performed.
