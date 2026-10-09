# Phase 2 acceptance

> Historical evidence: the old models and `artifacts/` were explicitly deleted during training finalization. Numerical results below describe previous bounded experiments, not currently deployed models. See [TRAINING.md](TRAINING.md) for the current configuration, empty catalog and launch/export workflow.

## Delivery status

**Implementation: complete for the tested, bounded workflows. Strategic quality: NOT YET VALIDATED in general poker.** The engine operates without neural models. Final choices come from legal-action search, explicit ranges and canonical chip settlements. Six behavior variants pass narrowly scoped prediction gates; none earns default strategic authority. All continuation variants remain rejected.

No deployment, published-policy replacement, GPU or multi-hour training was performed. The prior Phase 1 reports were explicitly deleted before this work and cannot serve as retained historical evidence. The existing code/tests were preserved; new evidence is independently recorded under `artifacts/phase2`.

## End-to-end acceptance A–H

| Scenario | Executed functionality and evidence | Result / boundary |
|---|---|---|
| A Complete HU hand | `examples/phase2_demo.py`; `test_hybrid_session.py`; fixture `phase2-hand-2.json`; actual static-browser smoke | Preflop through settlement, every action/reveal updates beliefs, each nonterminal decision produces legal mixed strategy and EV. Eight browser observations reach settlement. Explicit human-action likelihood assumption |
| B Complete 3-max hand | Same Python demo/fixture with count 3; full-hand and actual 3-to-2 tournament-transition regression; existing side-pot/all-in tests | Joint blockers and unknown folded hands retained. Eleven observations in final static browser hand. New hand creates explicit fresh priors and correct policy track |
| C Behavior sensitivity | Analytic latent-profile/hand posterior tests, known-profile held-out EV suite, off-tree posterior calibration | Bayesian examples match independent calculations; EV depends on selected behavior. This validates declared synthetic profiles, not real-human calibration |
| D Self-play lifecycle | `phase2_learning.py` complete trajectories, profile-specific distributions, grouped splits, 20/50+1 epoch fits, optimizer/data checkpoints, deterministic resumed weights, JSON/ONNX exports; downstream model-load comparison | Operational HU/3-max loop. Six narrow behavior prediction gates pass, two fail; downstream learned EV/posterior results do not justify replacing computed profiles. Protected Deep CFR two-track runner also resumes exactly |
| E Compute-quality benchmark | 8/32 vs independent128 rollout worlds; 16/64/256-world forests vs exhaustive-turn reference; 5/50/500 river CFR | Precision and policy changes measured, including stagnation/degradation. Adaptive allocation remains optional. Finite reference uncertainty retained |
| F Real browser | `tests/phase2_browser_smoke.mjs` against built static site; complete Specific spot HU/3-max hands, actual Test Live bot-action replay, 2.25 raise, profile/reference/exploitative controls, DEEP cancellation | Real Web Worker, no Python backend. Exact HU river/3-max river/turn fixtures have maximum measured EV/probability difference 0. Posterior Python/TS fixtures also pass |
| G Strategic evaluation | Heterogeneous exact river pool (472 rows), two 480-hand matched runs, 24 narrow and 24 broad/polarized state cases, 16 tournaments per count, published NN prior/direct ablations | Controlled action-EV improvement; realized chip-difference confidence intervals all include zero. Tournaments use explicit conservative/aggressive agents and are descriptive, not a full-hybrid tournament-strength certificate |
| H Failure modes | Existing hybrid tests plus new session/gate/worker tests | Missing NN works; unknown profiles fail explicitly; rejected continuation cannot load; zero-likelihood updates do not commit; insufficient sampled evidence is explicit; node exhaustion follows strict/degraded mode; unsupported off-tree mapping errors; cancellation publishes no stale partial result |

## Definition-of-DONE reconciliation

1. **Persistent Python/browser sessions:** implemented and exercised over complete hands, board reveals, actual bot responses and next-hand count changes.
2. **Nonuniform reactions:** computed hand-aware profile distributions drive default Python and browser UI broad continuations. Uniform remains an explicit baseline.
3. **Validated computed reference:** exact finite-game tables are accepted only after independent bounded-response checks; full ranges outside their scope use a disclosed conservative computed fallback.
4. **HU/3-max live reference:** both counts route to canonical river/chance CFR or explicit bounded behavior search. Three-player response gain is a robustness metric, not a two-player exploitability theorem.
5. **Objectives distinguished:** reference policies are separate from observation/exploitative policies; mixture assumptions and profile sensitivity are reported. Hand-chip delta remains the training objective.
6. **Meaningful offline data:** actual policy trajectories with conditional likelihood labels, independent rollout values, exact finite tables, generated/retained coverage and frozen replay sweeps exist. Retention has an opt-in inverse-inclusion-corrected allocation.
7. **Continuation gate:** varied roots, suit-aware inputs, independent labels, baseline comparisons, physical bounds and explicit rejection are implemented.
8. **No failed-model promotion:** additional fitting invalidates prior acceptance. Final fitted epochs are revalidated; rejected values stay disabled; downstream failures prevent default behavior-model adoption.
9. **Full-hand browser:** original pages and tournament table retain their shell; range history is reconstructed through real state transitions. Stale results are rejected by request IDs and transition guards.
10. **Calculated action EV:** every legal candidate is evaluated under stated beliefs/policies. Equity proxies belong to assumed behavior, not substituted bet EVs.
11. **Correctness and parity:** 195 Python tests and 134 browser tests pass; real static Chrome fixtures and production build pass.
12. **Ablations retained:** uniform / computed / profile / published neural prior / learned behavior / rejected value comparisons and performance reports remain local.
13. **Reproduction:** importable Python example and experiments, exact fixtures, static browser scripts and recipes below.
14. **Roadmap reconciled:** every original phase 0–16 and scenario A–H appears in the roadmap with owners, tests and limitations. No model-convergence or general-strength claim is used as a completion criterion.

## Model and strategy status

| Component | Implemented | Bounded training/computation | Validation | Accepted use / remaining work |
|---|---|---|---|---|
| Exact local tables | Yes | 500-iteration independent river cases | Independent response gain; six of eight expanded tables pass 0.05 threshold | Only covered finite information sets; recompute for different ranges/state |
| Conservative/profile policies | Yes | Deterministic 24-deal equity proxy + explicit synthetic parameters | Python/TS parity, hand sensitivity, paired EV tests | Explicit fallback or declared population; no equilibrium/human-accuracy label |
| Published average HU/3-max | Preserved | No new fit or mutation | Direct/prior finite-game ablations; SHA-256 unchanged | Optional prior/baseline; cannot claim global credibility |
| Behavior models | Yes | Eight variants, 12 trajectories/profile/count, widths32/96 | Six narrow prediction gates pass; aggressive three-player variants fail; OOD and downstream tests retained | Explicit profile-scoped optional use; not the default because downstream improvement failed |
| Continuation models | Yes | 16 variants over conservative/aggressive labels; class169/combo1326, widths32/96 | All fail raw-value criteria; independent exact/rollout probes retained | Disabled; no learned-leaf acceleration accepted |
| Deep CFR advantage/average | Preserved and instrumented | Tiny frozen fit and optional protected-memory runner lifecycle | Exact deterministic resume, collector and feature regressions | Training infrastructure works; tiny new fits establish no playing strength |

Published average hashes remained:

- HU: `d81b86908bddac4d174617f3d40b211851d24cfb1ea57fc4bc9f72097b16e6b3`
- 3-max: `37f615ca1d2d183515a7e877f343154e7e3c56ec19814cb92fa2ae80be8e6b9b`

## Evidence and reproduction

From the repository root, the actual completed verification is recorded in `artifacts/phase2/python-tests.log`, `browser-tests.log`, `static-build.log` and `evaluation/browser-static-smoke.json`.

```sh
PYTHONPATH=tests:. .venv/bin/python -m unittest discover -s tests -p 'test_*.py'
node tests/run_browser_tests.mjs
node ui/node_modules/typescript/bin/tsc --noEmit -p ui/tsconfig.json
```

The final Python suite completed 195 tests in 49.294 seconds. Browser unit/parity tests completed 134 with zero failures. Production static export completed successfully. ONNX exporter tracer warnings are retained in logs; numerical exported-artifact checks passed for the tested inputs, not every conceivable input/batch.

### Python demonstration

```python
from examples.phase2_demo import demonstrate

hu = demonstrate(2)
three_player = demonstrate(3)
print(hu["steps"][0]["analysis"])
print(hu["utilities"], three_player["utilities"])
```

The example plays an explicit passive line after a legal 2.25 raise and computes every decision. Its actions are a declared synthetic observation source, not secretly claimed to have been sampled from the engine's own mixed policy. For custom play use `HybridSession.observe_action(..., likelihood_source=...)` and `session.analyze(budget, mode="reference"|"exploitative")`.

### Browser demonstration

Build using `npm run build` in `ui/`, then run `node ui/scripts/serve-static.mjs --port 3108` from the repository root and open `http://localhost:3108`.

- In **Specific spot**, choose HU or 3-max and cards before starting the hand. Select explicit likelihood profiles and compute budget, then analyze and play legal actions. Posterior status and range views update through settlement; actual raise amounts are preserved.
- In **Test Live**, select a Hero seat and start a hand. The same analysis component reconstructs earlier bot actions and retains posterior beliefs after subsequent bots act. Bot likelihoods are explicitly uniform for these existing test opponents; continuation assumptions can be chosen separately.
- Use Reference/Exploitative controls to see which continuation assumptions change. Baseline probabilities and local action EV/probabilities are separate columns. A DEEP analysis can be cancelled without publishing a stale result.

`tests/phase2_browser_smoke.mjs` exercises this actual UI/worker via an explicitly running Chrome CDP endpoint. It accepts `POKER_UI_ORIGIN`, `POKER_CDP_ORIGIN` and `POKER_REPORT_PATH` environment settings; inspect the script defaults before running. No background Python service is required.

### Bounded experiment recipes

These are separate explicit operations, not an instruction to rerun every fit during inference. Use a new output directory to preserve delivered reports.

```python
from pathlib import Path
from hybrid.quality import evaluate
from hybrid.reference import ProfilePolicy
from hybrid.policy_source import UniformLegalPolicy
from hybrid.matches import matched_hands
from hybrid.phase2_learning import fit_behaviors, fit_values
from hybrid.reference_experiments import reference_ablation
from hybrid.robustness import forest_generalization

out = Path("artifacts/phase2-reproduction")
out.mkdir(parents=True, exist_ok=True)
policies = {
    "uniform": UniformLegalPolicy(),
    "conservative": ProfilePolicy("conservative"),
    "known-profile": lambda s: ProfilePolicy(
        "tight_passive" if "medium" in s.name else "loose_aggressive"
    ),
}
evaluate(out / "ev.json", candidates=policies,
         teacher_family="hand-conditioned", samples=(8, 32), reference_samples=128)
# Run individual additional experiments when needed:
# matched_hands(out / "hands.json", deals=16, samples=32)
# reference_ablation(out / "references", roots_per_count=4)
# forest_generalization(out / "forest.json", seeds=4)
# fit_behaviors(out / "behavior", hands=12, epochs=(20, 50), widths=(32, 96))
# fit_values(out / "values", per_cell=2, samples=64, epochs=(20, 50))
```

For optional opening protection, load an existing training config and add `config["replay_opening_fraction"] = 0.25` before constructing `TrainingRunner`. It requires external sampling and at least two slots per memory. Capacity is divided into opening/other strata; byte budgets are split into two explicit pools. Keep `stratified_recorded_opening` in the root sampler where intended. Original configs without this key keep their original hash and reservoir behavior. Checkpoints record the allocation and reject mismatches. This quota preserves the stream's expected weighted loss total using inverse inclusion probabilities; it does not make missing hand classes appear or guarantee a converged network.

## Remaining limitations and next measured compute proposal

- Sparse exact table coverage is the largest reference-policy limitation. Broad conservative continuations are auditable synthetic approximations, not balanced play. Better exact coverage should precede indiscriminate NN training.
- Public factors capture blocker dependence but not all arbitrary within-profile joint correlations. Python mixtures retain profile/hand correlation; browser currently selects explicit profiles rather than maintaining a latent profile mixture.
- Re-solving lacks upstream safety constraints sufficient for safe resolving. Three-player CFR has no general HU equilibrium guarantee. Sampled forests can still overfit; larger independent-world evaluations remain necessary.
- Human action likelihoods are assumptions. An unsupported profile, impossible observation or insufficient evidence produces an explicit error, not a confident fabricated posterior.
- Current confidence estimates quantify sampling variation under the assumed policies, not the truth of those policies. Model/range/continuation uncertainties are not collapsed into a misleading small SE.
- Broad reference errors and realized match intervals preclude a general-strength claim. The full-hybrid tournament objective has not been optimized or statistically validated.

A next research run should first expand **independent scenarios**, not epochs: for example ten roots per suite cell (240 roots), 64 candidate worlds and 512 independent reference worlds, plus 256 matched private-deal seeds at 32 worlds against the existing five profiles. The latter is 7,680 complete candidate hands, 16 times the delivered 480-hand budget. Use the measured seconds in the current JSON reports to project cost on the target machine; time a small shard before authorizing the whole run. This can exceed the bounded diagnostic budget and has **not** been launched.

For continuation learning, a proposed first shard is `per_cell=10`, `samples=256`, widths32/96 at20/50 epochs, with the current disjoint-root split and continuation profile kept explicit. Relative to the delivered conservative label budget this is 20 times the rollout count: approximately nine minutes of label generation by linear extrapolation from 26.91 seconds, before fitting and wider-root effects. This estimate is not a promise; measure the shard and stop on poor signal. Promotion requires raw held-out error below the configured threshold, improvement over table/rollout alternatives, and independent online action-loss/cost advantage. More training alone does not authorize promotion.

## Subsequent hover acceptance

The opponent-seat inspector is now implemented (it was not part of the original Phase 2 UI delivery). 136 browser tests pass, including independent exhaustive-product marginal comparison, uniform full-range normalization and invariance to simulator opponent cards/deck. A real-pointer Chrome run on the production static export inspected both opponents after actual bot responses: 169 cells each, clickable AA combo details, and unchanged Python/worker parity. Evidence: `artifacts/phase2/evaluation/range-hover-smoke.json` and `range-hover-0.png` / `range-hover-1.png`. Uniform opponents correctly retain uninformative action likelihoods; board and Hero blockers still affect their matrices.
