# Solver authority correction

Status: IMPLEMENTED LOCALLY; bounded validation complete; no production rollout. Existing production checkpoints are audit inputs, not a strategic reference. Strategic strength is NOT VALIDATED.

## Required work

| Requirement | Status | Implementation / acceptance |
|---|---|---|
| Live recommendations originate from search, not B | DONE | TestTable + HybridAnalysis + existing worker; computed EV, state-keyed results, cancellation, explicit errors |
| Preserve published bots as controlled opponents | DONE | B only supplies adversary actions and their Bayesian likelihoods; not Hero recommendations |
| Resume explicit CFR regrets and average masses | DONE | RiverSolver; identical interrupted/uninterrupted finite-game results; reject changed ranges/forest |
| Persist calculated supervision with range and continuation scope | DONE | hybrid/knowledge.py; training/solver_runner.py; no A-generated strategy required; absolute iteration targets, storage limits |
| Context-conditioned specialist fitting and numerical health diagnostics | DONE, experimental | hybrid/distillation.py; ml/health.py; training/solver_repair_study.py; no candidate accepted |
| HU/3-max and all-street bounded experiments | DONE, bounded only | 24 independent shallow roots, all sixteen specialists; comparison against explicit calculations |
| Browser validation | DONE | 161 browser tests, build/lint, actual Chrome HU/3-max decisions, cancellation, three Python/worker parity fixtures |
| Replace OVH production training | NOT APPLIED | Existing services/checkpoints/publications were not modified. Tiny test curricula and rejected neural candidates are not a production replacement |

## Algorithmic boundaries

Local CFR is belief-conditioned, not safe re-solving without upstream constraints. Broad-range rollout action values are responses to an explicit continuation model, not equilibrium values. Model-free calculation is required; model-free empirical opponent likelihoods are not inferred from nonexistent data. Published bots can still supply their own known action likelihoods.

References: [Deep CFR](https://proceedings.mlr.press/v97/brown19b.html) explains why A owns the existing Deep CFR strategy; [depth-limited solving](https://papers.nips.cc/paper/2018/file/34306d99c63613fad5b2a140398c0420-Paper.pdf) explains why fixed continuation leaves cannot be described as general equilibrium solving. We reuse the existing finite-game solver and retain these limitations.

## Existing evidence

See STRATEGY_CHAIN_AUDIT.md. A's dead ReLU head is measured; its historical cause remains unidentified. Changing an already collapsed head to LeakyReLU failed held-out validation. More depth and production retraining are not authorized repairs on that evidence.

## Delivered behavior

Test Live no longer calls B to populate Hero's action probabilities. It automatically requests actual action values from the worker using persistent public posteriors. Buttons show computed probabilities and EV in current BB. Results belong to an exact hand/session/observer key; changing state or cancelling clears their use. Failure never substitutes NN probabilities. Human play can continue when analysis is unavailable.

Overview remains a viewer of the published approximation. Hourly Training evaluations still measure the exported NN policy, not the online solver; their chart now says "NN policy" to keep that distinction visible.

The default live budget is 100,000 nodes, 128 paired worlds, depth 1, and 10 CFR iterations when the exact small-range river route applies. Broad ranges use explicit conservative scripted continuations. This is a model-conditioned response calculation, usually with a pure root best response, not a solved full-game equilibrium or a calibrated reaction model of the published opponents. The status identifies Monte Carlo versus CFR and unresolved rankings. Published bots and their range likelihoods still use the published models; human-action likelihoods retain the existing conservative assumption. Test Live therefore still needs published opponent models to run that particular opponent population; the calculation engine itself works without any NN.

Offline, SolverRunner refines declared finite games round-robin. Its primary checkpoint is the explicit regret/average table. A complete forest fingerprint includes information sets, action amounts, private-deal masses and terminal/rollout utilities. Changed games cannot reuse regrets. Per-game writes are atomic; the disk cap concerns retained artifacts (atomic replacement temporarily requires additional space for one file). No implicit fits, publication, or eviction occurs. The runner is importable and does not replace train-hu.py or train-3.py silently.

Stored labels retain the public root, exact ranges, continuation identity, chance seed, budgets, warnings and split. A targets are accumulated counterfactual regrets divided by completed iterations and current BB; B targets are accumulated own-reach average strategies. Zero-own-reach B rows are excluded explicitly. These are a separate finite-game distillation population, not relabeled Deep CFR external samples. Cached tables require the same root/range prior and fail on uncovered information sets.

SolverStudents owns four A and four B specialists per player-count track. Each has its own optimizer and update count. Distillation equalizes context mass while preserving relative target weights inside each context; it does not claim to preserve the old global Deep CFR objective. Splits are checked for shared public roots, and complete dataset fingerprints bind deterministic resume. The experimental A head uses normalized encoder features, a SiLU branch and a direct linear residual path. Fresh residual/baseline experiments share initial encoder weights. Existing checkpoints and the production feature/export schemas are unchanged. Experimental student checkpoints use a different schema and are not publishable as existing production checkpoints.

## Evidence

- Python: 41 hybrid regression tests, 10 Deep CFR tests, 4 context replay tests, 5 collector tests, and 6 authority tests pass. Authority tests include exact resume, changed-game rejection, storage bounds without corrupting the previous artifact, optimizer resume for all eight specialists, gradient flow through the direct path and exact observation-key parity.
- Browser: 161 tests pass; production static build and lint pass. Real Chrome records at runs/solver-authority/browser.json show HU/3-max recommendations with worker EVs, no stale probabilities after cancellation, and exact parity for two river fixtures plus one turn fixture.
- Observed live worker times: approximately 0.35 s and 0.77 s in the final two sampled hands, including belief work. These are examples, not a latency distribution or server benchmark.
- The 24-root solver/student study took 19.64 s under cProfile, peak process RSS 282.75 MiB. All roots are shallow and each range has two holdings. Training provides only two preflop/flop/turn root labels per training track and 8/24 river information sets: sufficient to exercise the workflow, emphatically insufficient to establish learning coverage.
- Sixty-four updates per specialist per variant; student checkpoint approximately 1.83 MB (original head) or 1.94 MB (residual head). All candidates remain unaccepted.
- On two held-out exact finite river games, summed bounded response gains after 200 explicit CFR iterations were 0.00872 BB (HU) and 0.06721 BB (3-max). B students gave 0.93874 and 2.31849 BB. These comparisons demonstrate a gap on those games, not general poker strength.
- A learned B prior at equal nodes helped the HU example slightly but worsened the 3-max example and increased runtime. It is not enabled by default. Full per-budget results and every street's root-deviation diagnostics are retained in report.json.

## Actual replay repair study

The immutable 3-max checkpoint #2494 preflop replay was fitted with four fresh variants, 256 updates each, batch 128, one CPU thread and preserved grouped splits. Gradient/activation traces were recorded every 64 updates. The existing model had 82.81% dead heads in the fixed 128-opening probe; this is a probe statistic, not a replacement for the earlier whole-replay estimate.

| Fresh variant | Opening test weighted MSE |
|---|---:|
| ReLU, original objective | 88.9265 |
| ReLU, context-balanced objective | 88.0315 |
| Residual, original objective | 88.6818 |
| Residual, context-balanced objective | 88.1228 |
| Zero prediction baseline | 87.8362 |

None passes this opening gate. No fresh ReLU collapse occurred within the bounded 256-update trace, so the experiment does not identify the historical cause of collapse. Residual connectivity fixes the demonstrated blocked-gradient mechanism, but does not establish useful regret predictions. Nothing from these experiments was exported or activated.

## Profiling correction

cProfile identified recursive dataclass copying in observation keys/history transport as a significant finite-game overhead. The implementation now serializes fields directly, with exact string-parity regression tests. A retained 8,000-key microbenchmark measured 0.980 s before versus 0.154 s after (6.35x for that operation only). The earlier full-study time is not presented as an optimized end-to-end benchmark. Profiles and numerical results are under runs/solver-authority/.

## Reproduction

```python
from pathlib import Path
from examples.solver_authority_demo import demonstrate

# Refuses an existing directory; explicit independent experiment output.
report = demonstrate(Path("runs/my-solver-authority-study"))
```

For iterative calculation without neural fitting:

```python
from pathlib import Path
from hybrid.solver_experiments import bounded_roots
from hybrid.state import ComputeBudget
from training.solver_runner import SolverRunner, SolverTask

tasks = [SolverTask(f"hu-{split}-{root.street}", root, ranges, split)
         for split in ("train", "validation", "test")
         for root, ranges in bounded_roots(2, split)]
runner = SolverRunner(Path("runs/my-computed-hu-games"), tasks,
                      ComputeBudget(iterations=32, samples=32, max_depth=1, seed=818))
runner.refine_to(128, max_seconds=60)
# Calling this again resumes unfinished games; completed iterations are not repeated.
```

Actual browser test: serve the built UI with ui/scripts/serve-static.mjs on port 3149 and an isolated Chrome CDP on 9449, then run tests/solver_authority_browser_smoke.mjs. It requires existing published opponents but independently tests the model-free worker fixtures.

## Remaining product limitations and deployment

No Git push, Vercel deployment, OVH restart, publication, Supabase mutation or production checkpoint change occurred. The site changes require the normal frontend deployment. OVH has not been redirected to the new solver runner. The old Deep CFR pipeline remains available and its existing production status was not re-audited here.

Before a production offline replacement, specify and benchmark a wider curriculum of legitimate public roots/range priors, including deeper stacks and varied preflop responses. The delivered runner can refine these games but does not turn an arbitrary tiny curriculum into full NLHE knowledge. Fixed continuation forests can overfit sampled worlds; the broader live response route still depends on synthetic continuation assumptions. Current neural inputs do not encode exact range priors, so student generalization across distinct priors is unaccepted. No safe re-solving or multiplayer Nash guarantee is claimed.

## Subsequent range-input correction

The original measurements above describe the v1 students. Their missing-range limitation is now addressed by root-only range-conditioned students with a separate v2 checkpoint contract. See [RANGE_CONDITIONED_STUDENTS.md](RANGE_CONDITIONED_STUDENTS.md) for the 224-game study, changed target semantics, sixteen-specialist resume tests, numerical evidence and remaining quality failures. Production Deep CFR remains separate; the experiments do not constitute its replacement.

Next executable action: improve finite-forest label precision and independent-root coverage using the measured sample-budget study, before selecting any OVH workload.
