# Expresso implementation and validation

## Readiness and current objective

**SMOKE-TRAIN READY** for bounded current-hand validation and tiny fits. The solver optimizes the current hand's terminal chip delta,
conditional on its exact private/public state. It does not traverse later hands,
optimize tournament winner utility, or use ICM. Tournament persistence,
elimination, heads-up transition, and winner-take-all payout remain environment
behavior. These are distinct contracts.

Earlier full-tournament traversal and outcome diagnostics are historical
baselines for a retired objective. They must not be interpreted as numerical
or performance measurements of current per-hand training. No substantial
training has been launched and no model is claimed GTO.

## Engine, units, and blind schedule

`TournamentState` owns stable player IDs, conserved physical stacks, active
players, rotating button, hand number, immutable `BlindSchedule`, blind level,
completed hands, and winner. The initial chip unit is the initial BB: stacks
start at 25 chips each and total chips remain 75. Increasing blinds changes
chip amounts posted, never rescales bankrolls or the conserved total.

The explicit default **simulation preset** is SB/BB 0.5/1 from hand 1, 1/2
from hand 11, 2/4 from hand 21, 4/8 from hand 31, 8/16 from hand 41, and
16/32 from hand 51 onward. This is not an official Expresso timetable.
`BlindSchedule.fixed()` remains an explicit finite-fixture configuration.
Stages must start at hand 1, increase in hand index, never reduce either blind,
and increase at least one blind. New levels apply only after settlement at
`start_hand`; each hand keeps its own blind configuration and metadata.

Live public amounts are **current BB**: stacks, pot, street bets, amount to call,
legal raise targets, and history amounts divide physical chips by that hand's
BB. Thus the displayed chip total is `75 / current_big_blind`, not permanently
75 BB. The table exposes physical chips per current BB and blind-level index.
Accumulated Hero P&L sums each settled hand's delta divided by that hand's own
BB; it is a historical BB sum and does not rescale when a later level changes.

`HandState` owns exact cards, action order, contribution accounting, pot,
minimum/full raises, reopening rights, history, showdown, and settlement.
Stacks behind plus pot conserve chips; after settlement the pot is zero.
Identity differs from BTN/SB/BB role. In heads-up, button/SB acts first preflop
and BB first postflop. The 3-player-to-HU transition advances the BB before
assigning the new SB/button, avoiding consecutive BB posts (TDA rule 36C).

Arbitrary legal raise-to amounts are supported by the engine. Short all-ins
are distinguished from full raises. Calls are capped by actual live opposing
contributions when every opponent is all-in, preventing phantom calls against
a short posted BB; the full nominal BB remains due when multiple players can
still bet. Focused cross-language fixtures cover that dry-side-pot correction.
Side pots, refunds, folded contributions, ties, and split awards are explicit.
Invalid states and amounts raise errors with their relevant values.

`actions.py` and its parity-tested browser counterpart define the same masked,
deduplicated 13-action abstraction. Preflop targets are 2.0/2.5/3.0/4.0 times
BB or the current raise-to; postflop targets are
`highest + fraction * (pot + to_call)`. Minimums are enforced and equivalent
stack-capped actions collapse to one all-in branch. Abstract IDs refer to
internal chip targets; browser views convert their computed amounts to current
BB exactly once. Short all-in calls use `CALL`.

## Current-hand observation and artifact contracts

`Observation` version 3 contains exact Hero cards, board slots/unknown masks,
street, legal-action mask, continuous numeric values, and full **current-hand**
public history. `observe(tournament)` is exactly `observe(tournament.hand)`.
Prior hands' cards/history are excluded from conditional cEV input. Model
history keeps within-hand token index 1; numeric metadata and recall retain the
actual tournament hand number and blind-level context.

Amount features and history amounts are normalized by `25 * current_big_blind`.
The appended `blind_level_index` is unbucketed, and `chip_unit_big_blind`
records the physical chip scale. Hand number is normalized by 25. Raw utility
and regret targets remain conserved initial-BB chip deltas, allowing exact
zero-sum hand results. Neither hidden opponent cards nor future deck enter the
observation. Winning by folds does not reveal opponents' cards in live play.

Checkpoint version 3, architecture `cards8_numeric32_historyGRU32_head64_v3`,
state version 3, replay version 4, and ONNX manifest version 2 reject prior
contracts. The manifest binds `amount_units=current_big_blinds`,
`utility_units=initial_big_blind_chips`, `history_scope=current_hand`, and
`payout_scope=winner_take_all`, plus actions, features, model SHA256 and coverage.
The payout scope describes the environment; the only learning objective is
`hand_chip_delta`. Retired tournament-winner models are never converted or
silently loaded.

## Reference traversal and averaging

`ExternalSamplingMCCFR` freezes the current regret-matched strategy for each
iteration. Every traverser decision recursively evaluates every legal action,
including later traverser decisions in each continuation. Opponents are sampled
from their current strategy. Chance deals are sampled with injected RNG state.
The sink receives `action_value - strategy_weighted_node_value`; opponent
sampling probabilities do not multiply these regret targets. Regret matching
uses positive cumulative regrets and a defined uniform distribution when all
legal positive regrets are zero. Cumulative regrets are not clipped.

Continuation is pruned only when the traverser's payoff is already fixed:
a folded player in the hand chip-delta objective has its exact settled chip
loss. Tournament contexts are unwrapped to the current hand for learning;
later hands and tournament elimination utility are not part of that traversal. This removes irrelevant opponent continuations without estimating or
truncating a live traverser's value. All legal traverser actions still branch
recursively whenever that player's payoff remains unsettled.

Average-policy collection now follows one separate full-support uniformly
sampled trajectory for every player's actions, including the averaging
player's. At each averaging-player node, a strategy sample has weight
`own_strategy_reach / sampled_all_players_action_prefix_probability`. Updating
own reach uses that player's current strategy probability, while the sampled
prefix multiplies the uniform action-selection probabilities at every decision.
Chance is sampled from its true distribution; its reach cancels in the
normalized average at a perfect-recall infoset. A zero own reach stops the
trajectory because subsequent samples have zero weight. This replaces
own-action enumeration in the average pass while retaining the same
own-reach-weighted average target. The tabular reference averages iterations
uniformly. Deep CFR training additionally weights samples linearly by iteration.
Current regret-matched play and average policy remain separate objects;
evaluation and the UI query the latter.

Node/depth budgets raise `TraversalBudgetExceeded`. An unsuccessful iteration
does not publish partial solver updates. The separate averaging pass can have
high variance and large importance weights; its correctness does not establish
practical convergence or full-game approximation quality.

## Approved outcome-sampling integration

`outcome_sampling.py` implements the estimator recorded in
[the outcome-sampling contract](outcome-sampling.md). Select
`traversal_mode="outcome_sampling"` explicitly in the Deep CFR iteration.
Every player samples one legal action from
`q = (1 - epsilon) * target_strategy + epsilon * uniform_legal`, initially with
`epsilon=0.6`. Importance-corrected reverse values produce regret targets at
all visited traverser decisions; a separate own-reach/prefix correction weights
average-policy samples on that same path. Prefixes and continuation ratios are
tracked in log space. A zero target reach is distinguished from numerical
underflow. No importance clipping, artificial leaf payoff, hand-profit
substitute, or retry-until-short-success procedure is used.

Traversal mode is explicit in task, generated-sample, replay, and checkpoint
metadata. Replay persistence is version 4 and rejects mixed/incompatible
traversal contracts. Budgets and numerical failures retain diagnostic details
and abort atomically without publishing partial training samples. Exact settled hand payoff can stop a traverser path early. Full-tournament
paths are environment diagnostics, never the current training target.

**Historical retired-objective evidence:**
[the earlier full-tournament diagnostics](outcome-diagnostics.json) contain all 96 attempts:
12 seeds, two target policies (uniform and mildly skewed), 24 independent full
tournaments, and 72 traverser paths, each capped at 300 nodes/depth. Total wall
time was approximately 0.67 seconds on CPU. All 24 full tournaments completed;
60 traverser paths reached a tournament terminal and 12 stopped at exact settled
payoff. Full-tournament hand lengths had median 5, p95 12.7, and maximum 22.
These empirical finishes do not prove almost-sure termination or finite
importance variance on the infinite-horizon game.

The raw importance distributions already have extreme tails:

| Quantity | Median | p95 | Maximum |
| --- | --- | --- | --- |
| Average-policy sample weight | 3.47149e5 | 1.4948e16 | 3.12184e24 |
| Total regret importance correction | 5.98060e3 | 3.27711e12 | 1.74496e21 |

The maximum absolute regret target was 7.75540e20. No raw weight underflow or
overflow occurred in this probe. A separate one-node guard probe retained all
eight censored attempts; they are failures, not usable successful-subset
training data. The importable `scripts.outcome_diagnostics.run_probe` records
raw rows, log/weight quantiles, structural-zero counts, representational
under/overflow, and full-terminal versus early-settled paths for that preserved baseline. It adds no CLI
and launches no training.

Those historical finite float64 targets could overflow float32 squared error:
the largest exceeded `sqrt(float32_max)`, approximately 1.84e19. The retired
objective is no longer a supported learning path. These measurements do not
establish the same target magnitudes for current per-hand learning. Current
loss validation still rejects unrepresentable inputs/outputs and never clips
regret targets. [Current hand diagnostics](hand-diagnostics.json) evaluate the
new objective separately. Only bounded validation and tiny controlled neural
fits have run; approximation quality, variance, and representative resource
costs still require validation before FULL-TRAIN READY.

## Deep CFR and worker ownership

Each player identity has an `AdvantageNetwork`; a separate shared
`AveragePolicyNetwork` learns strategy samples. The modest encoder combines
exact-card embeddings, numeric features, and a full-history GRU, followed by a
64-wide head. Each network has 16,465 parameters. Three advantage networks plus
one average network total 65,860 parameters; float32 parameters alone occupy
approximately 0.25 MiB. Optimizer state, activations, replay, and Python objects
are additional memory.

Advantage targets are masked CFR instantaneous regrets; training uses masked
squared error. The average network produces a masked distribution and uses a
weighted distribution loss. Metrics include training/held-out losses, nodes,
sample counts, traversal values, and average-policy change on replay probes.
Tiny held-out splits are smoke diagnostics, not reliable generalization scores.

`ReservoirMemory` uses uniform Algorithm R sampling with explicit capacity,
schema, objective, seed, iteration, player, mask, target, sampling weight, and
model version. Memory persistence uses schema version 4 and rejects incompatible
state/action/objective contracts. Memory also has an explicit retained-object byte budget; overflow
fails rather than silently truncating histories or changing samples.

At an outer iteration, `ModelSnapshot` freezes all advantage weights and its
version. Seeded `TraversalTask`s generate samples under that snapshot. Workers
do not train or merge local regret tables. `collect_samples` supports direct
single-worker execution and spawned processes, returns task-ordered results,
and rejects incompatible versions. Only the central trainer aggregates replay,
fits new models, and publishes the next complete version. Replay is copied to
stage updates; budgeting must include both old and staged copies and in-flight
worker results. Each task has a 64 MiB combined generated-sample byte budget. Collection also
has a 256 MiB aggregate byte budget and at most one pending task per worker.
Frozen weights are serialized as ordinary bytes to avoid implicit tensor shared
memory ownership. Overflow rejects the batch rather than returning partial
results. Training and evaluation use bounded batches; stochastic training
normalizes loss weights against the global mean sample weight.

Average checkpoint exports identify architecture, state/action/numeric schema,
objective, iteration, and observed player counts. Missing or incompatible files
fail explicitly. Unsupported player counts return unavailable coverage.
A supported count does not prove that a particular poker state was covered or
that the approximation is accurate; the UI labels neural output experimental.

## Browser runtime and cross-language parity

The frontend is a static Next.js export (`output: "export"`), built into
`ui/out/`. `npm run dev` runs Next.js alone; `npm run build` then `npm start`
serves the export through `ui/scripts/serve-static.mjs`. There is no live Python
engine service, API route, or training process in the frontend runtime.
Development `.next-dev` and production `.next` output directories are separate.
The superseded `server.py`, `tests/test_service.py`, API proxy route, and
`tests/ui_proxy_smoke.mjs` are retired.

`ui/src/lib/poker/engine.ts`, `actions.ts`, `evaluator.ts`, and `observation.ts`
implement the browser hand/tournament, legal actions, card evaluation, and exact
state schema. Python remains the offline training reference. The two languages
are validated against checked-in fixtures generated from Python rather than
allowed to evolve as untested independent rule interpretations.
`tests/browser_engine.test.ts` compares every transition's player order/roles,
stacks, street/total contributions, pot, actor, board, raise context, legal
amounts, history, terminal awards, elimination, button, and complete rich
observations including parsed recall. It covers 37 full hands (18 seeded random
deals and focused rule regressions), nine 3-player-to-HU transitions plus a four-hand blind-progression sequence spanning
all button/eliminated-seat combinations, every hand category, and 100 bounded
seven-card evaluator comparisons. The 51 parity tests passed, and the checked-in
fixtures reproduce exactly from current Python. Evaluator ordering is compared
rather than assuming both languages use identical numeric rank codes.

`BrowserTable` in `game.ts` drives local play and scripted opponents.
`TestTable.tsx` preserves the original visual components and layout while
rendering persistent tournaments and canonical actions; `page.tsx` defaults to
Test Live. Overview and Cas précis retain their original analysis views and
packed-policy helpers, explicitly marked as legacy 50 BB single-hand analysis.
Those helpers and data never determine Test Live transitions or probabilities.
A new neural policy is not forced into a 169-hand heatmap.

`ml/export_onnx.py` exports the explicitly chosen average network as a singleton
batch ONNX graph with dynamic full-history length, plus a strict manifest
binding the bytes by SHA256. Export validates original and extended-history CPU
inference against PyTorch before publishing either file. Browser inference in
`onnx-policy.ts` uses locally served ONNX Runtime WebAssembly, one CPU thread,
and no GPU. User-selected model/manifest pairs must match architecture,
state/action/numeric/history schemas, objective, player count, and digest.
Missing coverage is explicit; malformed outputs fail rather than silently
returning a uniform distribution. No substantially trained per-hand average policy is bundled.

## Important source changes

| File | Responsibility |
| --- | --- |
| `poker_game_expresso.py` | HandState, HandPlayer, ActionEvent, authoritative betting/order/side pots and invariants |
| `classes.py` | Validated Card/Deck primitives for isolated push/fold tools |
| `tournament.py` | Persistent environment lifecycle, elimination, HU, levels, button rotation and payout |
| `blind_schedule.py` | Explicit immutable hand-index blind schedules and simulation preset |
| `actions.py` | Canonical 13-action IDs, legal sizes and deduplication |
| `infoset.py` | State v3, exact current-hand observations, current-BB features and hand/blind metadata |
| `cfr_solver.py` | Recursive external-sampling reference, regret matching and tabular average-policy oracle |
| `outcome_sampling.py` | Explicit single-trajectory importance-corrected MCCFR alternative |
| `scripts/outcome_diagnostics.py` | Bounded full-tournament/traverser path and importance diagnostics |
| `policy.py` | Versioned tabular average-policy persistence and explicit coverage |
| `config.py` | BB defaults, explicit modest budgets and new artifact paths |
| `ml/memory.py` | Versioned bounded reservoir memories |
| `ml/model.py` | Card/numeric/history encoder and separate network heads |
| `ml/train.py` | Central bounded-batch training and held-out metrics |
| `ml/deep_cfr.py` | Frozen snapshots, traversal samples, atomic iterations and average-model export/query |
| `scripts/parallel_cfr.py` | Spawned traversal generation without local learning |
| `scripts/benchmark_policy.py` | Scripted smoke opponents and bounded tournament evaluation |
| `scripts/estimate_infoset_coverage.py` | Coverage queries against the new state contract |
| `evaluation.py` | Bounded conditional-subgame policy and infoset-consistent best response |
| `scripts/smoke_validation.py` | Small deterministic validation and cost evidence |
| `ml/export_onnx.py` | Offline validated average-policy ONNX export with SHA256 manifest |
| `ui/src/lib/poker/{engine,actions,evaluator,observation}.ts` | Parity-tested standalone browser rules and observations |
| `ui/src/lib/onnx-policy.ts` | Strict optional CPU WASM model loading and policy queries |
| `ui/scripts/{prepare-onnx,serve-static}.mjs` | Local runtime asset staging and static export hosting |
| `ui/src/lib/game.ts` | Browser-owned table/controller and scripted opponents |
| `ui/src/lib/{infoset,policy}.ts` | Retained legacy-only secondary analysis helpers |
| `ui/src/components/TestTable.tsx` | Persistent tournament interaction |
| `ui/src/app/page.tsx` | Test Live default and secondary analysis surfaces |
| `tests/generate_browser_fixtures.py` | Importable deterministic Python fixture generator |
| `tests/fixtures/browser_parity.json` | Canonical full-transition and observation fixtures |
| `tests/browser_engine.test.ts` | Cross-language rules, state, history and evaluator parity |
| `tests/browser_table.test.ts` | Browser controller, RNG, hidden cards and complete tournament regressions |
| `tests/run_browser_tests.mjs` | Strict isolated compile and 64-test browser suite |
| `tests/ui_browser_smoke.mjs` | Static browser and selected ONNX inference check through Chrome CDP |
| `tests/ui_build_isolation.mjs` | Development refresh regression during production builds |
| `tests/test_onnx_export.py` | Small CPU export parity and manifest regressions |
| `tests/` | Focused rules, solver and neural regressions |
| `legacy/` | Isolated former implementation; see its README |

## Validation evidence

The final current-objective Python suite passes 93 tests in 6.660 seconds.
The state-v3 browser suite passes 64 tests in approximately 0.12 seconds:
51 parity tests and 13 browser table/controller tests. Fixtures cover 37
complete hands, nine HU transitions, and a four-hand progressing-blind sequence,
including current-BB/history/P&L conversions and dry-side-pot regressions.
UI typecheck, lint, static production build, Python compileall, and
`git diff --check` also pass. See [the current cEV validation report](hand-cev.md).

| Command | Recorded result |
| --- | --- |
| `.venv/bin/python -m unittest discover -s tests -p 'test_*.py'` | Passed: 93 tests, 6.660 seconds |
| `node tests/run_browser_tests.mjs` | Passed: 64 tests, approximately 122 ms |
| `cd ui` then `./node_modules/.bin/tsc --noEmit` | Passed: current-version UI |
| `cd ui` then `npm run lint` and `npm run build` | Passed: static export generated |
| Python compileall and `git diff --check` | Passed |
| `node tests/ui_build_isolation.mjs` with development on port 3100 | Earlier static-runtime check passed 25 refreshes during a production build |

[Current hand diagnostics](hand-diagnostics.json) retain all 1,024 attempts
across external and outcome sampling, 3-max/HU, and 25 BB/5 BB starting states.
All completed; none were censored. Eight value comparisons and 39 root-regret
comparisons were within six paired standard errors. This is a broad numerical
smoke criterion, not proof of convergence or small approximation error.
The report retains raw rows, sampling weights, target magnitudes, and empirical
uncertainty rather than selecting only favorable estimates. No standalone
current-version neural smoke run or substantial training was launched; the
Python tests exercise bounded neural fits.

The earlier state-v2 real Chrome check loaded the static export with no Python listener on port
8765, imported a temporary untrained ONNX validation fixture, and performed ten
interactions. Desktop 1440 px and mobile 390/320 px viewports had no horizontal
overflow, browser errors, or API requests. Observed application fetches were
legacy `avg_policy.json.gz` analysis data and locally served WASM assets.
This verifies browser functionality and inference plumbing; untrained fixture
probabilities provide no strategic performance evidence.

`tests/ui_browser_smoke.mjs` provides the repeatable Chrome CDP check. Start
Chrome remote debugging and the static UI separately, then set
`POKER_CDP_ORIGIN`, `POKER_UI_ORIGIN`, `POKER_ONNX_MODEL`, and
`POKER_ONNX_MANIFEST` to those exact endpoints and artifact paths.
`POKER_SCREENSHOT_DIR` optionally captures screenshots. The checked browser
path requires no Python process; offline export and fixture generation are
separate operations. No large training run was launched.

A direct `node --experimental-strip-types` test invocation failed because
Node did not resolve the browser modules' extensionless imports. The supported
`node tests/run_browser_tests.mjs` command compiles the tests strictly into a
temporary CommonJS tree, runs them successfully, and removes generated output.

The browser runner uses the existing TypeScript compiler and reads canonical
fixtures from the repository working directory. The fixture generator is an
importable module and does not add a Python CLI. Python fixtures are used
offline in tests, never fetched or executed by the browser. No prior
Python-service/proxy validation is treated as evidence for the standalone
runtime.

Tests cover betting and reopen rights, insufficient calls, side pots/ties,
chip/card invariants, positions/HU/tournament lifecycle, action sizing,
observations, recursive deeper traverser updates, exact controlled values,
averaging, network shapes/masks, memory persistence, frozen worker generation,
and explicit invalid-state/error handling. A deterministic test compares seeded
single-worker and two-worker generated results directly for both initial
uniform and trained frozen snapshots. This establishes the
sample-generation contract, not large-scale statistical performance.

`docs/smoke-results.json` is historical: it retains an earlier state-v2 tiny
CPU fit and conditional river best-response calculation. No standalone
current-version neural smoke rerun has been launched. Its best-response gain is a finite conditional-subgame diagnostic,
not full-game exploitability, NashConv, or tournament equilibrium evidence.
`docs/hand-diagnostics.json` records raw attempts and aggregate statistics for
the current hand-only traversal contract. The earlier
`docs/outcome-diagnostics.json` remains a retired tournament-winner baseline.

## Remaining limitations

- Solvers use a finite bet abstraction; the hand engine's arbitrary legal
  sizing support does not make the solver continuous-action.
- The environment's final preset level is held, so the tournament has no
  enforced finite horizon. Current learning terminates at the current hand
  rather than following later tournament hands.
- Both full-support averaging and outcome-sampling importance corrections
  exhibit substantial variance; bounded diagnostic weights have extreme tails.
- Historical tournament-winner outcome targets overflowed float32 MSE; that
  objective is retired. The current hand pipeline still checks numerical errors
  explicitly and does not clip targets.
- Three-player general-sum strategic settings do not inherit a two-player
  zero-sum Nash-convergence guarantee from CFR terminology.
- Neural held-out errors on the tiny replay do not demonstrate useful
  approximation accuracy at 25 BB; state-level confidence is uncalibrated.
- Exact best response is limited to supplied finite hand subgames and weighted
  deals. Full-game exploitability/NashConv is not implemented.
- Full history, cloning, replay copies, model construction, and process startup
  are current performance costs. No substantial GPU optimization is attempted.
- Blind progression uses a configurable hand-count simulation preset, not an
  official real-time Expresso schedule.
- No substantial trained per-hand average policy is supplied or automatically
  loaded. Scripted opponents and tiny smoke artifacts are not GTO evidence.

## Cost preflight before additional training

No large job is approved or launched. A future proposal must report model
architecture and parameter count, workers, samples per outer iteration, replay
counts and byte budgets, RAM including transient copies, CPU/GPU use, measured
runtime estimates, exact function/command, and smoke validation before asking
for approval.

For cost illustration only, a bounded controlled-subgame proposal could use
`DeepCFRSolver((0, 1), "hand_chip_delta", seed=2,
advantage_capacity=1000, strategy_capacity=1000)` and call
`run_iteration(controlled_river, traversals_per_player=4, workers=1,
epochs=1, batch_size=32, max_nodes=1000)` ten times. This is not a full-tournament
run and has not been executed as part of this estimate. It would use 49,395
parameters, one traversal worker, no GPU, and eight seeded traversal tasks per
outer iteration, each doing regret and averaging passes. Sample counts depend
on branches and must be measured; the earlier four-task tiny iterations
produced 9–10 total samples. Count scaling is only a rough planning proxy.

Default retained-object limits are 64 MiB per reservoir and 64 MiB combined
generated samples per task. The two-player illustration has three reservoirs:
192 MiB maximum retained replay and up to 384 MiB across old and staged copies.
The retained collection cap adds up to 256 MiB, with up to
`workers * 64 MiB` pending generated data beyond that accounting. Only up to
`workers` jobs are submitted at once, rather than all eight tasks being queued.
Torch baseline RSS, serialization, model/optimizer state, and activations are
additional; these caps
are not a total-process RAM guarantee. The default three-player configuration
has four reservoirs (256 MiB retained, up to 512 MiB across replay copies).

The historical tiny fit is not a cost estimate for the current state/feature
contract. Interpreter startup, legal-tree branching, target variance, and worker
overhead require current measurements. A representative full 25 BB per-hand training runtime must be measured for
the current contract; historical tournament probes do not predict that cost. Approval must follow a concrete measured
proposal, not an extrapolation from the tiny river benchmark.
