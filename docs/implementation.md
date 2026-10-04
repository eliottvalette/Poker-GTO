# Expresso implementation and validation

## Readiness and scope

**SMOKE-TRAIN READY**, not full-tournament training ready. The authoritative
engine and CFR-derived sample generation are usable for bounded validation.
The recorded two-iteration neural smoke trained a tiny fixed-deal heads-up
river subgame with the explicit `hand_chip_delta` objective. This does not
validate convergence on a 3-player, 25 BB tournament.

The full 25 BB tournament probe exceeded its 100-node budget at node 101,
depth 96, in approximately 0.33 seconds. The exception is part of the evidence:
no arbitrary depth cutoff is assigned a terminal utility. Fixed blinds permit
an indefinitely repeated fold cycle: after three rotating hands the initial
stacks can recur. The game therefore has no established finite traversal
horizon. Terminal recursion and full-support averaging also create substantial
branching. Resolving this without changing the objective or biasing targets is
required before a credible full-run cost estimate or launch.

## Authoritative contracts

`TournamentState` owns stable player IDs, stacks, active players, rotating
button, fixed `BlindLevel`, hand number, completed hands, and winner. The default
chips total 75 BB. A live `HandState` owns cards, street contributions, total
contributions, pot, current actor, raise rights, pending actions, settlement,
and canonical public events. During a hand, stacks behind plus pot conserve
chips; after settlement the pot is zero. Player identity is distinct from the
current BTN/SB/BB role. Heads-up assigns the button to SB, acting first preflop;
BB acts first postflop. At the 3-player-to-HU transition, the next BB is
advanced before assigning the new button/SB, preventing the same player from
posting BB twice consecutively when the other seat busts (TDA rule 36C).

The engine accepts explicit legal raise-to amounts, tracks the last full raise,
and distinguishes a short all-in from a full raise for reopening action.
Settlement handles contribution layers, folds, ties, and side pots. Invalid
states and amounts raise errors containing the relevant contract values.

`actions.py` is the only solver sizing authority. Preflop raises use
2.0/2.5/3.0/4.0 times BB for an unopened pot or the current highest contribution
for a re-raise. Postflop targets are `highest + fraction * (pot + to_call)`;
the increment uses the pot after matching the outstanding bet. Targets below
the minimum legal raise are raised to that minimum, equivalent targets are
deduplicated, and stack-capped targets are represented once as `ALL_IN`.
A short all-in call is represented by `CALL`. Masked actions are never trained
or sampled as legal branches.

`Observation` version 2 contains exact hero cards and board slots, an explicit
unknown-card sentinel, street, continuous numeric features normalized by 25 BB,
legal mask, and full history tokens. Numeric fields include stacks,
contributions, effective stacks, pot, amount to call, raise context, player
count, position, button, and canonical action targets. No hand169 bucket or
packed-u64 key is used. The tabular key serializes the complete observation.
Tournament recall includes prior hands' public events, initial and settled stack
records, the hero's previously observed private cards, and opponents' cards
actually exposed at showdown. Initial/final STACK tokens and exposed card tokens
retain this information in the neural history. Unexposed opponent private cards
and the future deck are excluded from observations. Winning by folds does not
expose the winner's cards.

## Reference traversal and averaging

`ExternalSamplingMCCFR` freezes the current regret-matched strategy for each
iteration. Every traverser decision recursively evaluates every legal action,
including later traverser decisions in each continuation. Opponents are sampled
from their current strategy. Chance deals are sampled with injected RNG state.
The sink receives `action_value - strategy_weighted_node_value`; opponent
sampling probabilities do not multiply these regret targets. Regret matching
uses positive cumulative regrets and a defined uniform distribution when all
legal positive regrets are zero. Cumulative regrets are not clipped.

Average-policy collection uses a separate traversal. It enumerates the
averaging player's own supported actions and samples each opponent uniformly
from all legal actions. A strategy sample has weight
`own_strategy_reach / sampled_opponent_prefix_probability`. Chance is sampled
from its true distribution. This estimates an own-reach-weighted average
without inadvertently weighting it by the opponents' strategy reach. The
tabular reference averages iterations uniformly. Deep CFR training additionally
weights samples linearly by iteration. Current regret-matched play and average
policy are separate objects; evaluation and the UI query the latter.

Node/depth budgets raise `TraversalBudgetExceeded`. An unsuccessful iteration
does not publish partial solver updates. The separate averaging pass can have
high variance and large importance weights; its correctness does not establish
practical convergence or tractability for the full tournament.

## Deep CFR and worker ownership

Each player identity has an `AdvantageNetwork`; a separate shared
`AveragePolicyNetwork` learns strategy samples. The modest encoder combines
exact-card embeddings, numeric features, and a full-history GRU, followed by a
64-wide head. Each network has 16,401 parameters. Three advantage networks plus
one average network total 65,604 parameters; float32 parameters alone occupy
approximately 0.25 MiB. Optimizer state, activations, replay, and Python objects
are additional memory.

Advantage targets are masked CFR instantaneous regrets; training uses masked
squared error. The average network produces a masked distribution and uses a
weighted distribution loss. Metrics include training/held-out losses, nodes,
sample counts, traversal values, and average-policy change on replay probes.
Tiny held-out splits are smoke diagnostics, not reliable generalization scores.

`ReservoirMemory` uses uniform Algorithm R sampling with explicit capacity,
schema, objective, seed, iteration, player, mask, target, sampling weight, and
model version. Memory persistence uses schema version 3 and rejects incompatible
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

## UI ownership

`server.py` runs a loopback-only Python service with version-2 serialized table
views and commands. Views include persistent tournament state, public history,
hero cards, cards actually exposed at showdown, and server-computed action
amounts. A hand won by folds does not expose opponents' cards. Hidden
opponent cards and the future deck are not sent. Revision checking rejects stale
commands; candidate transitions are staged before a session is updated.

The Next.js `/api/table` route proxies same-origin JSON commands to the local
engine. `game.ts` contains the live table contract and validation, without a
second poker rules implementation. Thus live engine parity is achieved through
authority rather than independent TypeScript simulation. `TestTable.tsx`
preserves the original visual components and layout while rendering persistent
tournaments and canonical actions; `page.tsx` defaults to Test Live. Overview
and Cas précis retain their original analysis views and packed-policy helpers
in `infoset.ts` and `policy.ts`, explicitly marked as legacy 50 BB single-hand
analysis. Those helpers and data never determine Test Live transitions or
probabilities; no new neural policy is forced into a 169-hand heatmap.

## Important source changes

| File | Responsibility |
| --- | --- |
| `poker_game_expresso.py` | HandState, HandPlayer, ActionEvent, authoritative betting/order/side pots and invariants |
| `classes.py` | Validated Card/Deck primitives for isolated push/fold tools |
| `tournament.py` | Persistent tournament lifecycle, elimination, HU, button rotation and winner utility |
| `actions.py` | Canonical 13-action IDs, legal sizes and deduplication |
| `infoset.py` | Structured exact observations and full tournament recall |
| `cfr_solver.py` | Recursive traversal, regret matching and tabular average-policy oracle |
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
| `server.py` | Authoritative local table service and average-policy queries |
| `ui/src/app/api/table/route.ts` | Same-origin proxy to the loopback engine |
| `ui/scripts/run-app.mjs` | Starts the existing repository Python environment and Next.js together; stops both on exit |
| `ui/src/lib/game.ts` | Live serialized contract and response validation |
| `ui/src/lib/{infoset,policy}.ts` | Retained legacy-only secondary analysis helpers |
| `ui/src/components/TestTable.tsx` | Persistent tournament interaction |
| `ui/src/app/page.tsx` | Test Live default and secondary analysis surfaces |
| `tests/` | Focused rules, solver, neural and service regressions |
| `legacy/` | Isolated former implementation; see its README |

## Validation evidence

The final Python verification passed 39 tests in 2.493 seconds. TypeScript
typecheck and lint also passed after integrating the original-design live table.

| Command | Recorded result |
| --- | --- |
| `python3 -m unittest discover -s tests -v` | Passed: 39 tests, 2.493 seconds |
| `python3 -m scripts.smoke_validation` | Two tiny CPU iterations completed; full-root probe failed explicitly at its budget |
| `cd ui` then `./node_modules/.bin/tsc --noEmit` | Passed |
| `cd ui` then `npm run lint` | Passed |
| `cd ui` then `npm run build` | Passed: final live-table integration compiled and routes generated |
| `node tests/ui_proxy_smoke.mjs` with local services running | Passed: 8 commands, 8 hands, HU transition, player 0 wins, 75 BB conserved; cross-origin request rejected with 403; session closed |

A local Chrome rendering check caught and corrected an attempted numeric
format of unavailable in-progress hand results. Pending results now display
`En cours`. The corrected production page rendered Test Live with the existing
sidebar, poker table, cards, results/P&L panel, and history. This desktop check
does not establish exhaustive responsive-layout or browser coverage.

The integrated `npm run dev -- --hostname 127.0.0.1 --port 3100` startup was
also verified with the proxy smoke test: no separately started Python service
was required. The launcher uses `.venv/bin/python` explicitly because a Node
subprocess resolved a different Python installation without Treys. An explicit
`POKER_PYTHON` override is available; there is no dependency installation or
interpreter fallback. The engine binds successfully before announcing
readiness, and startup failures stop the application.

Tests cover betting and reopen rights, insufficient calls, side pots/ties,
chip/card invariants, positions/HU/tournament lifecycle, action sizing,
observations, recursive deeper traverser updates, exact controlled values,
averaging, network shapes/masks, memory persistence, frozen worker generation,
and service state/error handling. A deterministic test compares seeded
single-worker and two-worker generated results directly for both initial
uniform and trained frozen snapshots. This establishes the
sample-generation contract, not large-scale statistical performance.

`docs/smoke-results.json` records approximately 0.48 seconds for two tiny
iterations, peak process RSS of 286 MiB, and average serialized sample size
3,789 bytes. JSON bytes are not Python retained-memory bytes. Conditional
river best-response gain was approximately 0.87 BB on the supplied fixed deal.
This is not full-game exploitability, NashConv, or a tournament equilibrium
measurement. The tiny tournament traversal used 7 nodes; it does not establish
full-root tractability.

## Remaining limitations

- Solvers use a finite bet abstraction; the hand engine's arbitrary legal
  sizing support does not make the solver continuous-action.
- Fixed-blind tournaments admit nonterminating paths and unbounded recall.
  Full-root traversal cost and unbiased handling of this issue remain open.
- The full-support average pass may have prohibitive importance variance.
- Three-player general-sum strategic settings do not inherit a two-player
  zero-sum Nash-convergence guarantee from CFR terminology.
- Neural held-out errors on the tiny replay do not demonstrate useful
  approximation accuracy at 25 BB; state-level confidence is uncalibrated.
- Exact best response is limited to supplied finite hand subgames and weighted
  deals. Full-tournament exploitability/NashConv is not implemented.
- Full history, cloning, replay copies, model construction, and process startup
  are current performance costs. No substantial GPU optimization is attempted.
- Blind levels are represented explicitly but progression is not implemented.
- No substantial trained tournament average policy is supplied or automatically
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
run and has not been executed as part of this estimate. It would use 49,203
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

The recorded 0.48-second tiny run suggests seconds of actual training for a
similarly restricted sequential illustration, but this is not a reliable upper
bound; interpreter startup, budget failures, history growth, and worker overhead
need measurement. A representative full 25 BB runtime remains unknown because
the bounded probe did not finish. Approval must follow a concrete measured
proposal, not an extrapolation from the tiny river benchmark.
