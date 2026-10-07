# Offline Deep CFR training

## Readiness and scope

**PILOT READY**, pending explicit approval to execute the substantial pilot.
The migration implements phases A–F, with shared advantage approximation in each player-count track. Only bounded tests and two small measured
iterations have run. It does not certify convergence or a credible playing policy.
The original measured record is [deep-cfr-preflight.json](deep-cfr-preflight.json);
shared-model measurements are stored separately in [shared-advantage-preflight.json](shared-advantage-preflight.json).
The shared-model migration starts from `9fda3f2` on main; these changes
are uncommitted. Checkpoints record the base commit, dirty flag and source-code
fingerprint as well as the configuration hash.

The objective is permanently `hand_chip_delta` for this pipeline:
`u_i = final_stack_i - pre_blind_hand_start_stack_i`, in conserved physical chips
(initial BB = one chip). Settled utilities sum to zero. A traversal ends at the
current hand's settlement, or at an already exact folded-player payoff.
It never starts another hand. Persistent tournaments generate subsequent roots.
This is **conditional per-hand cEV**, not complete tournament equilibrium.

## Current offline architecture

```text
Persistent tournament rollouts + synthetic simplex strata + coverage grid
                                  ↓
                  HandState with canonical local seats
                                  ↓
     raw observable state + deterministic features + suit normalization
                                  ↓
             frozen external-sampling Deep CFR workers
                                  ↓
             advantage replay       strategy replay
                                  ↓
           shared advantage model   average-policy model
                                  ↓
              complete two-track atomic checkpoint
                                  ↓
       average_3max.pt / average_hu.pt → ONNX + manifests
```

External sampling remains the primary generator. Every future traverser decision
in an explored branch produces an advantage target; opponents are sampled.
The existing independent average pass remains reach/importance weighted.
Outcome sampling remains implemented, tested and explicitly selectable. It is
not tuned or enabled in the supplied configurations.

The modest architecture remains card embeddings (8), numeric MLP (32), history
GRU (32), and head (64). Each model has 17,329 parameters after adding 27 derived
numeric inputs. The two tracks have two advantage models and two average
models: one of each per player-count track. Models are **freshly initialized and
fitted from their reservoirs each outer iteration**, preserving the baseline semantics. Adam state exists only
inside one fit and therefore is not a persistent checkpoint requirement.
The method's research reference is [Deep CFR, Brown et al., 2019](https://proceedings.mlr.press/v97/brown19b.html).

## State and feature ownership

`infoset.Observation` remains state schema 3 and preserves exact raw cards,
all numeric state, legal targets and masks, complete public current-hand history,
and lossless JSON recall for tabular keys and diagnostics.

`features.neural.NeuralObservation` excludes recall. It stores all consumed
numeric/history values as packed little-endian float64 bytes, with exact cards,
street and legal mask alongside them. Float32 conversion occurs at the model
boundary. All public history tokens remain; no sequence truncation or manual
history buckets are introduced. Replay uses this compact state, not redundant
JSON strings. Conservative replay accounting counts scalar occurrences
independently and is stable across serialization and reservoir restoration.

Raw numeric amounts retain the normalization `physical_amount / (25 * current_BB)`;
hand number is divided by 25. The physical BB and unbucketed level remain inputs.
Derived inputs add:

- Capped call-payment pot odds, call/pot and hero stack/pot.
- Separate opponent effective-stack SPR values and legal raise-to/pot values.
- Available-card made category, immediate straight/flush completion counts,
  flush-draw indicator, overcards and public board rank/suit/pair texture.
- Ace-in-board-suit blocker count, without claiming a universal nut blocker.

Straight/flush counts are exactly defined one-card completion counts among
currently unseen cards, not winning outs or action EV. No draw counts remain on
the river. No hidden opponent cards or future deck cards are used.

Global suits are normalized in first observable occurrence order: hero cards,
then current board. The same mapping is applied to card history events. This
preserves rank/suit relationships under every global suit permutation while
leaving betting semantics unchanged. Raw diagnostic cards remain untouched.

Feature schema 2 also orders numeric opponents and history actors as Hero then
clockwise by public positions, and sorts the simultaneous initial STACK tokens
into that order. Card and betting chronology are preserved. Absolute player IDs
remain diagnostic sample metadata and do not select a neural model. Renaming
seats or rotating the ring starting point yields identical consumed tensors.
Python and browser neural inputs are parity-tested.

The equity engine accepts explicit hands or weighted ranges, including two
opponents. Blocked combinations are removed, ranges normalized, and independent
range priors conditioned on mutually compatible hands. Enumeration is used when
the explicit work bound fits; otherwise seeded Monte Carlo is used. Results
always identify exact/sampled status, evaluation count, sampling count and
estimated standard error. Impossible ranges and exceeded rejection budgets fail
explicitly. Showdown equity is distinct from action EV. This infrastructure is
not called with simulated opponents' actual hidden cards during training and is
not required by the initial policy.

## Controlled roots and tracks

Root sampler version 1 uses configurable probabilities, initially 50% on-policy,
25% synthetic and 25% stratified. Every active initial stack is positive and
physical stacks total 75 in both tracks.

On-policy rollouts use the preceding iteration's appropriate average policy;
iteration zero uses uniform legal play. The sampler retains tournament state,
positions, eliminations and blind progression. HU on-policy roots in the supplied
configuration are captured after actual 3-max eliminations. Survivor IDs are
remapped to local 0/1. A HU-only configuration may explicitly start HU tournaments
by setting `tournament_start_players=2`; requesting survivor roots without a
3-max policy fails. Rollout hand and decision budgets are explicit.

Synthetic exploration mixes balanced perturbations, dominant stacks, two short
stacks, near-elimination weights and gamma/simplex shapes, then permutes their
seat assignments. It does not uniformly sample arbitrary stack triples.
The stratified stream cycles all six blind levels, balanced/asymmetric shapes,
and buttons in a Cartesian grid (36 roots for 3-max, 24 for HU). It covers deep,
medium and shallow effective stacks. The guarantee applies to completion of
that stream's cycle, not a tiny probabilistic mixture prefix.

Terminal roots caused by blind posting are counted and retained as legitimate
zero-decision traversal draws, never retried to censor them. Undefined decision
features are absent for these roots; source/player/blind/terminal counts remain.
Coverage otherwise records source, player count, street, position, blind level,
effective stack bins, stack ratios, stacks, pot, SPR, call/pot, masks, targets
and full history length. Sample coverage is aggregated in workers and merged
centrally. Each evaluation interval persists the cumulative coverage report.

3-max and HU have independent replay, one shared advantage network each,
one average network each, and separate metrics. Advantage samples from all
traversers of a track enter one Algorithm R reservoir and one fresh fit. Position
remains a feature, not a model identity. The pilot preserves total replay size:
30,000 advantage samples for 3-max, 20,000 for HU, and 10,000 strategy samples per
track (70,000 total). Advantage byte budgets aggregate the former seat budgets:
384 MiB and 256 MiB respectively; strategy budgets stay at 128 MiB per track.
Config version 2 records the advantage capacity/budget inside each track. A single atomic runner checkpoint contains both tracks and their
independent sampler states, preventing publication of one completed track when
the other fails. Python and Test Live route average policies by the number of
seated players in the current hand; folding during a 3-max hand does not switch
to the HU model. A missing route gives an explicit unavailable result.

## Usage and outputs

Use the importable Python API from a script or notebook in the existing environment.
For a small cost preflight:

```python
from training.config import load_config
from training.preflight import run_preflight

config = load_config("configs/deep_cfr_pilot.json")
report = run_preflight(config)
```

This runs two measured iterations at four traversals per player, including the
uniform and learned-policy phases, fixed evaluation, checkpointing and ONNX
parity. It writes `runs/deep_cfr_shared_pilot/preflight.json` and discards temporary
measurement weights. It never launches the configured pilot.

After reviewing costs and explicitly approving the substantial run:

```python
from training.runner import TrainingRunner

runner = TrainingRunner(config)
runner.run()
runner.export(onnx=True)
```

For explicit resume:

```python
runner = TrainingRunner.load_checkpoint(
    "runs/deep_cfr_shared_pilot/checkpoints/iteration_000005.pt", config
)
runner.run()
```

There is no intrinsic final epoch. `runner.run(iterations=50)` requests fifty
additional iterations beyond the loaded state, including beyond the original
budget. The same config hash is required on resume; changed capacities, schemas
or configuration fail rather than silently replacing a run. A new runner's
`run()` refuses an existing metrics target and asks for an explicit checkpoint.
The full configuration is a 500-iteration planning template using the measured
pilot capacities, not a declaration that large training is ready. Larger replay
capacities need a separate measured and approved configuration.

`runner.export(onnx=True)` exports both average policies after successful fitting.
Other programmatic interfaces are `run_iteration()` and `save_checkpoint(path)`.
Missing or incompatible inputs fail explicitly. The former `scripts/train.py`
argument parser was removed; orchestration lives in `training.runner`.

```text
runs/deep_cfr_shared_pilot/
  metrics.jsonl
  preflight.json
  3max/metrics.jsonl
  3max/coverage.json
  hu/metrics.jsonl
  hu/coverage.json
  checkpoints/iteration_000005.pt
  ...
  policy/average_3max.pt
  policy/average_hu.pt
  policy/average_3max.onnx + average_3max.json  # explicit ONNX export
  policy/average_hu.onnx + average_hu.json
```

Checkpoints include iteration, complete config and hash, all schema identifiers,
traversal mode, root version, Python/Torch/solver/reservoir/sampler/tournament RNGs,
models, reservoirs, metrics, coverage, source metadata and tournament state.
Primitive/tensor payloads are checksummed and loaded with `weights_only=True`.
Temporary sibling writes are flushed and atomically replaced. Corruption,
inconsistent counts, incompatible models and configuration mismatches are errors.
Worker/fit/evaluation failures leave both tracks and sampler state unpublished.
JSONL is derived from checkpoint-owned metrics, so a restart reconstructs the
matching complete history. Each file is atomic; multiple diagnostic files are
not a filesystem transaction. A complete checkpoint can repair those logs.

## Evaluation and metrics

Fixed seeded suites cover balanced deep, asymmetric, shallow, flop, turn and
tractable river states for both tracks. Policy drift uses the same six probes
per track throughout a run. Reservoir fit metrics retain train/held-out losses.
Independent fixed-root validation labels are generated with separate seeds under
the preceding frozen snapshot and never inserted into training replay. These
measure conditional sampled regret/policy approximation, not an exact label for
the full historical average at every information state.

Bounded best response uses four weighted hidden river deals sharing public/hero
information and chooses information-set-consistent actions. It reports policy
value, BR value and gain in initial-BB chips. It is a finite conditional test,
not full-game exploitability. Five scripted opponents supply three fixed-hand
sanity results each; they are deliberately secondary to approximation, drift and
bounded BR metrics.

Every successful iteration logs time, root/worker/fit costs, traversals, node
counts and per-task node distribution, depth, generated samples, reservoir
seen/retained/bytes, losses, fixed drift and coverage. Scheduled evaluations and
checkpoint paths are machine-readable. One reported traversal/task consists of
one regret traversal plus the independent average-policy pass.

Workers use one Torch inference thread. The central trainer thread count is
explicit. Workers neither train nor merge local regret tables. GPU is unused.
The preflight reports worker/trainer RSS, total child CPU including spawn,
throughput, sample size, replay projections, checkpoints, ONNX size and runtime
estimates. Small-batch spawning and linear full-reservoir fitting extrapolations
are conservative planning approximations, not runtime guarantees.

## Future live path, deliberately deferred

```text
Current observable poker state
            ↓
Deterministic arithmetic/card/equity analysis
            ↓
Legitimate opponent range beliefs
            ↓
Reference average policy + optional continuation value
            ↓
Budgeted local search / Monte Carlo with uncertainty
            ↓
Final mixed policy
```

Future contracts are separate responsibilities:

- `ValueNetwork`: expected current-hand continuation chip utility conditioned
  on an information state, with solver/search-produced labels. No fabricated
  value labels or current implementation.
- `OpponentActionModel`: `P(action | candidate_opponent_hand, public_state_before_action)`.
  Public state excludes hidden cards. It supplies likelihoods, not posterior
  distributions or poker decisions.
- `RangeTracker`: initialize legal combination weights, remove observable dead
  cards, update `R_next(h) ∝ R(h) * likelihood(action | h, I)`, then normalize
  deterministically. Empty posterior mass is an explicit failure.
- `LocalResolver`: observable state, hero cards, legitimate ranges, candidate
  abstraction, base policy, optional value model and explicit compute budget;
  returns a mixed policy, action-EV estimates, uncertainty and consumed work.

Range entropy, class masses and range equity become separate belief-dependent
features only once legitimate beliefs exist. The base reference policy remains
separate from opponent/population exploit models. No LLM participates in numeric
poker decisions. Range visualization, Bayesian tracking, continuation training
and online resolving await the first credible offline policy.

## Changed modules and retired formats

New modules: `features/{cards,deterministic,equity,range_features,neural}.py`,
`training/{config,root_sampler,metrics,evaluation,checkpoint,runner,preflight}.py`,
`ml/policy_router.py`, both `configs/deep_cfr_*.json`, and
`ui/src/lib/poker/neural.ts`, plus feature/root/runner/browser parity tests and
fixtures.

Updated modules: `actions.py` (version identifier only), `ml/model.py`,
`ml/memory.py`, `ml/deep_cfr.py`, `ml/export_onnx.py`,
`scripts/smoke_validation.py`, `ui/src/lib/onnx-policy.ts`,
`ui/src/components/TestTable.tsx`, existing neural/export tests, browser test
runner, README, baseline-document labels and `.gitignore`.

No engine or reference traversal was replaced. Legacy stays isolated. The per-seat advantage-model/replay layout, raw JSON recall replay and
previous neural ordering are retired. Old checkpoints are rejected rather than
merging arbitrary seat weights or guessing how to convert encoded data.
Current versions: raw state 3, feature 2, action 1, replay 6, average artifact 5,
runner checkpoint 2, ONNX manifest 4. Tabular state remains unchanged.

A discovered float32 softmax rounding error is repaired at the neural probability
transport boundary: mass must already be within 1e-6 of one, then float64
normalization creates a valid distribution. Invalid mass still raises; the
reference solver's stricter validation and regret targets are unchanged.

## Verification and approximations

Shared-model validation passes 120 Python tests and 78 browser tests, including
pooled fitting, seat/ring invariance, resume, worker atomicity, outcome sampling,
PyTorch/ONNX parity and Python/browser feature parity. Python compilation,
TypeScript checking and ESLint also pass. The preceding baseline production build
is recorded separately; an interactive WASM browser session
 is not part
of this preflight; full PyTorch/ONNX/browser-session parity remains a phase-H
verification against the eventual useful policies.

Known approximations remain per-hand cEV, discrete 13-action betting, neural
function approximation, finite/root-mixture coverage, sampled traversal deals,
conditional bounded BR and sampled equity when its exact work bound is exceeded.
Three-player self-play has no universal Nash-convergence claim. Policies remain
separate by player count. Blinds use the existing illustrative simulation preset,
not an official Expresso timetable. There is no learned range/opponent/value or
exploit model yet. Two measured iterations are not evidence of convergence.
