# Poker-GTO

An experimental Expresso training environment with a Python NLHE engine,
persistent 3-player tournaments, a tabular external-sampling MCCFR reference,
an explicitly selected outcome-sampling traversal, and a resumable two-track Deep CFR pipeline. **Training readiness: PILOT READY**, pending approval for substantial training.
The learning objective is conditional **per-hand chip EV**, not tournament
winner utility or ICM. No trained policy is claimed to be mathematically GTO.

Initial SB = 0.5, BB = 1.0, and stacks are `(25.0, 25.0, 25.0)` initial-BB
chips. The physical chip total stays 75 throughout a tournament. Blinds can
grow between hands; the UI displays stacks, pot, bets, and actions in the
**current** BB, so the displayed total is `75 / current_big_blind`.
The explicit default simulation preset doubles blinds every ten hands, through
16/32 at hand 51, then holds that level. This is not an official Expresso
schedule. Stacks persist, positions rotate, players bust, and live play moves
to heads-up until one winner. The solver evaluates only the current hand's
terminal chip delta in conserved initial-BB chip units. Tournament winner
payout is an environment result, not its training utility.

## Offline training

For ordinary use, run `python train-3.py` or `python train-hu.py`. Each resumes its
checkpoint automatically, trains for one hour (editable in `training/settings.py`)
and saves complete state. `python migrate.py` offers an ASCII menu to activate
checkpoint reference policies, or independently export and activate the 3-max
or HU ONNX model for Test Live.
See [the timed-script workflow](docs/training-scripts.md) for paths, safe stopping
and publication. There are no argument-parser flags.


The importable runner trains separate 3-max and HU policies, each with one shared advantage network
and one average-policy network, with compact replay,
observable deterministic features, suit normalization, controlled tournament/
synthetic/stratified roots, evaluation and atomic resumable checkpoints.

Use the importable Python API from a script or notebook:

```python
from training.config import load_config
from training.runner import TrainingRunner

config = load_config("configs/deep_cfr_pilot.json")
runner = TrainingRunner(config)
runner.run()  # Substantial compute: run only after reviewing and approving costs.
runner.export(onnx=True)
```

For resume, call `TrainingRunner.load_checkpoint(explicit_checkpoint_path, config)`;
`runner.run(iterations=N)` requests additional iterations. The core is configurable
and has no argument parser. No online resolver, range model or UI work is required
to train. Fresh network fitting each iteration is preserved.

See [the shared-advantage migration](docs/shared-advantage.md),
[the complete training contract and migration report](docs/deep-cfr-training.md),
[measured preflight data](docs/deep-cfr-preflight.json), and
[the preflight summary](docs/deep-cfr-preflight.md), and
[production-sized CPU profiling](docs/deep-cfr-profile.md). Only small measured iterations
have run; no substantial pilot or full run has been launched.

## Browser Test Live

The UI runs independently of Python and preserves the existing visual design.
For development:

```sh
cd ui
npm run dev
```

For a static production build:

```sh
cd ui
npm run build
npm start
```

The build produces `ui/out/`; `npm start` serves these static files with Node.
There is no API route or Python subprocess. The exported directory can also be
served by another static host. Development output uses `.next-dev`, separately
from production `.next` output.

Test Live opens by default and runs the parity-tested TypeScript tournament
engine entirely in the browser. Stacks persist, positions rotate, eliminated
players leave, play transitions to heads-up, and chip totals remain conserved.
Scripted opponents are smoke-test opponents. All three views automatically load
catalog-selected ONNX exports from `/policy/index.json`, routing by seated player
count. The catalog is checked every 15 seconds and when the window regains focus;
a newly published HU or 3-max export is loaded without a button or file selection.
Missing tracks and loading failures are explicit.

Overview queries all board-compatible exact hole-card combos and averages their
action mixtures uniformly into a 169-hand matrix. Cas précis queries a selected
exact hand in the same public state. Both support 3-max/HU, positions, stacks,
blinds, streets, explicit boards and legal betting continuations with full public
history. They do not read the old packed policy or historical visit counts.

The browser validates schema, objective, model SHA256, player-count coverage,
current-hand scope, units, legal masks and output probabilities. ONNX inference
uses local CPU WebAssembly assets. A loaded policy is a neural approximation;
its equilibrium quality has not been certified. Legacy policies and
`ml/trained_policy_model.pth` are unused by these views.

Python is needed only for offline training, canonical fixture generation, and
model export. The importable exporter
`ml.export_onnx.export_average_policy(checkpoint, model_path, manifest_path,
example_observation)` produces the paired browser artifacts and verifies CPU
inference against PyTorch with two complete history lengths. It needs existing
PyTorch, Treys, ONNX, and ONNX Runtime environments; it does not install them or
launch training automatically.

## Architecture

```text
TournamentState -> HandState -> canonical actions and Observation
                                  |
              external-sampling reference / outcome-sampling traversal
                                  |
                 advantage memories + strategy memory
                                  |
             frozen worker snapshots -> central CPU trainer
                                  |
               advantage networks + average-policy network
                                  |
             offline ONNX export + manifest -> browser policy query
             Python fixtures -> parity-tested browser tournament
```

The hand engine accepts arbitrary legal raise-to amounts. Solvers use a masked,
deduplicated 13-action abstraction with preflop multiples and postflop pot
fractions. Observations retain exact hero/board cards, numeric BB features,
full current-hand public betting history, hand number, and blind-level context.
Earlier hands are excluded from model input; `observe(tournament)` observes its
current hand with the same conditional cEV contract.

See [the migration audit](docs/audit.md),
[implementation and validation details](docs/implementation.md),
[the outcome-sampling estimator contract](docs/outcome-sampling.md), and
[legacy isolation](legacy/README.md). The old packed infosets, rollout solver,
policy distillation, and stale-regret merge implementation are archival only.

## Validation

From the repository root:

```sh
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
python3 -m scripts.smoke_validation
```

From `ui/`:

```sh
./node_modules/.bin/tsc --noEmit
npm run lint
npm run build
```

With a development server running at `http://127.0.0.1:3100`, the following
regression check builds production while repeatedly requesting the development
page, scripts, and build manifest. Set `POKER_UI_ORIGIN` for another local port.

```sh
node tests/ui_build_isolation.mjs
```

Cross-language parity and browser-controller tests use the existing TypeScript
installation and Node's built-in test runner, with temporary compilation cleaned
up automatically:

```sh
node tests/run_browser_tests.mjs
```

`tests/ui_browser_smoke.mjs` also checks the static app in a separately started
Chrome with remote debugging. Set `POKER_UI_ORIGIN`, `POKER_CDP_ORIGIN`,
`POKER_ONNX_MODEL`, and `POKER_ONNX_MANIFEST` explicitly; the last two identify
an exported model/manifest pair. `POKER_SCREENSHOT_DIR` is optional. This browser
check imports a model for inference and does not run training.

[The current cEV validation report](docs/hand-cev.md) and
[hand diagnostics](docs/hand-diagnostics.json) retain 1,024 completed
bounded traversal attempts without censoring; eight value and 39 root-regret
comparisons pass the documented numerical smoke criterion. Those baseline suites passed 93 Python and 64 browser tests. [Earlier neural smoke results](docs/smoke-results.json)
are historical state-v2 evidence; current neural fits are exercised by bounded
tests rather than a new standalone training run. External sampling
remains the all-action reference; outcome sampling is explicitly selectable as
`traversal_mode="outcome_sampling"`. Neither mode follows future tournament
hands for learning. Missing or incompatible checkpoints fail explicitly.

Raw state schema 3 remains unchanged. Feature schema 2, replay version 6,
average artifact version 5, runner checkpoint version 2 and ONNX manifest version
4 explicitly reject retired neural contracts. Browser loading computes the same
observable features/suit normalization and retains separate models for 3-max/HU.
The preceding baseline reports remain historical evidence; current validation
passes 120 Python and 78 browser tests. No substantial training has been launched.
