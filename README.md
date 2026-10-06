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

The importable runner trains separate 3-max and HU policies, with compact replay,
observable deterministic features, suit normalization, controlled tournament/
synthetic/stratified roots, evaluation and atomic resumable checkpoints.

```sh
python -m scripts.train --config configs/deep_cfr_pilot.json --preflight
```

After reviewing measured costs and explicitly approving the substantial pilot:

```sh
python -m scripts.train --config configs/deep_cfr_pilot.json
```

Use `--resume <checkpoint>` to resume and `--iterations N` for additional
iterations. `--export-onnx` exports separate average-policy artifacts. Core logic
lives in `training.runner.TrainingRunner`, independently of the CLI. No online
resolver, range model or UI work is required to train. Fresh network fitting each
iteration is preserved.

See [the complete training contract and migration report](docs/deep-cfr-training.md),
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
Scripted opponents are smoke-test opponents. Overview and Cas précis retain
the original analysis surfaces, explicitly labeled as the former 50 BB
single-hand policy; that data does not drive tournament play.

Policy information is unavailable until a compatible average-policy `.onnx`
file and its `.json` manifest are explicitly selected. The browser validates the
full schema, objective, player-count coverage, model SHA256, current-hand scope, explicit amount/utility units, legal mask, and
output distribution. ONNX inference uses locally served CPU WebAssembly assets.
Loaded probabilities are experimental with uncalibrated confidence. Old saved
policies and `ml/trained_policy_model.pth` are never used for current play.

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

Raw state schema 3 remains unchanged. Feature schema 1, replay version 5,
average artifact version 4, runner checkpoint version 1 and ONNX manifest version
3 explicitly reject retired neural contracts. Browser loading computes the same
observable features/suit normalization and retains separate models for 3-max/HU.
The preceding baseline reports remain historical evidence; current validation
passes 113 Python and 77 browser tests. No substantial training has been launched.
