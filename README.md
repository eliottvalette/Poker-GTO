# Poker-GTO

An experimental Expresso training environment with a Python NLHE engine,
persistent 3-player tournaments, a tabular external-sampling MCCFR reference,
and a small Deep CFR pipeline. **Training readiness: SMOKE-TRAIN READY.**
The current implementation has not solved the full 25 BB tournament and does
not establish a GTO policy.

All chip amounts are actual big blinds: SB = 0.5, BB = 1.0, and initial stacks
are `(25.0, 25.0, 25.0)`. Chips persist between hands, positions rotate,
eliminated players leave the hand engine, and play transitions to heads-up.
The tournament objective is winning the tournament; hand chip-delta utility
is a separately declared objective for controlled validation subgames.

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
full schema, objective, player-count coverage, model SHA256, legal mask, and
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
                         recursive external sampling
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
full public betting history, and the player's recall across tournament hands.

See [the migration audit](docs/audit.md),
[implementation and validation details](docs/implementation.md), and
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

[Recorded smoke results](docs/smoke-results.json) include a two-iteration CPU
Deep CFR validation on a controlled river subgame and a bounded full-tournament
probe. The probe exhausted its traversal budget and raised an error rather than
inventing a terminal value. Full tournament training requires further work on
unbounded fold cycles and traversal growth, plus a cost preflight and approval.
