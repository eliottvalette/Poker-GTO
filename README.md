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

## Local Test Live

Use the existing Python and Node environments. The Python environment needs
PyTorch and Treys; the UI dependencies are declared in `ui/package.json`.
No imports launch training or install dependencies.

Start the application with one command:

```sh
cd ui
npm run dev
```

This command starts the Python engine and Next.js together, waits until the
engine is listening, and stops both on exit. `npm run start` does the same for
the production build. The Python executable is the repository's `.venv/bin/python`; set
`POKER_PYTHON` explicitly to use a different existing environment. No packages
are installed automatically. Startup errors are displayed in the terminal.

Open the local Next.js URL. The application opens on **Test Live**. The browser
sends canonical action IDs through the same-origin Next.js proxy to Python;
Test Live uses Python for all poker transitions and preserves the existing
visual components and layout. Test Live displays
persistent stacks, positions, hand number, elimination, tournament result, and
server-computed legal bet targets. Scripted opponents are smoke-test opponents.

Without a compatible average-policy checkpoint the UI explicitly reports that
policy information is unavailable. Overview and Cas précis retain the original
analysis surfaces, explicitly labeled as the former 50 BB single-hand policy;
that legacy data does not drive tournament play. To load an explicitly selected new-format
checkpoint, set `POKER_AVERAGE_POLICY` before starting the Python service.
Invalid checkpoints fail to load. Loaded neural probabilities are labeled
experimental with uncalibrated confidence. The old saved policy and
`ml/trained_policy_model.pth` are never selected automatically.

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
                    Python policy query -> Test Live
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
python3 -m unittest discover -s tests -v
python3 -m scripts.smoke_validation
```

From `ui/`:

```sh
./node_modules/.bin/tsc --noEmit
npm run lint
npm run build
```

With the Python service and Next.js server running, from the repository root:

```sh
node tests/ui_proxy_smoke.mjs
```

[Recorded smoke results](docs/smoke-results.json) include a two-iteration CPU
Deep CFR validation on a controlled river subgame and a bounded full-tournament
probe. The probe exhausted its traversal budget and raised an error rather than
inventing a terminal value. Full tournament training requires further work on
unbounded fold cycles and traversal growth, plus a cost preflight and approval.
