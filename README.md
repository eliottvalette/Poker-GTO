# Poker-GTO

A range-aware poker engine with exact chip accounting, explicit Bayesian beliefs,
local CFR and bounded simulation. Neural policies are optional strategic tools.
General poker strength is not certified.

## Start the corrected training

```sh
.venv/bin/python -u train-hu.py
.venv/bin/python -u train-3.py
```

Each invocation runs a 30-minute session, stopping between complete iterations.
The first starts fresh; later invocations resume the corresponding local run.
The configs use recorded-player stratification, protected opening replay and
bounded optimizer updates, larger frozen CFR batches and independent average-policy fitting.
See `docs/TRAINING_CADENCE.md` for the measured budgets; the earlier 50/20-epoch configuration was rejected. No old published model is an implicit
resume source. See [training and publication](docs/TRAINING.md) for exact budgets,
validation requirements and export steps.

```sh
.venv/bin/python migrate.py
```

Select 2 for three-player UI export, 3 for HU, then review and confirm `APPLY`.
Old model releases and experiment artifacts were deleted. Overview remains empty
until new policies are exported; no diagnostic model was promoted as trained poker
strength.

## Browser

Run `npm run dev` inside `ui/`. The existing Test Live table offers computed
card-aware opponents, trained policies after both exports exist, and an explicit
random baseline. The chosen opponent source also supplies Bayesian likelihoods.
Hover an opponent for its compatible range, class/combo probabilities and reaction
view. Heavy analysis stays in the worker; no Python backend is required.

For static distribution, `npm run build` and `npm start` inside `ui/`. Rebuild after
publishing new models. Generated build outputs are disposable.

## Python

```python
from examples.phase2_demo import demonstrate
analysis = demonstrate(2)  # or 3
print(analysis["steps"][0]["analysis"])
```

The canonical rules, side pots, tournament transitions, corrected HU/three-player
collectors, exact references and deterministic checkpoints remain in source.
Historical [architecture](docs/HYBRID_BOT_PHASE2_ARCHITECTURE.md) and
[acceptance](docs/HYBRID_BOT_PHASE2_ACCEPTANCE.md) explain algorithmic scope and limits.
Rejected continuation models are not enabled. Browser learned-opponent play and
tracking are supported; learned exploitative continuation search is explicitly
unsupported, with reference calculation available.

## Tests

```sh
PYTHONPATH=tests:. .venv/bin/python -m unittest discover -s tests -p 'test_*.py'
node tests/run_browser_tests.mjs
```
