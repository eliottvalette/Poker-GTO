# Corrected training and live policy lifecycle

## Current state

Old experiment artifacts, legacy average-policy files, published Python checkpoints, ONNX releases and old run outputs were removed explicitly. The catalog `ui/public/policy/index.json` is empty. Overview therefore has no trained strategy until a new export is published. No old model is silently substituted. Historical Phase 2 documents describe previous measurements; their artifact directories no longer exist.

## Run training

From the repository root, run the tracks separately:

```sh
.venv/bin/python -u train-hu.py
.venv/bin/python -u train-3.py
```

Both use the corrected collectors, invariant card features and these explicit settings:

| Setting | Value |
|---|---|
| Recorded-player card sampling | `stratified_recorded_opening` |
| Fit limits | At most 4 epochs and 800 optimizer updates per fitted model |
| Advantage cadence | Once after each complete frozen generation batch |
| Average cadence | First batch, then every third batch |
| Initialization | Fresh first fit; reuse model weights afterward; fresh Adam optimizer |
| Root rollout policy | Current advantage regret matching; independent of B |
| Advantage / strategy replay capacity | 50,000 / 50,000 per track |
| Protected opening allocation | 20%, with inverse-inclusion weights |
| Traversals per player / iteration | HU 256 (512 total); three-player 128 (384 total) |
| Generation transfer chunk | 128 traversals; frozen profile across the complete batch |
| Encoding | Fit-local tensor cache, 256 MiB tensor storage guard |
| Worker processes / trainer threads | HU 8 / 1; three-player 4 / 1 |
| Accounted byte budget per retained memory | 512 MiB |
| Maximum nodes per traversal task | 200,000 |
| Session duration | 30 minutes per invocation |

The duration is checked between complete iterations; a session can exceed it by the final iteration. Change duration in `training/settings.py`. Track settings live in `configs/train_hu.json` and `configs/train_3max.json`. The previous 50/20-epoch settings were rejected by the measured audit. The new limits are an efficiency-tested experimental configuration, not convergence guarantees. Inspect generated and retained coverage, effective weights, held-out fitting and policy diagnostics as memory grows.

The first run initializes fresh models and replay. Subsequent invocations resume only their own `runs/<track>-batched/checkpoint.pt`. A missing checkpoint with existing training history is an error; deleting a checkpoint alone does not authorize overwriting its history. A published checkpoint is never an implicit resume source. Explicit programmatic `train_track(..., resume_active=True)` remains available when deliberately resuming a compatible published checkpoint.

The old `runs/hu` and `runs/3max` checkpoints are preserved unchanged. Main entrypoints now use distinct `runs/hu-batched` and `runs/3max-batched` directories. They do not silently resume or overwrite the rejected runs. No new 30-minute session was launched. See [TRAINING_CADENCE.md](TRAINING_CADENCE.md) for measured full-batch pilots and limitations.

Warm initialization explicitly reuses weights, not optimizer moments. The within-fit optimizer state is not needed across fits because the declared policy resets Adam each fit. Both fresh and warm modes remain available under `fit_schedule`, and resume reproduces this behavior. Average-policy metrics explicitly mark skipped fits; zero model drift during a skipped fit is not reported as convergence. Export training metadata records the actual last average-model fitting iteration.

The root distribution, corrected collectors, iteration/reach weights and uniform all-nonpositive-regret fallback are preserved. Terminal roots are still counted, not silently rejected or replaced. Useful opening coverage remains a measured diagnostic, not a guaranteed property of a batch.

## Validate, then export

Review `runs/hu-batched` and `runs/3max-batched` metrics and coverage before choosing a checkpoint. Iteration count, low AA fold frequency or low fitting loss alone do not establish poker quality. The corrected pipeline cannot manufacture missing per-hand supervision; opening protection guarantees allocated retention, not adequate coverage of every class/context.

```sh
.venv/bin/python migrate.py
```

Choose **2** to export three-player UI policy and **3** for HU. The menu selection starts preparation, validation and publication without another confirmation. The source, iteration and checksums are printed before publication. Export publishes new immutable bundles and atomically selects them in the UI catalog. Menu 1 activates a Python reference checkpoint separately; it is not required to display an exported policy. Technical export/parity validation is not a playing-strength certificate.

All three migration options extract average-policy weights and metadata directly from the checksummed checkpoint. They do not instantiate a training runner, restore replay objects, or restore tournament/RNG state. ONNX export uses the same deterministic reference observation as training export, without a second checkpoint load. Preparation prints each stage and its elapsed time. Inference artifacts retain model weights and training provenance, but no longer duplicate the full per-iteration diagnostic history, which remains in the complete training checkpoint. Identical already-published policies are reused after checksum verification. The current checkpoint envelope still requires complete primitive deserialization, including replay bytes; this is not a random-access model-only file format. Training resume retains its full replay validation. Migration validates the checkpoint contract, configuration hash, dedicated track, model version, strategy coverage, and exported model; bundle integrity and ONNX numerical parity checks remain enabled.

Overview and Hero's displayed NN probabilities load the newly exported models. An already open development page polls the catalog; a static export must be rebuilt to include new model files.

## Test Live opponent selection

Test Live now starts with **Computed conservative**, an explicit card-aware scripted behavior, not uniform random play and not a trained network. Selecting another opponent starts a new tournament:

- Card-aware scripted profiles.
- **Trained policy**, enabled when both HU and three-player models are exported; player-count routing follows the real hand.
- **Random baseline**, explicitly selected only.

For trained opponents, ONNX inference supplies their actual actions and the hypothetical-hand likelihoods used by Bayesian range tracking. These evaluations receive only the acting candidate hand and public observation. Worker inference is cached and uses absolute asset URLs. The opponent likelihood selector is locked to the chosen opponent source; Hero's human-action likelihood remains an explicit assumption.

Reference analysis still uses local CFR/computed baseline continuations. Published models are not yet a synchronous continuation source for the browser's exploitative tree search: selecting that unsupported combination returns an explicit error rather than substituting a different policy. Neural action buttons, opponent play, range inference and hybrid action-EV calculation remain distinct.

The engine and live table operate without any model. Empty/corrupt model catalogs clear stale loaded models; missing learned policies never silently fall back to random opponents.

## Verification during finalization

The real static-browser test selected trained opponents, played a Hero call, executed ONNX opponent responses and obtained updated Bayesian posteriors from the worker. It used the old artifacts solely to test the interface before those artifacts were deleted. No old weights were retained or re-exported. Regression tests also force opponent actions through an injected asynchronous policy and independently compare external candidate likelihoods with the existing Bayesian update.

Final regression result: 196 Python tests passed in 53.119 seconds; 140 browser tests passed. Production build and TypeScript checking passed. A second actual Chrome check after clearing the catalog confirmed there are no stale policies, trained-opponent selection is disabled until both exports exist, and computed live play/range tracking still work. Generated production/development caches and static outputs were removed after validation. `npm run dev` recreates development output; predev/prebuild creates an empty catalog only when absent and never overwrites an existing publication.

## Measured execution configuration

See [TRAINING_PERFORMANCE.md](TRAINING_PERFORMANCE.md) for repeated CPU benchmarks, native and Python profiles, full-iteration parity and rejected settings. Main entrypoints may now resume with explicitly configured changes to `workers`, `trainer_threads` and `generation_batch_size`; they print and record those execution-only changes. Models, replay and RNG are restored from the existing checkpoint. Changes to learning budgets, minibatch size, sampling, learning rate or architecture are still rejected. No checkpoint is rewritten until a resumed session actually saves its state.
