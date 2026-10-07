# Deep CFR pilot preflight — 2026-10-06

> Historical measurements for the former per-seat advantage models and feature
> schema 1. Current training shares one advantage model/replay per track;
> see [shared-advantage.md](shared-advantage.md). Total replay capacity is preserved.

**PILOT READY. No substantial training has been launched.**

The original cost extrapolation below is retained as historical measurement.
[Production-sized CPU profiling](deep-cfr-profile.md) supersedes its runtime
estimate: approximately 25–40 minutes for the pilot under the measured workload.

The final isolated preflight completed two real iterations of both tracks, at
four traversals per player per iteration. It exercised uniform and subsequently
learned nonuniform policies, neural fitting, scheduled evaluation, atomic
checkpoints and both ONNX exports. The complete machine-readable evidence,
including losses, node distributions and coverage, is
[deep-cfr-preflight.json](deep-cfr-preflight.json).

The checkout is `main` at the base commit below. The remote's former GTO_Bot URL
redirects to Poker-GTO. Implementation changes remain uncommitted; the source
fingerprint identifies the measured code.

| Measurement | Value |
| --- | --- |
| Git base commit | 8b7c63d9efe130c58b701efa1c2c500768a047e7 |
| Uncommitted source fingerprint | ffd8e628f55333e353721bd841a34625c693e7b96c0771567c1c08eeafdc145a |
| Feature schema | 1 |
| Architecture | cards8_numeric32_historyGRU32_head64_features1 |
| Parameters/model | 17329 |
| Traversal mode | external_sampling |
| Workers / worker Torch threads / trainer threads | 8 / 1 / 1 |
| Root mixture | 50% on-policy tournament / 25% synthetic / 25% stratified |
| Proposed iterations / traversals per iteration | 50 / 640 |
| Proposed total traversals | 32000 |
| Measured iterations / traversals | 2 / 40 |
| Tasks = traversals per second | 7.95 |
| Nodes per second | 930 |
| Nodes/traversal mean / p95 | 116.95 / 455 |
| Samples/traversal | 22.85 |
| Matched replay bytes/sample before / after | 9024 / 3512 |
| Generated replay bytes/sample mean / p95 | 4024 / 4662 |
| Replay capacity per advantage / per strategy | 10000 / 10000 |
| Total replay capacity | 70000 |
| Retained replay projection | 268.6 MiB |
| Staged replay projection | 805.8 MiB |
| Total RAM planning estimate | 3.33 GiB, plus unmodeled checkpoint buffers / allocator growth |
| Worker / trainer measured peak RSS | 187.1 / 341.8 MiB |
| Mean CPU cores / host CPU utilization | 3.86 / 38.6% |
| Host hardware | 24 GiB RAM / 10 logical CPUs |
| GPU | None |
| Measured complete checkpoint | 4.09 MiB |
| ONNX per policy | 73186 bytes |
| Projected checkpoint/artifact footprint | 3.37 GiB |
| Estimated 50-iteration pilot | 1.51 hours |
| Estimated 500 iterations, same capacities | 15.09 hours |

These runtime and storage values are extrapolations, not guarantees. Measurement
includes process spawning and root generation at a deliberately tiny task count.
Full-reservoir fitting is linearly scaled to 70,000 retained samples. The total
RAM estimate includes three conservative replay copies, measured worker/trainer
RSS and configured generated/pending budgets. Full-size checkpoint serialization
buffers and allocator growth can add memory beyond it. The host has 24 GiB RAM;
pilot capacities leave considerable headroom. No larger capacities are adopted.

The 100,000-per-advantage / 300,000-per-strategy hypothesis would retain 1.1 million
samples across both tracks: about 4.1 GiB replay using the overall measured mean,
and about 12.4 GiB across three conservative copies, before workers, batches,
checkpoint buffers or allocator overhead. Strategy samples were smaller in this
small probe, but the observed distribution is too limited to choose final large
capacities. Existing per-reservoir byte budgets must also increase explicitly
before those counts could fit. Evaluate a larger reservoir only after the pilot.

A traversal here includes the regret walk plus the separate average-policy pass.
One HU root in the probe settled at blind posting; it remains counted as a valid
zero-decision root. All generated roots are accounted for. Both tracks reached
all six blind levels in the two-iteration mixture; complete Cartesian coverage
is separately validated by root-sampler tests.

The matched before/after sample comparison uses the same fixed observations and
metadata with identical conservative object accounting, replacing raw
Observation/JSON recall with compact NeuralObservation. It is separate from the
mean of samples actually generated during traversal. Accounting is conservative
and does not equal process RSS.

Two iterations do not establish policy quality. Fixed-probe L1 drift after the
second iteration was approximately 0.134 (3-max) and 0.126 (HU). Conditional river
BR gains changed from approximately 0.984 to 0.749 chips (3-max) and 0.358 to 0.394
chips (HU). Held-out losses were not consistently decreasing in this tiny fit.
These results validate the evaluation path; they do not justify a convergence
claim or advanced online resolving.

Validated commands:

```sh
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'  # 113 passed
node tests/run_browser_tests.mjs                             # 77 passed
.venv/bin/python -m scripts.smoke_validation                  # passed
.venv/bin/python -m compileall -q features training ml
./ui/node_modules/.bin/tsc --noEmit -p ui/tsconfig.json        # passed
(cd ui && npm run lint)                                      # passed
(cd ui && npm run build)                                     # passed, static export
git diff --check                                            # passed
```

ONNX emits the existing GRU/exporter warnings; singleton dynamic-history inference
parity passes. Browser feature parity passes for both tracks and all streets.
A real WASM session with useful trained artifacts remains a phase-H check.

Recommended pilot: retain the supplied 50-iteration, 640-traversal/iteration,
8-worker CPU configuration with 10,000 samples per reservoir, two fit epochs,
64 batch size, 3e-4 learning rate, external sampling, 5,000 nodes / 64 depth,
checkpoint/evaluation every five iterations. Stop on explicit failures; never
censor over-budget traversals or invent leaf values. Review the first checkpoint
interval before increasing compute or capacities.

After explicit approval, use the Python API:

```python
from training.config import load_config
from training.runner import TrainingRunner

runner = TrainingRunner(load_config("configs/deep_cfr_pilot.json"))
runner.run()
runner.export(onnx=True)
```

See [the migration report](deep-cfr-training.md) for contracts, changed modules,
resume/export usage, known approximations and the deferred live architecture.
