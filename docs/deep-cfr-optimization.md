# Deep CFR performance refactor

The implementation reduces repeated work while preserving per-hand chip EV,
external-sampling branches, legal actions, targets, average-strategy weighting,
fresh network fitting and the existing checkpoint/model/feature schemas.

## Source changes

- `HandState.clone()` explicitly copies mutable containers and mutable players.
  Frozen blind levels, public action events and card tuples are shared safely.
- `infoset.observe_fields()` owns raw observable encoding. `observe()` adds lossless
  JSON recall for tabular keys and diagnostics. `observe_neural()` combines the
  raw fields with deterministic features without constructing diagnostic JSON.
  External Deep CFR uses the neural observer; the reference traversal defaults
  to its original lossless observer.
- Pure card structural features use an 8192-entry LRU cache keyed exclusively by
  hero cards and currently visible board. Numeric betting features are recomputed
  at each state. No opponent private cards, future cards or belief inputs enter
  the cache key or neural inputs.
- `encode_batch()` unpacks each compact numeric/history vector once per sample.
  Full validation, padding, packed GRU operations and float32 conversions remain.
- A `FrozenStrategy` is loaded once per snapshot in each worker. Worker cache
  identity includes the complete serialized snapshot bytes, not just iteration.
  A different track or different weights with the same version refresh the model.
- `run()` and `run_for()` own one process pool across their outer iterations.
  Each worker uses one Torch inference thread. Main-process root inference and
  fitting use the configured trainer thread count. Workers never fit models.
- `DeepCFRSolver.fork()` isolates RNG and mutable iteration bookkeeping while
  sharing previous models/replay read-only. The solver still builds separate new
  replay and fresh networks before publishing the completed iteration.

The standalone APIs still own and close their workers when no executor is
provided. The session API owns a supplied executor and closes it when the session
ends. Python's [ProcessPoolExecutor documentation](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ProcessPoolExecutor)
describes the lifetime and process context used here.

## Measured real workload

Measurements use six actual HU tasks reconstructed from checkpoint 40: IDs
0, 20, 104, 128, 218 and 220. They include both tasks that exceeded the old guard.
Both implementations use the same snapshot, root deals, task seeds, legal action
abstraction, 20000-node guard and one Torch CPU thread. The six tasks visit 13280
nodes in total. Fitting uses all 20000 retained advantage samples, two epochs,
batch size 64 and the same initialization/shuffle seed.

The comparison baseline is source commit
`804858ce6ca1a18cd903d1b613bdabfa3682520d`, run in an isolated source copy with
the captured real inputs. Budget-error message additions do not affect completed
traversals. No synthetic replay duplication or replacement labels are used.

| Measurement | Before | After |
| --- | ---: | ---: |
| Serial generation median, three repetitions | 11.44 s | 6.33 s |
| Advantage fitting, one measurement | 7.36 s | 6.19 s |
| Deep-copy loaded solver vs staging fork | 0.681 s | 0.000066 s |

The generation speedup is 1.81x on this selected workload. Individual timings
varied substantially; the fit and worker figures are limited measurements, not
stable full-session predictions. A separate optimized four-worker measurement
took 4.39 s for its first batch and 2.30 s for its second batch with the same pool.
Cold pool startup remains: retaining workers helps subsequent iterations. No hour-long run or end-to-end pilot acceleration
is claimed from these numbers.

All generated states, sample weights, targets, values and node counts compare
exactly. Advantage-network weights after the controlled fit are bitwise equal.
An independent reconstruction through `traversal_tasks()` also matched the
captured baseline. Persistent-worker tests cover changed weights at the same
version and multiple outer iterations using the same executor.

## Reproduction and retained outputs

`training.benchmark.benchmark_checkpoint(...)` is importable, requires an explicit
checkpoint, track, task IDs, limits, workers, repetition count and output path,
and does not advance or publish a training run. It restores ambient RNG/thread
state. Acquire the workflow's compute lock around a benchmark to avoid concurrent
training. The root sampler used for diagnosis is an isolated copy.

Full raw profiles, captured inputs, sample comparisons, fitted weights, timings
and the isolated baseline source are retained under
`profiling/deep_cfr_optimization/` (local generated artifacts, ignored by Git).
`docs/deep-cfr-optimization.json` contains the paired machine-readable summary.
The benchmark output is separate from policy and working checkpoint paths.

## Remaining costs and boundaries

Each decision still performs a single-state PyTorch/GRU inference. Task granularity
is unchanged. Fitting still restarts both networks from their reservoirs every
outer iteration, and the average policy still drives tournament root generation.
Changing those semantics or introducing batched traversal scheduling needs its
own measured comparison. Metrics and complete checkpoints retain their existing
atomic persistence behavior.

The node guard remains finite and explicit. No action pruning, regret clipping,
sample dropping, fabricated leaf values, GPU work or outcome-sampling default
change was introduced. Existing trained checkpoints and ONNX contracts remain
compatible. The active HU checkpoint was not advanced by these benchmarks.
