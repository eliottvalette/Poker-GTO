# Training performance investigation

Status: COMPLETE for the bounded workloads and validations below. Maximum eight CPU cores authorized. Existing training checkpoints and learning semantics must remain intact.

| Stage | Status | Evidence / next action |
|---|---|---|
| Isolate training from diagnostics | DONE | HU stopped externally; automatically started 3-max interrupted with SIGINT and checkpoint preserved |
| Hardware and workload selection | DONE | Mac17,2, 10 physical/logical cores, 24 GiB RAM; cap experiments at eight |
| cProfile generation, fit, full iteration | DONE | Profile real checkpoint with representative replay; separate cold start |
| Worker/chunk sweep | DONE | Identical tasks/seeds; 1/2/4/6/8 workers, repeated warm measurements, exact sample parity |
| Trainer threads and minibatches | DONE | 1/2/4/8 threads, separate throughput from optimization semantics |
| Source optimization | DONE | Only measured bottlenecks; numerical regression tests required |
| Complete-iteration confirmation | DONE | Same checkpoint and tasks, baseline versus selected configuration |
| Resume and configuration integration | DONE | Preserve checkpoint hashes; explicit compatible execution-budget changes only |
| Final report | DONE | Retain raw timings, profiles, rejected options, uncertainty and reproducible workflow |

Initial experiment ceiling: approximately 15 minutes of diagnostic elapsed time, including repeats, plus targeted correctness tests. Begin with representative small batches; expand only after measured costs. No paid compute, dependencies, production retraining or GPU-heavy workload.

Method references: [PyTorch performance tuning](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html), [PyTorch profiler](https://docs.pytorch.org/docs/stable/profiler). Actual APIs will be checked against the installed version. Profiler measurements are kept separate from uninstrumented throughput measurements.

## First measured results

HU checkpoint iteration13.64 fixed tasks, 11,363 nodes and6,423 generated samples. Three measurements/configuration; the first warm-pool measurement is excluded from warm medians for consistent comparison. Parent plus workers are not pinned to specific performance/efficiency cores. Background applications and thermal state remain sources of variation.

| Workers | Chunk limit | Warm median seconds | Exact sample parity |
|---|---:|---:|---|
| 1 | 32 | 2.977 | True |
| 1 | 128 | 2.762 | True |
| 2 | 32 | 1.959 | True |
| 2 | 128 | 2.074 | True |
| 4 | 32 | 1.821 | True |
| 4 | 128 | 1.317 | True |
| 6 | 32 | 1.733 | True |
| 6 | 128 | 1.201 | True |
| 8 | 32 | 1.701 | True |
| 8 | 128 | 1.295 | True |

The initial64-task probe cannot distinguish64 from128 effective chunk size. Complete512-task iterations validate the actual128 limit separately. Cold process startup is retained in raw results and excluded from warm throughput claims. Worker/chunk timings are wall time; recorded CPU-time estimates include parent work plus workers.

Fit comparison uses8,192 training example visits (128/64/32 updates at batch64/128/256), full final weighted loss scans, the same initialization, replay and seed. Changing batch size changes optimization semantics. Larger batches were benchmarked but are not selected solely for speed. Thread comparisons at fixed batch64 produced identical probe predictions.

| Network | Threads | Batch | Median seconds | Held-out loss |
|---|---:|---:|---:|---:|
| advantage | 1 | 64 | 2.171 | 248.318298 |
| advantage | 1 | 128 | 2.012 | 248.664406 |
| advantage | 1 | 256 | 1.938 | 249.881963 |
| advantage | 2 | 64 | 2.236 | 248.318298 |
| advantage | 2 | 128 | 2.055 | 248.664406 |
| advantage | 2 | 256 | 1.868 | 249.881963 |
| advantage | 4 | 64 | 2.259 | 248.318298 |
| advantage | 4 | 128 | 2.084 | 248.664406 |
| advantage | 4 | 256 | 1.879 | 249.881963 |
| advantage | 8 | 64 | 2.378 | 248.318298 |
| advantage | 8 | 128 | 2.183 | 248.664406 |
| advantage | 8 | 256 | 1.924 | 249.881963 |
| strategy | 1 | 64 | 2.255 | 0.414675 |
| strategy | 1 | 128 | 1.971 | 0.414671 |
| strategy | 1 | 256 | 2.016 | 0.415025 |
| strategy | 2 | 64 | 2.509 | 0.414675 |
| strategy | 2 | 128 | 2.029 | 0.414671 |
| strategy | 2 | 256 | 1.827 | 0.415025 |
| strategy | 4 | 64 | 2.147 | 0.414675 |
| strategy | 4 | 128 | 2.362 | 0.414671 |
| strategy | 4 | 256 | 2.164 | 0.415025 |
| strategy | 8 | 64 | 2.586 | 0.414675 |
| strategy | 8 | 128 | 2.470 | 0.414671 |
| strategy | 8 | 256 | 2.166 | 0.415025 |

## Source changes driven by profiles

1. Bulk conversion of compact little-endian float64 observation buffers to contiguous float32 arrays replaces per-observation Python tuple decoding and tensor construction. NumPy was already an installed project dependency. Invalid/overflowed inputs still fail explicitly.
2. Specialized sample byte accounting preserves the original occurrence/container-alias semantics exactly. Generic objects retain the general accounting path.
3. Deepcopy shares only proven immutable observation/sample payloads. Mutable containers, replay slots, RNG and accounting state remain independently copied.
4. One weighted replay snapshot per dataset and iteration replaces repeated `ProtectedReplay.samples` reconstruction. Sampling, weights and ordering remain unchanged.

First full-iteration comparison:23.639s baseline,18.569s source-only,14.045s with six workers/128-task chunks. All three final model and replay SHA256 fingerprints match. These are complete measured iteration times excluding the separately measured approximately5.85s explicit checkpoint save. Profiled runs are not used as throughput timings.

No minibatch, fit-update budget, collector, action abstraction, root distribution, model size or learning-rate change is needed for these gains. MPS is not used or claimed faster. Native CPU profiles show many small GRU/autograd operations rather than a large matrix-multiplication workload. A GPU port is not part of this accepted optimization.

## Second profiling pass and final repeated confirmation

The optimized full-iteration profile identified repeated weighted-replay reconstruction as the next avoidable cost; it was reduced to one materialization per dataset. Checkpoint-save profiling independently found6.27s of its11.05s instrumented time in recursive `asdict`; explicit packing of the same primitive schema reduced that component to0.25s. The uninstrumented checkpoint save decreased from approximately5.8s to2.99s. Reopening the new checkpoint preserves replay exactly. Tensor weights, contracts, checksums and atomic publication remain unchanged in meaning; serialized file bytes need not be identical.

Whole-iteration worker choices were retested in reverse order8/6/4/4/6/8 on each track. All workers must report their process IDs as ready before timing; initial snapshot installation during the actual iteration remains included. Each trial starts from the identical saved state, with128-task chunks, one trainer thread, unchanged batch64 and fit budgets. Each track produced one identical model SHA256 across all six trials. The HU hash also equals the original pre-optimization model hash. Separate baseline/source-only/final runs verified identical replay SHA256 as well.

| Track | Workers | Trial seconds | Median seconds |
|---|---:|---|---:|
| hu | 4 | 12.450, 11.359 | 11.904 |
| hu | 6 | 10.994, 11.561 | 11.278 |
| hu | 8 | 10.150, 10.413 | 10.282 |
| 3max | 4 | 10.135, 9.756 | 9.946 |
| 3max | 6 | 11.474, 9.026 | 10.250 |
| 3max | 8 | 10.871, 8.724 | 9.798 |

Final configuration:

- HU:8 workers,1 trainer thread,128-task chunks. Median complete iteration10.28s versus23.64s in the original single baseline timing, approximately2.30x throughput for this fixed next iteration. The baseline is one observation; no statistical confidence bound or universal speedup is claimed.
- Three-player:4 workers,1 trainer thread,128-task chunks. Although eight wins the small-generation probe, its whole-iteration median is within2% of four and timings overlap substantially. Four is retained because an eight-worker end-to-end advantage is not established. The fastest single trial is not used to select a configuration.
- Keep batch64 and the existing learning/update budgets. Larger minibatches give different optimization trajectories, so throughput alone does not justify switching them.

The Mac reports4 performance cores and6 efficiency cores (10 total),24GiB RAM. Experiments used at most8 worker processes and one trainer thread; the OS schedules the coordinator separately. No CPU affinity or hard process-wide core reservation is asserted. Submaximal aggregate CPU use does not imply that more threads improve elapsed time.

## Fit and native-profile confirmation

After source optimization, three-player fits were repeated at1/4/8 threads, same128 updates and batch64. One thread is consistently fastest for both models. Detailed metrics and profiles are in `3max-fit-confirmation`. Native PyTorch CPU traces cover10 warmed optimizer steps/network, with operator shapes retained. They confirm significant GRU and many small autograd/optimizer operations; no GPU speedup is inferred from these CPU measurements.

## Checkpoints and reproduction

Both original `runs/hu-batched/checkpoint.pt` and `runs/3max-batched/checkpoint.pt` SHA256 values are unchanged. Actual resumes were validated at iterations13 and16 under the selected configs, without advancing or saving either production run. Execution-only changes are opt-in at the runner API and enabled/audited by the standard workflow: workers, trainer threads and generation chunk size. Learning-rate, sampling, minibatch and fit-budget changes are still incompatible resume requests. Metadata and console output identify accepted execution-setting changes.

Use the existing entrypoints to resume when desired:

```sh
.venv/bin/python -u train-hu.py
.venv/bin/python -u train-3.py
```

They run sequentially when submitted to the same shell. No production session was started by the benchmarking workflow. No model was exported or accepted for poker quality.

Importable reproduction lives in `training/performance_experiment.py`: `sweep`, `iteration_trial`, `iteration_grid`, `fit_confirmation`, and `native_fit_profile`. Pass explicit checkpoints/output directories and guard process-pool calls with `if __name__ == "__main__":`. No CLI was added. `sweep` rewrites its declared report; whole-iteration trials require a new output directory. Preserve existing reports when defining a different experiment.

Artifacts: `runs/performance/summary.json` indexes selected settings, whole-iteration measurements and original-checkpoint integrity. The directory retains raw JSON, cProfile binaries/text, native Chrome traces, comparison checkpoints, metrics and original encoding/accounting source snapshots. Current storage is approximately5.6GiB. These are diagnostic artifacts, not published policies.

## Validation and remaining limits

64 relevant tests passed after integration:46 training/Deep-CFR/weighting/checkpoint/workflow/optimization tests and18 feature/ONNX-export tests. A final four-test performance-contract rerun also passed after adding mutable-buffer isolation coverage. Exact tests cover packed-buffer encoding, mutable versus immutable deepcopy, generic versus optimized byte accounting, checkpoint primitive schemas, worker/chunk replay parity and deterministic resume. Source/config changes were restricted to measured performance bottlenecks and execution settings.

Workload limits: one recent HU and one recent three-player checkpoint, small frozen probes plus repeated full next iterations; full-iteration comparisons here do not include a B-fit batch. B fits were measured independently. Final weighted-loss evaluation still scans the replay. Coverage reconstruction and serialization remain measurable costs; they were not removed to inflate throughput. The native trace excludes cache construction so it complements, rather than replaces, cProfile.

Two repetitions per whole-iteration setting are insufficient for tight latency confidence intervals; background applications and heterogeneous-core scheduling remain variable. Results select conservative settings on this machine, not a global optimum across all poker states/hardware. MPS, GPU porting, altered neural architectures, learning-rate/batch changes and weaker poker abstractions were not promoted as performance shortcuts. Playing-strength conclusions are unchanged.


## Migration extraction and ONNX preparation

Migration options 1, 2 and 3 no longer restore a TrainingRunner, replay reservoirs,
RNGs or tournaments. Exports share the average-policy artifact builder with the
solver and use the original deterministic ONNX probe without rereading the full
checkpoint. Published identical releases can be reused after checksum validation.
Preparation and publication display elapsed times. No production catalog was
changed by these diagnostics.

The first extraction-only experiment was rejected as a performance result:
HU took 39.42 s and three-player 39.95 s, versus 27.84/30.38 s for full runner
restore plus export (the latter excluded inference-artifact validation). Profiling
identified a second cost: inference artifacts duplicated the complete diagnostic
metrics history. The instrumented HU extraction spent 21.47 s reloading that
artifact alone. New inference artifacts exclude this history; complete metrics
remain in the training checkpoint. Model weights and inference provenance are
unchanged. Existing immutable releases remain untouched.

Final uninstrumented measurements, one trial per current production checkpoint:

| Track | Iteration | Checkpoint bytes | Extract and validate | Inference bytes | Inference reload |
|---|---:|---:|---:|---:|---:|
| HU | 109 | 1,347,202,093 | 21.19 s | 77,922 | 1.31 ms |
| Three-player | 117 | 1,279,870,957 | 22.22 s | 77,922 | 1.36 ms |

These timings exclude checkpoint copying, bundle hashes and ONNX conversion.
The existing envelope still deserializes all primitives; this change does not
promise instantaneous first extraction. Raw evidence is in
`runs/performance/migration-extraction.json`, `migration-extraction.prof`,
`migration-extraction-profile.txt`, and `migration-final.json`.

Validation: 22 workflow/Deep CFR/ONNX tests passed, followed by the focused updated
artifact-contract test. Tests cover both tracks, exact weights and metadata parity,
identical predictions, ONNX parity, rejection of inconsistent versions, no runner
restoration during export, reuse of published releases, publication conflicts and
failure cleanup. Full training-resume validation remains unchanged.
