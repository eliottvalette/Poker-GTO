# Deep CFR CPU profile — 2026-10-06

The earlier 90-minute pilot estimate was too pessimistic. It multiplied the cost
of starting process pools for tiny task counts as if that cost increased linearly
with traversals. Production-sized frozen lots measure approximately 9.5 seconds
for 640 tasks with eight workers. Seven fresh full-reservoir fits take approximately
17.5 seconds combined. A more useful planning range for the 50-iteration pilot is
**25–40 minutes**, conditional on the measured state shapes and weak learned policy.
This is not a runtime guarantee for stronger later policies.

No production engine, solver, feature or training semantics were changed for this
profiling task. No substantial pilot or GPU job was launched. Two importable
profiling scripts and this report were added.

## Evidence and workload

The [machine-readable summary](deep-cfr-profile.json) contains timings, the full
worker matrix, microbenchmarks and the planning calculation. Complete primary
reports and raw `cProfile` files remain locally under
`profiling/deep_cfr_costs/`: `report.json`, `supplement.json`,
`capacity_stage_costs.json`, and the `.prof`/`.txt` files. They are ignored by Git
because they are local diagnostic artifacts, not training outputs.

Traversal comparisons use identical roots, cards, seeds and frozen neural weights
within each worker sweep. Both tracks were measured under uniform and learned
strategies. Small lots contain 16 or 128 tasks per track; production lots contain
384 3-max and 256 HU tasks, with the production player-block assignment. Production
counts were repeated twice per worker setting. Full-capacity fitting uses real
emitted state shapes, repeated to 10,000 records per model; those numerical values
are benchmark data, not independent learning examples or convergence evidence.

Unprofiled `perf_counter` timings determine throughput and runtime estimates.
`cProfile` identifies call attribution; its Python instrumentation overhead means
its percentages should not be treated as exact uninstrumented wall percentages.
The checkpoint benchmark additionally gives each record independently owned
numeric/history bytes. Sharing repeated sample objects would otherwise
artificially shrink pickle output. Both synthetic checkpoint artifacts have a
profiling-only contract and are explicitly rejected by `TrainingRunner.load_checkpoint`.

## Parallel scaling

| Track / tasks | 1 worker | 2 workers | 4 workers | 8 workers |
| --- | ---: | ---: | ---: | ---: |
| 3max / 16 | 0.45 s | 0.86 s | 0.76 s | 1.07 s |
| hu / 16 | 1.57 s | 1.43 s | 1.10 s | 1.50 s |
| 3max / 128 | 4.37 s | 3.23 s | 1.93 s | 2.18 s |
| hu / 128 | 8.01 s | 4.80 s | 3.07 s | 4.89 s |

| Production lot | 4 workers, median of 2 | 8 workers, median of 2 |
| --- | ---: | ---: |
| 3max / 384 | 5.69 s | 4.62 s |
| hu / 256 | 5.74 s | 4.87 s |
| Combined / 640 | 11.43 s | 9.50 s |

Four workers win at 128 tasks, while eight win at the actual full iteration size.
Retain eight for the proposed pilot; small smoke jobs should use fewer workers.
Pool startup is a fixed component, not a per-traversal constant. The current code
creates two pools per outer iteration, one per track. IPC payload serialization
alone was small: approximately 1.4 ms for 128 root tasks and 8.2 ms for 128 result
objects in the isolated pickle benchmark. Those numbers exclude snapshot payloads,
queue transport and scheduling, which are included in end-to-end generation.

## Traversal attribution

For the learned HU profile, disjoint high-level regions inside `generate_samples`
account for approximately:

| Region | Instrumented cumulative share |
| --- | ---: |
| HandState clone, primarily recursive deepcopy | 29% |
| Raw observation construction | 19% |
| Strategy inference, including neural preparation | 36% |
| Sample sink, including features and accounting | 8% |

The 3-max proportions are similar. These regions explain most observed cost.
The strategy region includes `encode_batch`, so its subcomponents must not be
added again to the total.

`observe()` still constructs lossless recall JSON at each decision, even when
neural inference is the consumer. Compact replay removes that JSON from storage
but does not yet remove its construction from the hot path. The builder also
serializes public history events with `dataclasses.asdict`.

A singleton forward with pre-encoded input measured about **117 microseconds**;
raw-observation-to-forward measured about **233 microseconds** on the fixed
preflop probe. About half of that example's inference time is preparation.
Actual postflop/history shapes differ, so these are microbenchmarks, not universal
per-node constants.

Model reload is real but smaller than the traversal hot path. On 16 loaded-worker
tasks, snapshot deserialization plus advantage-network construction/loading took
approximately 0.095 seconds in the 3-max profile (10% of that instrumented short
lot), and 0.051 seconds in the longer HU profile (1%). A single network construct
and load measured approximately 0.56 ms unprofiled. Caching frozen models by
track/version is useful, but it does not eliminate the dominant cloning and
input-preparation costs.

Card-derived features cost approximately 9–12% of the instrumented learned
traversal profiles, included in the inference/sink regions above. `card_features`
checks up to 52 unseen cards through repeated straight-window construction and
can be recomputed for both inference and replay emission at the same node.
The same hero cards and public board also recur across betting branches.
**Equity is not called by the current traversal pipeline**; it is not a current
bottleneck. Root generation itself measured approximately 0.75 seconds for the
complete two-track task batch.

An isolated selective-copy prototype copied the same HandState in approximately
0.099 seconds per 10,000 clones versus 0.469 seconds for deepcopy: about **4.8x**.
It copies all mutable containers/player objects and shares immutable values.
This was only a prototype with equality checking, not a production change or a
complete branch-independence certification.

## Central fitting, replay and checkpoints

| Stage | Measured cost |
| --- | ---: |
| Advantage fit, 10,000 records, 2 epochs, batch 64 | 2.47 s/model |
| Average-policy fit, same capacity/settings | 2.61 s/model |
| Five advantage + two average fits | 17.54 s |
| Runner deepcopy at 70,000 independent replay records | 1.36 s |
| Solver replay staging, another full copy | 1.34 s |
| Complete independent-buffer checkpoint | 3.71 s / 221.3 MiB |
| Checkpoint repeated after loading same records | 3.41 s |

`encode_batch` takes **62–63%** of the instrumented full-fit time. For the advantage
fit, tensor construction alone takes 26% and history decoding 20%, both contained
inside encoding. The network forward accounts for approximately 13% and backward
16%. Preparing the data is a larger immediate target than increasing the network
or moving its arithmetic to a GPU.

The encoder reads the history property five times per consumed observation:
validation, row validation, lengths, slice length and tensor creation. The profile
records 130,000 history decodes for 26,000 consumed observations in one 10,000-record
fit. Training and final train/held-out evaluation repeatedly recreate those inputs.

Replay admission spends most of its instrumented cost inside generic recursive
`sample_bytes` accounting. It is called during generation, collection and replay
admission. The measured append cost was 0.54 s per 10,000 advantage samples and
1.97 s for strategy samples; the latter had one timing repetition and should not
be treated as a precise scaling constant. Full replay staging costs approximately
2.7 s each iteration. Frozen samples permit investigating structural copying
without copying every immutable record, while preserving separate mutable lists
and reservoir RNG states.

In the full checkpoint profile, `dataclasses.asdict`/`pack_memory` took about 67%
of the instrumented time, with repeated deep copying and conversion of immutable
records. Binary buffer ownership substantially affects archive size. The initial
shared-record benchmark produced about 40 MiB; independently owned buffers produced
about 221 MiB. Only the latter is useful for a realistic capacity projection.
The post-load capacity diagnostic reached 2.58 GiB parent peak RSS, including both
loaded raw checkpoint data and live replay; it is not a direct total RAM estimate
for the normal runner with eight workers.

Average-policy fitting is not needed to compute the next regret-matched strategy:
that uses advantage models. However, the current average policy drives the 50%
on-policy tournament-root generator. Fitting it less often is a possible explicit
experiment, but changes root-policy freshness and needs separate version/metrics
handling. Fresh advantage fitting remains the established baseline. Neither fit
frequency nor initialization semantics was changed during profiling.

## Corrected planning estimate and priorities

Measured steady-capacity components sum to approximately **32.5 seconds per iteration**, or **27.1 minutes for 50 iterations**, before unmeasured log growth and strategy changes. The planning range is **25–40 minutes** for the pilot and **4.2–6.7 hours** for 500 iterations at the same capacities and comparable state distribution. Early iterations have less replay; later strategies can generate more expensive trees. These are conditional estimates, not promised completion times.

Prioritize conservative CPU changes in this order:

1. Decode compact history/numeric fields once per encoding and prepare reusable
   fit inputs rather than reconstructing tensors each epoch/evaluation pass.
2. Replace recursive HandState deepcopy with a tested copy of mutable fields;
   preserve all rules, public history and branch isolation.
3. Separate neural observation construction from diagnostic recall serialization,
   using one shared source of raw observable fields.
4. Cache or compute card-only descriptors efficiently; retain exact descriptors
   and all raw cards, and avoid recomputing them for inference plus replay.
5. Cache frozen models per worker/track/version, then evaluate persistent pools
   and batched task transport while retaining atomic failure and byte budgets.
6. Specialize replay accounting/serialization for the immutable compact schema
   and avoid redundant deep copies in staging/checkpoint preparation.

Changing average-policy frequency, continuing optimization instead of fresh fits,
adding GPU inference or redesigning traversal batching are later measured
experiments, not prerequisites to the first meaningful policy training.

Reproduce the primary and supplemental profiling with the existing environment:

```sh
.venv/bin/python -m scripts.profile_training
.venv/bin/python -m scripts.profile_training_supplement
```

The scripts run bounded diagnostic workloads and overwrite their exact local
profiling targets. They do not launch the pilot. Python compilation passed;
both synthetic checkpoint artifacts were checked to fail explicit training resume.
No unchanged engine/UI suite was rerun for these diagnostic-only additions.
