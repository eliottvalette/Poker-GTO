# HU traversal budget recovery

The HU checkpoint at iteration 40 could not generate its next external-sampling
batch with `max_nodes=5000`. This limit applies separately to the recursive
advantage pass and average-strategy pass, not to the whole iteration. Terminal
nodes count toward the limit. Depth was not the binding constraint.

A generation-only replay restored the checkpoint, root sampler, frozen model
and task seeds. Of 256 tasks, two exceeded the original limit:

| Task | Traverser | Seed | Pass | Total nodes after completion | Maximum depth |
| --- | --- | --- | --- | --- | --- |
| 104 | 0 | 48896305 | advantage | 6137 | 15 |
| 218 | 1 | 451369277 | advantage | 6121 | 15 |

Both completed with a 10000-node guard. All 256 tasks completed when those two
were replayed from their original seeds with that guard. The measured mean was
427.91 total nodes/task and the p95 was 2194. Serial generation and diagnostic
replays took 79.97 seconds; this is not the normal eight-worker iteration cost.
Raw measurements are in `hu-traversal-budget-profile.json`.

`configs/train_hu.json` now uses a 20000-node guard, with depth still 64. This
provides measured headroom without truncating branches, discarding samples,
changing targets, modifying the root distribution or retrying automatically.
The three-player configuration remains unchanged. Future learned policies may
still exceed the finite guard; that failure remains explicit.

## Controlled resume

`TrainingRunner.load_checkpoint(path, config)` still requires the exact config
hash by default. The timed workflow explicitly enables
`allow_budget_increase=True`. This permits only monotone increases of
`max_nodes` and/or `max_depth`. All other configuration changes, reductions,
checksum errors and schema incompatibilities are rejected.

The checkpoint's stored config hash is verified before applying an increase.
Models, replay, reservoir RNG, solver RNG, root sampler, iteration and historical
metrics are restored unchanged. The change is announced and recorded in
`metadata.resume_budget_changes`, including original/new config hashes,
iteration, source checkpoint, before/after limits and current source metadata.
New iteration metrics include their config hash and traversal limits. An exact
reload of the updated checkpoint does not append a duplicate change.

No checkpoint is rewritten merely by loading it. The next successful save
atomically stores the new config and its audit record. On a traversal failure,
the last complete iteration is saved and its path is printed; no incomplete
iteration or sample batch is published. Errors now include task ID, frozen model
version, traverser, seed and traversal pass for reproducible diagnosis.

Training and policy publication are independent. Increasing a guard does not
activate a reference policy or publish an ONNX export.

## Real resume validation

One normal eight-worker iteration resumed from checkpoint 40 with the audited
20000-node guard. It completed iteration 41, fitted both HU networks and atomically
saved `runs/hu/checkpoint.pt`. No hour-long session was launched and no policy
was activated or exported by this validation.

- 256 traversal tasks; 109544 total nodes; maximum 6137 nodes/task, depth 15.
- 23428 generated advantage samples and 270 generated strategy samples.
- Iteration wall time: 30.28 seconds; generation 19.81 seconds; neural fit 10.07 seconds.
- Timed session including metrics/checkpoint persistence: 34.48 seconds.

This unusually large batch should not be treated as a stable throughput estimate
for future learned policies. `hu-resume-validation.json` records the result and
new config hash. A normal `python train-hu.py` launch now continues from iteration
41. Checkpoint 40 remains available as its periodic numbered artifact.
