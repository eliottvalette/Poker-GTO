# Continuous self-play

The production migration replaces the finite scoped-solver allocation with full-hand
Deep CFR self-play, independently for HU and 3-max. There is no total iteration
limit in the service: `run_continuous` repeatedly calls the timed iteration runner
and resumes the selected checkpoint after a restart. `outer_iterations` belongs to
the separate finite batch API and does not stop this service.

## Learning contract

`configs/train_{hu,3max}_selfplay.json` owns the new learning configuration.
Each iteration samples 64 traversals per player, using current advantage policies
for tournament root generation and frozen self-play traversals to terminal chip
settlements. The objective is per-hand chip delta, not tournament payout or ICM.
Legal betting actions are discretized by the existing engine.

The average strategy is fitted every iteration. Shared advantage and average
networks are fitted from fresh initialization against bounded historical reservoirs;
this does not continue the previously audited persistent specialist weights.
Each fit is capped at 256 optimizer updates. Both networks and replay are saved;
subsequent traversal policies use the previous fitted network before the next fit.
No old model, replay, or scoped solver table is imported into this new run.

This implements an experimental approximation. Running indefinitely is not a
proof of convergence, particularly in multiplayer poker. Short pilot measurements
validate execution, not poker strength or resolution of all prior approximation
failures. The old audits remain relevant to evaluating future policies.

## Resources and lifecycle

The explicit data roots are `/var/lib/gto/{hu,3max}/selfplay-v1`. A contract file
rejects incompatible learning configurations. Initialization rejects foreign data.
One CPU worker and one PyTorch thread run per track; existing systemd memory guards
remain active. An interrupted iteration preserves the last complete iteration.

Each advantage/strategy reservoir retains at most 50,000 samples with a separate
512 MiB accounted-byte guard. A guard violation is an explicit error, never silent
sample truncation. Two numbered checkpoints, one stable checkpoint, two local
exports, and bounded compressed diagnostic journals are retained. The first export
occurs after approximately 60 seconds (between complete iterations); later exports
are hourly. Publication retains two selected releases plus a two-hour grace period.
The pinned evaluation reference is retained independently.

Supabase receives compact `selfplay/{track}/status.json`, ONNX policy releases and
bounded evaluation history. It never receives replay or PyTorch checkpoints.
The Training page displays live iteration counters and independent policy
performance against a pinned initial reference and a fixed opponent pool. Those
BB/100 measurements are not full-game exploitability or a GTO certificate.

## Validation before activation

- Fourteen focused initialization, continuous resume, publication, retention and
  deployment tests passed. A regression test advances beyond the configured finite
  batch count and resumes the complete checkpoint.
- Production-size local pilot: three iterations per track with 64 traversals per
  player. HU took 10.1, 14.4 and 17.5 seconds; 3-max took 15.9, 9.3 and 6.4 seconds.
  All six 64-sample advantage-head checks had zero fully inactive heads and nonzero
  encoder gradients. These are small diagnostics, not exhaustive health guarantees.
- Checkpoint reload and ONNX export are verified by the pilot; peak local process
  RSS after HU was approximately 1.06 GB. Filled reservoirs and Linux cgroup memory
  require separate observation after activation.
- The production frontend build passed.

Sources: [Deep CFR](https://proceedings.mlr.press/v97/brown19b.html) and
[Pluribus](https://noambrown.github.io/papers/19-Science-Superhuman.pdf).
