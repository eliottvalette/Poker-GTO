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

## Production verification (2026-10-22)

Revision `bb4f7874e2c228eaf2c1f705d4f74f0f668ee93e` passed all 62 Python
CI tests, the frontend build/lint and 162 browser tests. GitHub run
`39829215046` deployed both self-play workers. Both remained active with zero
restarts at inspection. HU passed iteration 11 and 3-max passed iteration 24.
Observed peak cgroup memory was approximately 2.58 GB for HU and 1.66 GB for 3-max;
these measurements do not establish a long-run maximum.

The first public ONNX exports were HU iteration 2 and 3-max iteration 3.
Both publishers succeeded after the validated replacement credential was installed
on OVH and synchronized to both the GitHub repository and `training` environment
secret scopes. Both independent evaluations completed 128 deal groups successfully
at 18:28:47 UTC (HU) and 18:29:17 UTC (3-max). These first observations establish
baselines; improvement requires subsequent model evaluations.

## Opening-range correction (2026-10-22)

The first 3-max export was strategically unsuitable: at the 25 BB opening it
jammed AA at 84.2% and 72o at 84.4%. Native ONNX inference reproduced the browser,
so this was not a rendering error. A frozen-replay audit found only two retained
AA openings and zero KK openings after three iterations. First openings carried
0.0794% of the shared strategy objective in 3-max (0.7298% in HU). Re-fitting only
those sparse opening records still reproduced the weak targets; simply increasing
fit epochs was rejected as a repair.

The corrected learning contract uses `selfplay-v2` data roots, fresh networks and
replay, stratified recorded-player cards, and a 50% opening replay reservation.
Inverse inclusion correction still applies within replay. Fitting explicitly
normalizes iteration/reach/inclusion mass separately for first openings and each
remaining street, then samples from that normalized objective. With all five
strata present, first openings carry 20% of the approximation objective. Weights
inside each stratum retain their original ratios. This deliberately changes the
function-approximation objective; it must not be described as an unchanged Deep
CFR stream objective or an established equilibrium guarantee.

A four-iteration pilot measured 20% opening mass for both networks in both tracks.
HU took 4.0–10.6 seconds per iteration, 3-max 5.5–9.4 seconds. Both candidates were
rejected: neither covered all 169 starting-hand classes with four retained
records each, and the 3-max maximum action span remained only 12.0 percentage
points. The pilot validates the repair to coverage allocation, not poker strength.

Exports now carry a sealed opening-health report. The self-play publisher requires
coverage of all 169 classes in both replays, at least four records per class,
and a maximum action-probability span of at least 15 percentage points across
1,326 holdings. Candidates meeting these preliminary checks also receive two
128-world independent-seed fixed-policy continuation probes (AA and 72o). A
conservative action-deviation margin above 0.5 BB rejects the candidate. These are
minimum rejection checks, not a GTO certificate or a formal sequential test.
Candidates are exported every five minutes for screening; rejected candidates do
not replace the public pointer and are visible in training status.

Overview independently withholds an existing grid when its class probabilities
vary by less than 15 percentage points, and labels all learned ranges experimental.
Thus old rejected exports cannot continue to appear as usable GTO ranges while
the corrected run gathers data. No prescribed AA/72o action frequencies or
hand-written opening strategy were inserted into the learned output.
