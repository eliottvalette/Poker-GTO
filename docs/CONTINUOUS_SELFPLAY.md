# Continuous self-play

The current learning contract is v4, described in the final section below.
Earlier sections record previous contracts and their validation history.

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

### Fitting-budget diagnosis and continuation

After the coverage repair, production v2 reached all 169 classes with at least
43–50 advantage and 44–45 strategy records per class. Nevertheless, exported
maximum action spans remained approximately 7.1% (HU) and 6.1% (3-max), so both
were withheld. This demonstrates that coverage protection alone was insufficient.

An explicit frozen-input study used preflop samples extracted from HU iteration
359 and 3-max iteration 352. Complete inputs, measurements and reproduction code
are retained under `runs/opening-fitting-budget-study/` (ignored diagnostic assets).
The sampled regret targets distinguish strong and weak holdings: mean 3-max fold
regrets were -21.17 for AA and +7.86 for 72o. The approximator failed to reproduce
that distinction at the previous optimization budget.

On identical frozen preflop records, initialization, partition seed and architecture,
increasing fresh-fit updates from 256 to 2,048 reduced held-out-record advantage
MSE from 343.96 to 330.53 in HU and from 317.25 to 304.76 in 3-max. At the fixed
opening, the 3-max fitted regret policy's AA fold probability fell from 42.1% to
0%, while 72o rose from 52.8% to 97.3%. These frequencies are diagnostics of the
fit, not prescribed targets or equilibrium references. The record split shares
public contexts and does not establish independent-root strategic generalization.
The study isolates preflop fitting; end-to-end production still includes all streets.
A logarithmic-input variant was also tested but is not deployed: increasing the
budget alone already repaired the demonstrated distinction without a schema change.

The v3 service raises both fit caps to 2,048 updates and allows up to 32 epochs to
reach those caps. It continues from an explicitly validated v2 checkpoint into
`selfplay-v3`, preserving learned samples, RNG state, weights and iteration count.
A new run ID and recorded migration make the changed fitting protocol explicit.
The migration refuses any unrelated configuration difference. Each service then
fits fresh networks using the larger budget on subsequent self-play iterations.
No fabricated strategy labels are introduced. The public rejection checks remain
in force; a completed migration alone does not authorize publication of a weak model.

### Verified v3 activation

GitHub run `39897847105` validated and deployed revision
`d67bdf2ffd22505b39cd835761da663122765648`. Direct checkpoint inspection confirmed
2,048 completed updates for both networks at HU iteration 370 and 3-max iteration
361. The migration retained source iterations 364 (HU) and 358 (3-max) and recorded
the original run IDs. No replay reset occurred in this fitting-budget migration.

Native fixed-opening checks on these production checkpoints showed zero AA/KK
fold mass and 100% 72o fold mass for the regret-matched advantage policies in both
tracks. These are point-in-time spot diagnostics, not equilibrium certificates.
The average networks still assigned similar probabilities to strong and weak
holdings because their replay includes older weak policies. Their publication
remains gated; they were not relabelled as accepted simply because advantage
fitting improved. Subsequent self-play samples must update that historical average.

Production Vercel chunks were read back and contained all three new controls:
`Experimental preflop policy`, the near-constant-range refusal, and the latest
candidate-withheld status. Thus the old published iteration-2/3 models no longer
produce the misleading range grid shown before the correction.

## Traversal-budget continuation: v4 (2026-10-25)

Both dedicated tracks use 256 traversals per player per iteration: 512 total
for HU and 768 for 3-max. A is fitted every iteration, capped at 1,024 updates.
B is fitted on iterations divisible by three, capped at 1,024 updates; its
existing weights remain in use between fits. The absolute iteration counter
and cadence survive restarts. The 50,000-record reservoirs, shared A/B network
architecture, collector, current-advantage root policy, and hybrid solver are
unchanged. Existing publication rejection checks and hourly evaluations remain
enabled with the same pinned evaluation reference.

`migrate_v4` explicitly continues each stopped `selfplay-v3` run into
`selfplay-v4`. It validates the source checkpoint and rejects configuration changes
outside the traversal count, fit caps, B cadence, and output directory. It
preserves replay, weights, sampler state, RNG states, metrics, and iteration;
the new run identity records the source checkpoint checksum and migration boundary.
The complete destination is published atomically. The original v3 files remain
available. `verify_v4.py` compares fingerprints of every learning-state field
against that source and reloads the complete runner before activation.

Each migration records a fixed-parameter observation horizon of 500 additional
iterations. The continuous services keep running beyond that horizon; it is not
an automatic stop. Only technical incidents justify a parameter change during
this period. No additional ablation campaign is scheduled.

The focused migration, cadence/resume, opening-health, VPS lifecycle, and
publication suite passed all 15 tests. The production activation script uses
the existing CPU environment and service units, serializes migrations under
deployment locks, and restores v3 service selection if activation fails.
Completed HU milestones remain archived under v3; pending milestones follow
the continued v4 run with its explicit new identity.

### Verified v4 activation

OVH activated the immutable source snapshot
`selfplay-v4-e2fcc504325562b3` at 2026-10-25 05:17:48 UTC, based on revision
`d67bdf2ffd22505b39cd835761da663122765648` with the explicit v4 changes.
Its full source digest is
`e2fcc504325562b3c2012c76be1190d920730baf05c7060cde9f56b1e288c741`.

| Track | Preserved iteration | Minimum unchanged-parameter horizon | New run ID |
| --- | ---: | ---: | --- |
| HU | 1,757 | 2,257 | `d787d731c6184850a977f16ff8e1eab2` |
| 3-max | 1,375 | 1,875 | `18e8ef09ec2145d6b31cc44499b90c9a` |

Both production verification reports confirm matching fingerprints for every
learning-state field and successful full-runner reloads. Reports are retained
in each run's `migration-verification.json` and copied to
`runs/selfplay-v4-deployment/activation-verification.json` locally. Both training
services and the existing publication/evaluation timers resumed successfully.
The HU 1,900 milestone now follows v4; saved 1,650 and 1,750 files remain in v3.

The initial activation rolled back safely because the verification helper did
not support sets in serialized sampler state. The corrected helper handles
unordered state deterministically; the complete migration/verification test for
both tracks and an additional fingerprint regression test passed before retry.
The first attempt's files are retained separately; no failed verification was
bypassed and no learning configuration changed during this correction.

The first completed v4 iterations confirmed the requested execution budgets:

| Track / iteration | Total traversals | A updates | B updates | Wall seconds |
| --- | ---: | ---: | ---: | ---: |
| HU 1,758 | 512 | 1,024 | 1,024 | 122.57 |
| HU 1,759 | 512 | 1,024 | 0 | 113.38 |
| HU 1,760 | 512 | 1,024 | 0 | 62.70 |
| HU 1,761 | 512 | 1,024 | 1,024 | 98.51 |
| 3-max 1,376 | 768 | 1,024 | 0 | 464.86 |

The first 3-max iteration spent 408.72 seconds generating traversals and 28.81
seconds fitting. These are deployment observations, not a controlled performance
comparison or evidence of strategic improvement. Both workers completed exports
with zero service restarts. The first HU candidate was withheld by the existing
AA action-deviation publication check; v4 does not override that rejection.
Runtime evidence is retained locally in
`runs/selfplay-v4-deployment/runtime-verification.json`.
