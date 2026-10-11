# Published preflop policy audit — 2026-10-10

## Artifacts and scope

- HU iteration 690: `fd8368906c0d09a81615252d623db000cc4dbfb98bb2ea43d591b22b70b5dc03`.
- 3-max iteration 851: `362705048f3b57c856dbc2fcddc2dfe4021c2bbf2e7dce7cd7eae2cd29e7c9ba`.
- Both are public ONNX average policies. Manifest hashes were checked by `OnnxPolicy`.
- Overview directly queries these models. It does not run local CFR.
- Overview defaults are HU SB at 37.5 BB and 3-max BTN at 25 BB, balanced stacks, unopened pot.
- Server SSH authentication failed. Current advantage weights, replay and server fitting metrics were not inspected. Local historical checkpoints cannot establish their current state.

## Exhaustive observation probes

`training.preflop_diagnostics.opening_matrix` queried all 1,326 combinations and their reversed card order at each root. Reversal error was exactly zero on both exports.

| Hand | HU fold | HU jam | 3-max fold | 3-max jam |
| --- | ---: | ---: | ---: | ---: |
| AA | 4.42% | 1.17% | 1.61% | 24.34% |
| 22 | 0.39% | 2.55% | 16.32% | 32.99% |
| 72o | 2.15% | 1.09% | 40.66% | 22.51% |

The 3-max 22 distribution reproduces the displayed values. Display aggregation is not the explanation for that mixture. Mixing many actions alone is not evidence of incorrect play; their conditional values must be compared.

## Fixed-policy continuation experiments

Each action was forced at the opening decision. Subsequently **every seat followed the published policy**, with uniformly distributed compatible opponent hands and conditional future boards. All actions shared private deals, future deck order and per-seat random streams. Terminal utilities used the canonical engine. Results are physical chip deltas divided by the current big blind, **BB per hand, not BB/100**.

Discovery: 256 deals per holding for AA, 22 and 72o, both tracks, seed 19001; seven root actions; 10,752 branches. Including the exhaustive observation probes, this took 7.53 seconds locally.

Confirmation: a different seed, 29001; 1,024 deals per holding for AA and 72o, both tracks; 28,672 branches; 17.51 seconds locally. Hands used were `(48,49)` for AA and `(20,1)` for 72o. These are specific combinations; the observation table above averages classes.

| Confirmation measurement | HU | 3-max |
| --- | ---: | ---: |
| 72o published mixture EV | -4.737 | -2.202 |
| 72o fold EV | -0.500 | 0.000 |
| Fold improvement over 72o mixture | +4.237 | +2.202 |
| Approximate paired 95% interval | [3.131, 5.343] | [1.764, 2.640] |
| AA min-raise improvement over mixture | +0.758 | +1.631 |
| Approximate paired 95% interval | [-0.072, 1.588] | [1.087, 2.175] |

These are bounded one-decision deviations under fixed continuation assumptions. They are not equilibrium preflop solutions, cumulative CFR regret targets, or full-game exploitability. Intervals are descriptive normal approximations. Candidate actions in the confirmation were informed by the discovery experiment, then evaluated with independent random draws. No successful model or configuration promotion follows from this diagnostic.

## Training findings and unresolved cause

The exports have substantial measured decision defects even under their own continuation policies. More training iterations are not evidence that these defects are shrinking. The hourly heterogeneous-opponent evaluation cannot attribute errors to the advantage model, replay targets or average-policy fitting.

Source inspection also confirms that warm-start fits retain network weights but create a new Adam optimizer for each fit. Each fit partitions the current reservoir again; samples held out in one fit can have been trained on earlier. Consequently, its held-out loss is a fit diagnostic, not an independent longitudinal generalization measure. Neither fact alone establishes the cause of the observed strategic errors.

Next required diagnosis once server access is restored: capture one immutable current checkpoint, compare A predictions against independently sampled action advantages at fixed public roots, inspect generated and retained target coverage for those roots, and compare B against the weighted historical strategy targets. Keep direct action-deviation measurements alongside replay fitting loss. Do not change the all-nonpositive regret fallback, hand-code opening actions, or increase epochs based only on these results.

No training parameters, running services, checkpoints or published artifacts were changed in this audit.

### Server access restored

After reloading the SSH identity, read-only access succeeded. Latest metric snapshots were HU 843 and 3-max 1014, later than the exported models evaluated above. They must not be treated as measurements of identical checkpoints. The raw snapshots are retained in `runs/published-preflop-audit/server-metrics.json`.

Both tracks retain 10,000 opening examples in each 50,000-record memory. HU strategy openings include AA=42, KK=71 and AKs=26; 3-max includes AA=47, KK=45 and AKs=36. These counts aggregate different contexts and do not establish sufficient coverage at the fixed Overview root.

HU strategy objective ESS is 30,110, but 3-max strategy objective ESS is only 126.1. The largest 3-max record carries 8.43% of objective weight; the top ten carry 14.13%. Its held-out strategy ESS is 10.50, and river strategy ESS is 8.50. Thus extreme strategy-weight concentration remains an active 3-max problem despite the corrected collector's exact small-tree validation. These metrics do not establish a biased collector or justify clipping weights.

Latest A train/held-out losses are 202.43/434.30 (HU), 126.36/294.57 (3-max); B losses are 0.321/0.657 and 0.224/0.681. These losses mix contexts and are not independent strategic quality measurements. A checkpoint-level target/prediction attribution is still required.

## Reproduction

Local diagnostic inputs and complete class/action summaries are retained under `runs/published-preflop-audit/`: `average_hu.onnx/.json`, `average_3max.onnx/.json`, both publication manifests, `report.json`, and `confirmation.json`.

```python
import json
from pathlib import Path
from ml.onnx_policy import OnnxPolicy
from training.preflop_diagnostics import opening_matrix, paired_opening_values

root = Path("runs/published-preflop-audit")
policy = OnnxPolicy(
    (root / "average_hu.onnx").read_bytes(),
    json.loads((root / "average_hu.json").read_text()),
    2,
)
matrix = opening_matrix(policy, 2, 37.5)
confirmation = paired_opening_values(policy, 2, 37.5, (20, 1), 1024, 29001)
```

Regression checks cover exact-combo multiplicities, uniform action masses, deterministic pairing, impossible input rejection, and analytical opening-fold utilities for both player counts.

## Training display

The page now connects compatible hourly measurements with straight line segments and shows their 95% uncertainty band. Failed measurements break the line. The four KPI cards and both expandable documentation/history sections were removed. A single selected value and a four-opponent by three-stack table show the chosen metric consistently. Baseline self-comparison is labeled Baseline. Actual failure and budget-limit states remain visible.

Validation: static production build, three Python diagnostic tests, real Chrome HU/3-max history rendering, point selection infrastructure, time-window controls, multi-point curve/band, twelve opponent segments, missing history, HTTP failure and recovery. Browser history transport fixtures are synthetic and are not poker quality evidence.
