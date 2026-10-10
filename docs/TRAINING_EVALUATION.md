# Hourly policy evaluation

The `Training` tab displays evaluation of exported average policies, separately
for HU and 3-max. It does not measure the complete hybrid decision engine, certify
GTO, or automatically promote a model on a noisy performance estimate.

## Execution

After the normal GitHub deployment, `gto-evaluate.timer` checks every five minutes
for new published releases. One process evaluates HU then 3-max. Already evaluated
releases are skipped. Models are loaded from the same checksummed ONNX artifacts
as the browser; no training checkpoint, replay, or PyTorch runtime is loaded.

The evaluator has one inference thread, a one-core CPU quota, a 512 MiB memory
limit, and low scheduling priority. Each track has a 120 CPU-second / 480
wall-second budget. The service has an 1100-second backstop. A deployment stops
the evaluator before changing source; a failed deployment restores its timer.

No additional secrets or SQL migration are required. The existing server-only
Supabase credential writes to the existing Storage bucket. The browser only reads
public aggregate results. Committing and pushing the source activates the service
through the existing deployment workflow; local development does not change the
running VPS.

## Method: hourly-paired-policy-v1

- 128 independently seeded groups, never added to training replay.
- Four fixed controlled opponents: uniform, loose/passive, tight/aggressive,
  and push/fold. These are synthetic populations, not calibrated human models.
- Three starting-stack configurations: 6 BB, 25 BB, and asymmetric 35/8 BB
  (35/8/18 BB in 3-max).
- Every candidate and the fixed reference play each seat on matching full deals.
  Private cards are drawn from the full deck, not two-combination priors.
- Separate per-seat action RNG streams; candidate trajectories can diverge, so
  matched seeds reduce variance without promising identical subsequent actions.
- The maximum is 6,144 HU or 9,216 3-max hands including reference hands.
- Outcomes are exact engine settlements divided by the starting big blind.
  Aggregate and per-profile/stack metrics use BB/100.
- Uncertainty is computed over group means, not individual correlated seat
  rotations. Intervals are descriptive normal 95% approximations. Repeated
  hourly use of the same seeds does not increase the independent sample count.
- A small exact four-deal river best response probes one responding seat.
  Both opponents' holdings vary in 3-max. Its gain is measured in hand BB and is
  conditional on this tiny game, not an exploitability certificate.

The first evaluated policy is pinned as an immutable comparison reference per
track and suite. It is not accepted as a strong policy. Changing a suite requires
a version change; the UI never mixes curves with different references or suites.
The original automatic policy publication remains independent of evaluation:
`current.json` still means latest technically validated export, not poker champion.

CPU/wall exhaustion discards an incomplete balanced group, records discarded work,
and labels the result `budget_limited`. Path-cost stopping can bias estimates;
these points are amber and cannot certify improvement. Errors become `failed`
reports rather than synthetic zero rewards. Transport failures fail the service
explicitly and retry on the next timer tick. Stale results remain visibly dated.

## Compact retention

Storage paths:

```
evaluation/hu/index.json
evaluation/hu/history.json.gz
evaluation/3max/index.json
evaluation/3max/history.json.gz
evaluation/<track>/hourly-paired-policy-v1/reference.json
evaluation/<track>/hourly-paired-policy-v1/reference.onnx
evaluation/<track>/hourly-paired-policy-v1/reference.manifest.json
```

Each history retains at most 30 days, 720 evaluations, and 8 MiB uncompressed. Thus both
histories are bounded by 16 MiB in total, plus two small pinned references.
Publication uses one writer per track, an atomic object replacement, and a
read-back check. Unique release identities prevent duplicate retry records.
The normal policy retention stays at 24 releases with its existing grace period.

Detailed paired group outcomes remain only on the VPS under
`/var/lib/gto/evaluation/<track>`, compressed and capped at 48 hours / 32 MiB
per track. They are not uploaded. Retention runs on every evaluator check, even
if there is no new model. After trace deletion, the summary retains moments and
provenance, but does not permit arbitrary retrospective bootstrap analyses.

No permanent per-hand rows, realtime subscriptions, or unbounded JSON append log
are introduced. Dashboard reads the small checksum index every five minutes while
visible; compressed history is only downloaded when its checksum changes. Errors
are displayed and a 404 means the first evaluation has not yet been published.

## Validation and measured cost

Local CPU diagnostic on existing exported policies, each compared with itself:

| Track | Complete paired hands | CPU seconds |
| --- | ---: | ---: |
| HU | 6,144 | 4.29 |
| 3-max | 9,216 | 9.78 |

The combined evaluator process peaked at approximately 57.4 MiB RSS on the local
Mac, with no PyTorch import. This is not an OVH latency or memory measurement.
All paired differences were exactly zero, as required; this experiment measures
reproducibility and overhead, not strategic improvement. The ignored local report
is `runs/hourly-evaluation-validation/benchmark.json`.

Regression coverage includes ONNX/PyTorch parity across real streets, known fold
payoffs, deterministic paired HU/3-max outcomes, budget exhaustion, explicit
policy failure, 30-day/720-record and 48-hour retention, immutable reference
pinning, upload failure/retry, and browser history validation/filtering.

```
PYTHONPATH=.:tests .venv/bin/python -m unittest test_hourly_evaluation test_onnx_export test_vps_deployment
node tests/run_browser_tests.mjs
```

The static UI passed the Chrome smoke test for both tracks, period controls,
compressed history loading, missing data, HTTP failure and recovery. Reproduce
with a separately started static UI and debugging Chrome:

```
POKER_CDP_ORIGIN=http://127.0.0.1:9337 POKER_UI_ORIGIN=http://127.0.0.1:3034 node tests/training_browser_smoke.mjs
```

The browser smoke test explicitly intercepts history responses with synthetic
transport fixtures; it does not write to Supabase or claim live deployment.

On the VPS:

```
sudo systemctl status gto-evaluate.timer gto-evaluate.service
sudo journalctl --namespace=gto -u gto-evaluate.service -n 30 --no-pager
```

## Interpretation and limitations

The dashboard displays dated measurements and uncertainty, not a fabricated
strength rating. It must not label a reference match as "improved" solely because
its interval excludes zero. Significant promotion would need independent
confirmation and a multiple-testing/sequential protocol.

This first hourly suite does not include full tournament outcomes, AIVAT,
independent daily confirmation, advantage-network supervision accuracy, or online
hybrid-search ablations. The opponent pool and full-deal coverage are broader than
the former three-hand sanity check, but remain a controlled benchmark. More
training does not guarantee improvement on it or in general poker play.

Methodological references: [AIVAT](https://poker.cs.ualberta.ca/publications/aaai18-burch-aivat.pdf),
[Local Best Response](https://poker.cs.ualberta.ca/publications/aaai17ws-lisy-lbr.pdf),
[time-uniform confidence sequences](https://arxiv.org/abs/1810.08240).
