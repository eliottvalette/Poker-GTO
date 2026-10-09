# Bounded VPS training

Status: local implementation and bounded validation complete; deployment now targets
one OVH VPS (51.81.202.115, Ubuntu 24.04, 4 vCPUs, 8 GB RAM, 75 GB disk) for both
tracks. SSH/sudo and Linux memory validation are complete; both services are running. Historical CX23/macOS measurements below
are retained as evidence, not measurements of the new server. Replay capacities
and learning settings are preserved.

| Requirement | Status | Owner / next action |
|---|---|---|
| One worker / one PyTorch thread; compare two workers | DONE locally | training/vps.py; ten isolated fixed-checkpoint trials |
| Bound in-memory diagnostics and disk retention | DONE | training/runtime_storage.py, training/runner.py; retention/resume tests |
| Reduce checkpoint serialization peak | DONE | training/checkpoint.py; streaming disk envelope, legacy reader retained |
| Measure load / iteration / save memory including workers | DONE locally | training/vps_benchmark.py; measurements below |
| Export from live models on an hourly cadence | DONE | training/vps.py; shortened interval integration test and real ONNX exports |
| Deterministic resume / unchanged replay semantics | DONE | tests/test_vps_training.py; exact worker-comparison model hashes |
| Actual Linux VPS validation | DONE on OVH | Four cold worker trials and joint production lifecycle; results below |

No long-running training or deployment is started by diagnostics. Existing checkpoints
are immutable inputs. Diagnostics use isolated output directories. SSH probes on
46.224.1.152 and 46.62.239.78 timed out, including outside the local sandbox.

## Runtime policy and entrypoints

`training.vps.run_continuous` loads one dedicated checkpoint once, then repeatedly
runs up to 3,600 seconds of whole iterations and exports the in-memory average
model. It never reduces replay capacities or changes learning rates, epochs,
collectors or sampling distributions. Worker count is explicitly one by default,
with one PyTorch thread. Two workers remain an explicit measured alternative.

Operational retention (independent of learning configuration):
- One recent metrics record per track in RAM and the portable checkpoint.
- Per-iteration detailed metrics in compressed files: at most 256 records and 256 MiB.
- Three completed numbered checkpoints, plus the current session checkpoint.
- Two validated export bundles with an atomically replaced `exports/current.json`.
- The service template uses a dedicated journal namespace capped at 128 MiB persistent,
  32 MiB volatile and seven days. Install `journald-gto.conf` as
  `/etc/systemd/journald@gto.conf` before starting the service.

Root coverage is collected per iteration in bounded mode. Full detailed lifetime
history is deliberately outside the portable checkpoint; retained journal records
have an explicit count/byte quota. Original source checkpoints remain untouched.
Cumulative card-stratification state and replay RNGs are preserved. Metadata records
operational configuration/relocation and the first iteration of retained epsilon
provenance. This is not a change to the CFR objective.

Checkpoint writes now use a disk-backed checksummed envelope, without duplicating
the entire serialized payload in RAM. Old envelopes remain readable. Serialization
needs transient disk space for the payload and envelope as well as the existing
checkpoint. `resume.json` selects the last successful save, including periodic
checkpoints, so a restart need not discard a whole hourly session.

Exports validate ONNX CPU parity before selecting the completed immutable bundle.
An export failure leaves the previous pointer valid and surfaces an error. The
continuous process saves its completed iteration before exporting. Remote publication is handled independently by `training/publication.py` and its
systemd timer; see `docs/LIVE_PIPELINE.md`. Network failures do not stop training.

### One-time import on a machine with sufficient RAM

Do not perform the initial legacy import on a 4 GiB VPS: the measured HU peak exceeds
4 GB. Prepare an unused directory locally:

```python
from pathlib import Path
from training.vps import prepare_vps_checkpoint

prepare_vps_checkpoint('hu', Path('runs/hu-batched/checkpoint.pt'), Path('runs/vps-import/hu'))
prepare_vps_checkpoint('3max', Path('runs/3max-batched/checkpoint.pt'), Path('runs/vps-import/3max'))
```

Transfer each resulting `checkpoint.pt` to the matching VPS's data directory.
The server must use the new checkpoint reader. Preserve the original source as the
historical archive; do not upload all diagnostic benchmark folders to a 40 GB VPS.

### Continuous training (explicit launch only)

```python
from pathlib import Path
from training.vps import run_continuous

if __name__ == '__main__':
    run_continuous('hu', Path('/var/lib/gto/hu'), workers=1)
```

`deploy/vps/train.py` reads `GTO_TRACK`, `GTO_DATA_DIR`, optional
`GTO_SOURCE_CHECKPOINT`, and `GTO_WORKERS`. The systemd template is installed on OVH for both tracks. It expects `/opt/gto/current`, `/opt/gto/venv`, and an unprivileged `gto`
user. SIGINT requests a checkpoint of the last completed iteration. Normal service
logs and errors remain visible. The provisional 2,800 MiB soft / 3,200 MiB hard
service memory bounds must be checked against Linux measurements before 24/7 use.
The service stages temporary checkpoint reads under its data directory (`TMPDIR`),
which must be on the persistent disk rather than a RAM-backed filesystem.

The Python environment must provide the project's CPU dependencies, including
PyTorch, NumPy, ONNX and ONNX Runtime. `psutil` is required only for profiling; the original
local trials did not install remote environments. The OVH deployment below does.

### Reproduce bounded benchmarks

```python
from pathlib import Path
from training.vps_benchmark import compare_workers

if __name__ == '__main__':
    compare_workers('hu', Path('/var/lib/gto/hu/checkpoint.pt'), Path('/var/lib/gto/bench-hu'))
```

Each configuration starts in a new spawned process, runs one full iteration and
measures import, iteration, save and live export. Samples aggregate parent/worker
RSS every 20 ms; shared pages can be counted more than once. Parent high-water RSS
is reported separately. Output directories must be unused. Production sources are
read-only. The comparison checks exact model-weight equality.

## Local measurements (2026-10-09)

macOS arm64, one PyTorch thread. Ten fresh-process trials: one iteration per
track/worker setting from the legacy checkpoint, then one from the compact checkpoint.
Two additional 3-max trials include both A and B fitting at iteration 120.
Total: 4,352 traversals. No production source checkpoint was modified. Each trial
has a 300-second guard; these are single-iteration diagnostics, not latency confidence intervals.

Cold restarts from compact snapshots:

| Track | Workers | Load s / peak GiB | Iteration s / peak GiB | Save s / peak GiB | Export s |
|---|---:|---:|---:|---:|---:|
| hu | 1 | 12.40 / 1.50 | 36.97 / 1.05 | 3.25 / 1.97 | 0.34 |
| hu | 2 | 13.33 / 1.50 | 29.11 / 1.53 | 3.40 / 1.94 | 0.28 |
| 3max | 1 | 14.31 / 1.48 | 26.50 / 1.09 | 3.10 / 2.04 | 0.28 |
| 3max | 2 | 13.59 / 1.50 | 18.06 / 1.52 | 3.05 / 2.10 | 0.26 |

RSS includes children; shared pages can be double-counted. 20 ms sampling may miss
brief peaks; the JSON also records the parent lifetime high-water mark. Cold runs
include worker startup/shutdown. Linux cgroup memory (including charged file cache)
must be measured separately before accepting the service memory ceiling.

Compact checkpoints are approximately **273.5 MiB HU / 271.4 MiB 3-max**, versus
approximately 1.3 GB legacy files. Replay capacity remains **50,000 advantage +
50,000 strategy samples** per track; the size reduction removes cumulative diagnostic
history, not replay examples. Legacy imports peaked at **4.08 GiB HU / 4.69 GiB 3-max**
in these trials, so perform that conversion locally before transferring a checkpoint.

Two workers reduced this cold iteration latency by **21% HU / 32% 3-max**. Both
networks have exactly matching weight hashes between worker counts in all four paired
comparisons. Keep **one worker as the conservative VPS default** until the CX23 trial
confirms RAM and CPU behavior. These numbers do not predict 24/7 memory stability.

Raw reports and paired comparisons: `runs/vps-benchmark/*/report.json` and
`runs/vps-benchmark/*-comparison.json`. These are local ignored diagnostic artifacts;
the numerical summary above persists in this document. Benchmarks produce advanced
diagnostic checkpoints; they are not automatically selected as production policies.

Validation: **35 tests passed in 60.616 seconds**. Tests cover legacy/checksummed reads, corruption rejection, bounded metrics/retention,
deterministic replay/model resume, live export without checkpoint reload, export failure
atomicity, and continuous-loop restart. Reproduce with:

```sh
PYTHONPATH=.:tests .venv/bin/python -m unittest test_vps_training test_training_runner test_workflow test_training_performance test_outcome_pipeline
```

The first test invocation using package-qualified names failed to import existing shared
test helpers; the reproduction command explicitly includes `tests` on `PYTHONPATH`.

The original deployment backlog (SSH access, dependencies, compact checkpoint transfer,
Linux cgroup measurements and continuous service start) is now completed on OVH,
as recorded below. Hourly exports are picked up by the independent publisher; Supabase delivery and
browser refresh are now implemented and tested as recorded in `docs/LIVE_PIPELINE.md`.

### Additional 3-max A+B fitting check

Iteration 120 trains both networks, unlike iteration 119 above.

| Workers | Load s / GiB | Iteration s / GiB | Save s / GiB | Export s |
|---|---:|---:|---:|---:|
| 1 | 13.02 / 1.47 | 32.46 / 1.86 | 3.74 / 1.75 | 0.33 |
| 2 | 12.72 / 1.50 | 22.97 / 2.20 | 3.10 / 1.89 | 0.28 |

Exact A/B weight hashes match again. Peak sampled aggregate RSS is **2.20 GiB**
with two workers; retain one worker pending actual VPS measurements. Raw evidence:
`runs/vps-benchmark/fit-b/3max-comparison.json`. These trials do not establish
long-duration leak freedom or a hard worst-case bound on future game trees.

## OVH deployment validation (2026-10-09)

Target: `51.81.202.115`, Ubuntu 24.04 x86_64, 4 exposed Haswell-class vCPUs,
7,751 MiB reported RAM, 72 GiB root filesystem. Passwordless SSH and sudo are
verified. A restricted, dedicated GitHub deployment key is installed; the private
key is stored only in the GitHub Actions secret. Automatic deployment is gated until the updated single-host workflow reaches main
and the joint production lifecycle passes. GitHub Actions retains delivery evidence.

Latest local training seeds were compacted without changing replay capacities:
HU approximately 298.9 MiB and 3-max 289.8 MiB, under `runs/vps-import/`.
Original checkpoints remain unchanged. Each cold benchmark permits one iteration,
300 seconds, one PyTorch thread, and a 3,200 MiB cgroup memory ceiling. Diagnostic
outputs are separate from production checkpoints.

The first Linux load exposed a deployment defect: `source_metadata()` required
`.git` although releases are delivered as archives. `deployment-source.json` now
records source identity and validates the packaged source digest without Git.
Missing or modified provenance fails explicitly. GitHub packaging includes this
manifest; the initial manual deployment is a hashed working-tree snapshot with
its base commit and per-file hashes, not represented as an existing Git commit.
The failed first trial did not modify the production seed.

Validation after this repair: 13 publication/storage/deployment/provenance tests
passed in 17.318 seconds; 21 runner/workflow tests passed in 30.532 seconds.
Workflow YAML and embedded shell syntax were checked using existing tooling.

### OVH cold worker comparison

Each row runs one full iteration from the same seed per track (HU 188, 3-max 117).
This measures one workload, not a throughput confidence interval or poker quality.

| Track | Workers | Load s | Iteration s | Save s | Export s | Peak summed RSS GiB |
|---|---:|---:|---:|---:|---:|---:|
| 3max | 1 | 27.31 | 92.90 | 7.32 | 0.74 | 1.20 |
| 3max | 2 | 25.91 | 59.62 | 7.06 | 0.73 | 1.50 |
| hu | 1 | 27.26 | 166.94 | 7.58 | 0.53 | 1.23 |
| hu | 2 | 27.23 | 104.97 | 7.40 | 0.52 | 1.56 |

A/B model hashes match exactly between worker counts for both tracks. Two workers
reduce iteration latency by 37.1% HU and 35.8% 3-max in these trials. Both
server overrides select two workers and one PyTorch thread. The service
memory ceiling remains 3,200 MiB per track. Cgroup peaks reported by systemd
were approximately 1.4/2.0 GiB HU and 1.3/1.9 GiB 3-max for one/two workers.
Sampled RSS can miss brief allocations: parent lifetime RSS high-water marks
reached approximately 1.8 GiB during saving. Shared pages can inflate summed RSS.

Reports: `runs/vps-ovh-validation/worker-comparison.json` locally and
`/var/lib/gto/benchmarks/*-provenance-fixed/report.json` on the VPS.

The first service installation also exposed a permissions defect: the shared
code parent was root-only (0750), preventing the unprivileged trainer from
executing Python. Installation now creates `/opt/gto/releases` as 0755 and
keeps `/etc/gto` restricted separately. No checkpoint was altered by this failure.

### Joint production lifecycle

Both services completed an iteration simultaneously with two workers each and one
PyTorch thread: HU 188→189 in 110.36 seconds (118,355 nodes), 3-max 117→118 in
68.09 seconds (85,517 nodes). The sampled combined current cgroup memory immediately
before stopping was approximately 2.57 GiB; each service retains its 3,200 MiB hard
ceiling. Four exposed vCPUs support this bounded workload without changing replay
or fit budgets. This is not a multi-day leak or throughput stability claim.

A clean systemd SIGINT stop saved and exported both policies from memory. Both
publisher services succeeded and public Storage pointers were verified:

- HU 189: `3db279d439ad3c3f7560b9df8fabbe68809ef7fec826f1f637ca83b493ed465a`.
- 3-max 118: `ee375600b0e969ea033eb4417e2b931730c17ae1d89e3eee1842225c2b0fb707`.

Both training services restarted successfully, and both publication timers are
active. Exports recur approximately hourly at completed-iteration boundaries;
publication checks run every minute. The initial source is the documented working-
tree snapshot `d68703ee546f0bb17983626481e646c17d04c1de`; subsequent GitHub deployments
use the tested commit SHA and validated archive provenance.

Operational checks:

```sh
ssh ubuntu@51.81.202.115
sudo systemctl status gto-training@hu gto-training@3max
sudo journalctl --namespace=gto -u gto-training@hu -u gto-training@3max -f
```
