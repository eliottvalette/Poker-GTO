# Continuous training and policy delivery

Status: implementation operational locally and against real Supabase Storage.
VPS installation is blocked on SSH availability; the updated Vercel configuration
must reach the linked Git branch before the public website uses it.

| Requirement | Status | Evidence |
|---|---|---|
| Public object storage, private publication credentials | DONE | Dedicated poker-policies bucket; real public reads succeed; browser-key upload denied |
| Hourly exports without checkpoint reload | DONE | training/vps.py; VPS and publication regression tests |
| Independent retryable uploader and bounded remote retention | DONE locally | gto-publish service/timer; interrupted-upload, retry and retention tests |
| UI/worker refresh, hash validation, bounded ONNX session lifetime | DONE | policy-catalog.ts, policy-bank.ts; browser tests and real Chrome smoke |
| Automatic tested code deployment and compatible resume | IMPLEMENTED, VPS validation pending | GitHub Actions, install.sh, check_resume.py; shell syntax and local resume tests |
| Real export/upload/download pipeline | DONE | HU 109 / 3-max 117 uploaded; public downloads match local artifacts |
| Vercel automatic configuration | PREPARED | ui/vercel.json supplies public build settings; existing Git integration retained |
| Real VPS memory validation and service installation | BLOCKED | SSH blocked; one-worker default retained |

## Data flow

1. One dedicated trainer per VPS resumes its full local checkpoint. It exports a
   validated ONNX/JSON bundle approximately hourly, after a whole iteration. This
   does not reload the replay. Local retention remains bounded.
2. A separate systemd timer checks the completed export every minute. It uploads
   only ONNX and JSON, never replay, private credentials or `.pt` checkpoints.
   A failed upload leaves the prior pointer intact and retries on the next tick.
   The trainer continues independently of network availability.
3. Immutable files live at `<track>/releases/<content-hash>/average_<track>.*`.
   Each track has its own `<track>/current.json`, committed only after both objects
   have been publicly downloaded and compared to their source bytes. HU and 3-max
   cannot overwrite each other's pointers. Exactly one publisher per track is
   required; concurrent independent writers to the same track are unsupported.
4. The UI polls both small pointers every 60 seconds and when focused. Only changed
   tracks are downloaded. The existing full manifest, player-count and SHA-256
   validation runs before activation. A failed refresh displays a warning and
   preserves last validated UI models. The worker checks at request boundaries;
   refresh failure returns an explicit worker error rather than inventing ranges.
5. Shared query handles select the current model at each query boundary. In-flight
   queries finish against their original model; retired ONNX sessions are released
   once those queries finish. Existing games are not reset. Models are not frozen
   for an entire hand; a later observed action may use a newer behavior model.

Publication is a compatibility/inference gate, not evidence of improved poker
strength. The currently published policies remain experimental.

## Supabase project and measured remote checks

Project: `https://vcyctuntcwrrcizvadwh.supabase.co`, bucket `poker-policies`.
Supabase Storage is object storage backed by metadata in Postgres; no application
SQL table is needed for these binary artifacts and JSON pointers.

Initial remote pointers:
- HU iteration 109, release `d0b6c23c15f2081603ab0b762dcd1f5fd4d289ca147dadef3171ba4e9020ddc8`.
- 3-max iteration 117, release `8a2881cdb0ab51b279bea05852b028454c40a7e0cce510f653620e126c754a37`.

These are the already activated local policies, not diagnostic benchmark models.
All four uploaded files passed byte-for-byte public read-back. A request using the
browser publishable key was denied when attempting an upload. No anonymous write
policy was installed. The secret is in ignored `deploy/vps/.env.local` (0600), not
in the UI environment. GitHub's encrypted `SUPABASE_SECRET_KEY` is configured.

Retention keeps the current plus 23 prior releases per track and a 24-hour deletion
grace period. Old orphaned uploads are collected too. Cleanup uses Storage APIs;
it never deletes rows directly from Postgres. CDN-cached copies may outlive origin
retention. Browser tabs fetch the current pointer before downloading immutable files.
Network timeouts are bounded; the publisher service has a 180-second deadline and
256 MiB memory cap. Logs surface failures. A successful pointer commit is independent
of cleanup: a cleanup error is retried and does not undo the selected model.

## Reproduce

Seed or recheck existing policies without training:

```sh
set -a
source deploy/vps/.env.local
set +a
PYTHONPATH=. .venv/bin/python examples/publish_live_policies.py
```

Local regression and browser checks:

```sh
PYTHONPATH=.:tests .venv/bin/python -m unittest test_publication test_vps_training
node tests/run_browser_tests.mjs
cd ui
npm run build
npm run lint
```

Real Chrome smoke test: serve the static build on localhost:3118 and start a dedicated
Chrome with remote debugging on localhost:9335. Run
`node tests/live_pipeline_browser_smoke.mjs` from the repository root. The test
verifies actual Supabase ONNX downloads/inference, an emitted worker, no local policy
fallback, no repeated ONNX download on an unchanged pointer, and recovery from a
simulated pointer-fetch failure. The expected versions are read from the public remote pointers at test start.
Dedicated HU and 3-max worker requests exercise actual published-policy range
reactions, not merely worker creation.

## GitHub Actions and VPS setup

The repository already has a Vercel Git deployment integration for
`https://gto-bot.vercel.app`. `ui/vercel.json` declares only public build variables.
The Vercel project root must remain `ui`. No Vercel secret/API token is required for
these public settings. New model publication requires no Vercel deployment; code
changes follow the existing Git deployment integration.

`.github/workflows/deploy-training.yml` runs Python tests, browser tests, lint and
static build before creating a source archive of the exact tested Git SHA. This is
the automatic code update mechanism; no uncontrolled `git pull` runs inside a live
training directory. Deployments are serialized and do not cancel an in-progress
checkpoint save.

`GTO_DEPLOY_ENABLED=false` is configured on GitHub while the VPS are unavailable.
Before setting it to `true`, configure these repository/environment secrets:
- `VPS_SSH_KEY`: SSH private deployment key.
- `VPS_SSH_PASSPHRASE`: its passphrase, if encrypted.
- `VPS_KNOWN_HOSTS`: verified host keys for both VPS, obtained through a trusted
  channel. Host checking is mandatory; CI does not trust opportunistic keyscan.
- `SUPABASE_SECRET_KEY`: already configured; copied to root-only
  `/etc/gto/publication.env` during deployment.

HU target: `46.224.1.152`; 3-max target: `46.62.239.78`. SSH uses root for installing
services and dependencies, then training/publication run as the unprivileged `gto`
user. Install prerequisites are standard Ubuntu Python/venv and CA certificates.
Torch is installed from the official CPU wheel index; other dependencies are pinned
in `deploy/vps/requirements.txt`.

One-time checkpoint preparation and transfer follow `docs/VPS_TRAINING.md`.
Upload the locally compacted checkpoint to `/var/lib/gto/<track>/checkpoint.pt`.
No original legacy checkpoint should be loaded directly on a 4 GiB host.

On deployment: stop publication and request a clean SIGINT checkpoint from training;
reject an unclean stop; install the new isolated environment; load and validate the
checkpoint with the new code; atomically select the release; start training and
wait for its READY notification after checkpoint loading. Failure restores the
previous code and restarts it when it was previously active. Checkpoints are not
rewritten by the deployment compatibility check. Current and previous code releases
are kept separately from bounded training data. Unknown schema changes fail rather
than reset training. A rollback does not reverse future incompatible checkpoint
migrations; those still require explicit engineering work.

Useful operational checks after installation:

```sh
systemctl status gto-training@hu gto-publish@hu.timer
journalctl --namespace=gto -u gto-training@hu -u gto-publish@hu -n 100
```

No VPS install, continuous run, or forced Git push was performed during local setup.

## Validation results

- Eight Python publication/VPS tests passed (14.562 seconds).
- 152 browser regression tests passed.
- Static build passed with the Supabase configuration; lint/type checking included.
- Real Storage provisioning, both model uploads and public read-back passed.
- Anonymous browser-key upload was explicitly denied.
- The first additional worker smoke failed because the test forced module mode on
  a classic worker emitted by Turbopack. The test now retains the emitted worker
  options, matching the application's actual runtime.
- The corrected real Chrome smoke passed: public HU/3-max models loaded from
  Storage, both dedicated analysis workers responded successfully, unchanged
  refreshes avoided model downloads, and a simulated pointer failure surfaced
  explicitly before recovery. No browser errors were recorded.
