# OVH and Supabase recovery

Read-only inspection on 2026-10-22 found both solver workers at 32,000 visits,
their complete finite allocation. They were waiting for an explicit extension,
not executing the legacy trainer. OVH and the local repository both used revision
`c61bc6f40c6dfae5a7ad92ae0cc79fe441e3171e`. The persisted state incorrectly still
reported `running`; the local correction persists `allocation_complete`.

Publication and evaluation failed because the old Supabase hostname no longer
resolved. The new project configured in `ui/.env.local` responds, but public
Storage reads report `Bucket not found` for `poker-policies`.

## Configuration ownership

- `ui/.env.local`: ignored local browser configuration, public credentials only.
- `ui/vercel.json`: public deployment configuration. The GitHub deployment now
  derives the server project URL and bucket from this file instead of embedding
  a second project URL in the workflow.
- `deploy/vps/.env.local`: ignored server publication credentials. Set the new
  `SUPABASE_URL`, `SUPABASE_SECRET_KEY`, and `SUPABASE_POLICY_BUCKET` here.
- GitHub `SUPABASE_SECRET_KEY`: must match the new project before deployment.
- OVH `/etc/gto/publication.env`: root-only credentials consumed by systemd.

## Pending fresh-start procedure

The selected recovery is a fresh HU and 3-max run. No remote data has been deleted,
no worker restarted, and no new run launched during this inspection.

1. Validate the new server credential against its explicitly configured project.
   Provision the public bucket using `SupabaseStorage.provision`, verify public
   reads, and verify that browser credentials cannot upload.
2. Stop both publication timers, evaluation timer, their services, and both
   training workers. Require clean worker shutdown before replacing state.
3. Reset the exact active solver roots `/var/lib/gto/hu/solver-v1` and
   `/var/lib/gto/3max/solver-v1`. These contain tables, labels, references, student
   optimizer state, and the progress cursor. Do not import legacy Deep CFR
   checkpoints or old remote history into the new run.
4. Install the tested source and matching publication environment. The deployment
   preflight initializes fresh solver contracts without requiring legacy `.pt`
   files. Verify both workers progress from fresh cursors before publication.
5. Publish new status and reference histories; verify both through public reads.
   Rebuild the static UI against the new public configuration. New solver students
   remain unaccepted; missing legacy ONNX policies must not be fabricated.

The allocation is still finite: 2,000 roots and at most 32,000 visits per track.
Restarting repeats that allocation; it does not create an unlimited training loop.
Current local storage is capped at 1 GiB per solver track. Publication overwrites
four scoped tables plus compact status and retains at most 720 hourly history
points per track. Each table is capped at 8 MiB. No checkpoint or replay is uploaded.

Historical Deep CFR directories and benchmark artifacts remain separate from the
active solver roots. Their removal is not needed for correctness: the VPS had
55 GiB free at inspection. The size or cause of the deleted Supabase project's
previous storage usage could not be verified.

Local validation: five focused deployment/status tests passed. Remote recovery
remains pending the replacement server credential.
