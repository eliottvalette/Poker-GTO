# OVH and Supabase recovery

The subsequent continuous self-play migration is documented in
[CONTINUOUS_SELFPLAY.md](CONTINUOUS_SELFPLAY.md). The finite solver recovery below
is historical; it does not describe the new self-play service.

Read-only inspection on 2026-10-22 found both solver workers at 32,000 visits,
their complete finite allocation. They were waiting for an explicit extension,
not executing the legacy trainer. OVH and the local repository both used revision
`c61bc6f40c6dfae5a7ad92ae0cc79fe441e3171e`. The persisted state incorrectly still
reported `running`; the local correction persists `allocation_complete`.

Publication and evaluation failed because the old Supabase hostname no longer
resolved. At initial inspection the replacement project responded, but its
`poker-policies` bucket did not exist. Recovery is now complete, as verified below.

## Configuration ownership

- `ui/.env.local`: ignored local browser configuration, public credentials only.
- `ui/vercel.json`: public deployment configuration. The GitHub deployment now
  derives the server project URL and bucket from this file instead of embedding
  a second project URL in the workflow.
- `deploy/vps/.env.local`: ignored server publication credentials. Set the new
  `SUPABASE_URL`, `SUPABASE_SECRET_KEY`, and `SUPABASE_POLICY_BUCKET` here.
- GitHub `SUPABASE_SECRET_KEY`: must match the new project before deployment.
- OVH `/etc/gto/publication.env`: root-only credentials consumed by systemd.

## Fresh-start procedure

Recovery started fresh HU and 3-max runs. Only the two active `solver-v1` roots
were reset; legacy training directories were not imported or removed.

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

## Completed verification (2026-10-22)

- Revision `377ca1d6502753fbb8d379f437aba82bde8086b1` passed the full GitHub
  validation and deployed through run `39826153530`. The duplicate queued run
  `39826155436` was cancelled before deployment.
- The new public bucket was provisioned and browser-key uploads were denied.
  GitHub's publication secret and OVH's root-only publication environment were
  updated. The already-started workflow retained its original secret snapshot;
  after it completed, the current credential was installed again on OVH.
- Both workers stopped cleanly before their active solver roots were replaced
  with validated empty contracts. The new public solver namespaces were empty
  before workers and timers resumed.
- Both publishers and the evaluator returned `Result=success`. Public reads
  showed HU status at 19 visits and its first history point at 20 visits;
  3-max status was at 9 visits and its first history point at 10 visits.
  Each history contained exactly one fresh point, dated 2026-10-22 around 17:57 UTC.
- Storage contained 14 objects totaling 946,074 bytes at that inspection.
- The production site returned successfully and its served application chunks
  embedded the new Supabase origin, with no old origin found. Vercel reported
  successful production deployment for the same revision.
- Six focused local tests passed. The local production build also passed after
  rerunning outside the sandbox; the first attempt was blocked by Turbopack's
  local port binding restriction. CI build, lint, and browser tests passed.

The reset does not improve or extend the solver curriculum. The finite allocation
and unaccepted-student limitations above still apply. Legacy ONNX policy objects
were not copied into the replacement bucket.
