# Timed training scripts and policy publication

Use the existing Python environment from the repository root:

```sh
python train-3.py
python train-hu.py
python migrate.py
```

There are no command-line flags or argument parsers. `train-3.py` trains only the
3-max track and `train-hu.py` only HU. The duration is **one hour per launch**,
editable in `training/settings.py` as `SESSION_SECONDS`. Hyperparameters, root
mixtures and replay budgets live in `configs/train_3max.json` and
`configs/train_hu.json`. Each track has one shared advantage model and one average
policy; 30k/20k pooled advantage capacities remain unchanged.

## Resume and stop behavior

The exact working checkpoint paths are `runs/3max/checkpoint.pt` and
`runs/hu/checkpoint.pt`. On launch, a script:

1. Loads its working checkpoint if present.
2. Otherwise loads the full checkpoint associated with its active reference
   policy, if that track has been activated.
3. Otherwise initializes a fresh track and reports that explicitly.

An incompatible or corrupt existing checkpoint raises an error; it does not fall
back to an older active policy or restart silently. Checkpoint configuration hashes
must match the corresponding track configuration, except for explicitly logged
increases to `max_nodes`/`max_depth`. The timed workflow permits those guard
increases, preserves the complete trained state, and records old/new hashes in
checkpoint metadata; other changes and guard reductions still fail. See
[HU traversal budget recovery](hu-traversal-budget.md) for measured limits.
The duration is an operational budget outside that learning configuration, so
it can change between sessions.

The wall budget is checked between complete iterations. A session may overrun by
one iteration; traversal values are never fabricated to meet a timeout. At normal
completion or Ctrl+C, the complete current state is saved atomically to the working
checkpoint. On another failure, the last complete state is saved before raising
the error. Periodic numbered checkpoints, evaluation and metrics remain enabled.
A per-track process lock prevents overwriting the same run. A shared compute
lock prevents running both eight-worker jobs concurrently and oversubscribing
the machine; launch them sequentially.

Working checkpoints take precedence over active policy checkpoints so unpromoted
training progress is preserved between launches. Immediately after activation,
both represent the same state. If the working checkpoint is deliberately removed,
the next session resumes from the activated full checkpoint. An average-policy
network alone is insufficient to resume Deep CFR: the advantage model, replay,
RNG and root sampler states must also be retained.

Independent HU root rollouts explicitly start HU tournaments with 75 physical
chips (balanced 37.5/37.5 initialization, then persistent play), followed by the
same synthetic/stratified mixture. They do not require a previously trained 3-max
policy. The combined pilot configuration remains available through the Python API
and retains the 3-max-to-HU survivor-root setup.

## ASCII migration menu

`migrate.py` shows three operations:

```text
1  Activate checkpoint as the reference policy
2  Export and activate 3-max policy for UI
3  Export and activate HU policy for UI
0  Exit
```

Reference activation can select 3-max, HU or both. UI options 2 and 3 publish
only the selected track, require only its candidate checkpoint, and preserve the
other published UI entry. They do not replace the active reference policy. Missing or untrained candidates fail with
their exact path; no neighboring artifact or old policy is guessed.

The selected checkpoint files are first copied to stable temporary snapshots,
loaded and validated. Average-policy artifacts are produced from those exact
snapshots. Export also verifies PyTorch/ONNX CPU parity before publication.
The menu then displays the source paths, iterations and checkpoint hashes and
requires the exact token `APPLY` to switch the active catalog. Cancellation removes
prepared temporary files and leaves the active catalog unchanged.

Active releases are immutable content-addressed bundles under
`policy/releases/<track>/<checkpoint_sha256>/`, containing the full checkpoint,
`average_<track>.pt` and a checksummed `bundle.json`. UI releases live under
`ui/public/policy/releases/<bundle_sha256>/` and contain the selected tracks'
ONNX files, manifests and a bundle checksum record. Historical releases are retained, not overwritten
as backups or selected through a latest-file search.

`ui/public/policy/index.json` is the single authoritative catalog for active
checkpoint releases and exported ONNX bundles. It is atomically replaced only
once all selected bundles are complete. Thus the combined operation publishes
activation and export through one state change. A publication lock and catalog
hash comparison reject concurrent changes rather than overwriting them. Orphaned
unreferenced immutable bundles after an I/O interruption do not become active.

## Test Live and programmatic core

All UI views automatically load the catalog-selected available exports and route
by seated player count. The catalog is checked every 15 seconds and on window
focus, so a newly published HU export also loads without manual intervention.
A track that has not been exported remains explicitly unavailable. Training
checkpoints are not browser models: publishing via `migrate.py` is still required.
The development server serves public files directly; the bundled production
server serves `/policy/` from `ui/public/policy` so new exports require no rebuild.
Other static hosting must deploy the newly published policy assets.

Overview averages exact-combo policy queries into 169 classes for a fixed public
state; Cas précis queries selected hero cards in that state. Neither view reads
the legacy policy or claims historical visit statistics.

Core functions remain importable:

```python
from training.workflow import train_track, prepare_migration

# Substantial compute: execute only when intended.
result = train_track("3max", seconds=3600)

prepared = prepare_migration("both", ("3max", "hu"))
try:
    # Inspect prepared.summary before deciding to publish.
    prepared.publish()
finally:
    prepared.close()
```

`TrainingRunner.run_for(seconds, checkpoint_path)` implements time budgeting and
safe persistence. Publication work is separate from training. No GPU, online
resolver or range model is needed. Implementing these scripts does not launch an
hour of training; only bounded temporary sessions exercised the new workflow.

Performance implementation and measured real-checkpoint comparison are documented
in [Deep CFR performance refactor](deep-cfr-optimization.md). Timed sessions now
reuse traversal workers across outer iterations; existing checkpoint/config
contracts and training semantics remain compatible.
