# Archived experimental implementation

These sources retain the previous implementation for audit and comparison.
They are not supported training or interactive-play entry points. Their imports
and assumptions may no longer match the authoritative modules in the repository
root; they are intentionally not maintained through compatibility wrappers.

| Archived path | Superseded behavior |
| --- | --- |
| `rollout_cfr.py` | Rollout-based pseudo-CFR with nonstandard reach weighting |
| `parallel_regret_merge.py` | Local worker updates merged from stale regret tables |
| `packed_infoset.py` | Lossy bucketed packed-u64 states |
| `compact_policy.py`, `stats_policy.py`, `artificial_policy.py` | Unversioned compact policy tooling |
| `bucket_model.py`, `policy_distillation.py`, `bucket_viz.py` | Bucket-specific policy distillation and heatmaps |
| `benchmark_cfr.py`, `config.py` | Legacy independent-hand benchmarks and defaults |
| `ui/` | Duplicated TypeScript engine, old policy decoder, and independent-hand views |

The pre-existing modified model artifact remains at its original path,
`ml/trained_policy_model.pth`. It belongs to the archived bucket-specific model
contract. Existing old policy artifacts are also incompatible with the new
state, action, and objective contracts. Neither the engine service nor the new
solver loads them automatically. This preservation does not indicate that the
artifacts encode a valid equilibrium or a correct tournament policy.

`push_fold/` is a separate historical reduced-game experiment; it is not part of
the new tournament/Deep CFR pipeline. The live UI retains the original visual design and secondary Overview/Cas précis
analysis helpers; these are explicitly legacy-only and never drive Test Live.
The authoritative implementations are
`poker_game_expresso.py`, `tournament.py`, `actions.py`, `infoset.py`,
`cfr_solver.py`, and the current `ml/` source modules.
