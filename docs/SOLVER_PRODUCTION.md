# Solver production cutover

Implementation and local validation complete. Remote activation requires checking the deployed revision, service logs, public solver histories and production UI.

| Requirement | Status | Evidence |
|---|---|---|
| Persistent solver service, HU and 3-max | DONE | `training/solver_service.py`; `deploy/vps/train.py` now runs this service |
| Adaptive labels and optional persistent students | DONE | Separate response-label semantics, explicit unresolved precision, all 16 students fitted in preflight |
| Bounded publication and independent evaluation | DONE | `solver/{track}/` Storage namespace; evaluator does not modify candidate tables |
| Training UI shows solver measurements | DONE | `SolverProgress.tsx`, strict parser, no legacy NN curve fallback |
| Tests and resource preflight | DONE | 57-test deployment/regression suite, five focused service tests, 162 browser tests, lint and static build |
| OVH activation and public read-back | IN PROGRESS | Verify live processes, knowledge files, hourly history, Vercel deployment |

## Ownership and contracts

The same systemd training services now call the solver worker. Each track owns `/var/lib/gto/{track}/solver-v1`, independently of old Deep CFR replay/checkpoints. Deployment preflight validates or initializes that explicit contract. Old checkpoints remain untouched for recovery; they are not resumed by the new entrypoint.

All teacher targets come from the poker engine and explicit CFR regrets/average masses. Private worlds are sampled from compatible public ranges; sampled and exact game contracts differ. Production uses depth-one forests with explicit policy rollouts on every street, including river. The small river reference games remain full-tree solves. The range-conditioned student input correctly retains continuation identity and depth for approximate river leaves; it does not mislabel them terminal/exact.

Each computed artifact persists regrets, average masses, range/forest contracts, root EVs and cumulative committed work. Atomic writes plus per-game reconciliation recover a calculation committed before the global cursor. A single worker lock protects the run. Optional student fitting uses persistent weights and Adam states, with an explicit evolving-dataset permission; frozen-replay studies still reject changed datasets by default.

## Production allocation

Per track: five families, four streets, 100 roots per family/street (2,000 distinct roots), up to 16 visits of 16 CFR iterations per root, at most 256 requested iterations. Sixteen compatible private worlds, depth one, 300,000 nodes per visit. Completed iterations are retained if a node cap prevents the requested count; actual counts are reported. Roots that cannot complete one iteration are explicitly recorded, not retried indefinitely. Completion of the allocation is not convergence.

Each root also gets one adaptive fixed-policy response label: 32/128/512 paired worlds, 0.1 BB selection tolerance, 300,000 nodes. Unresolved labels remain unresolved and are not treated as strategy-average supervision.

Optional student fitting runs every 32 visits, with 16 updates per specialist and at most 64 tables stratified by street/split. Encoded tensors are capped at 128 MiB. Missing splits skip fitting with an explicit status; a failed optional fit is reported as rejected and does not stop CFR generation. Students remain unaccepted and never steer teacher traversals.

Local storage cap: 1 GiB per track, including tables, labels and the student checkpoint; an individual table is capped at 8 MiB. Atomic replacement temporarily needs the replaced artifact's size. Once the finite allocation is exhausted the service waits for an explicit extension instead of running redundant computation indefinitely.

## Publication and UI

- Every minute: compact progress plus four latest scoped street tables under `solver/{track}/status.json` and `solver/{track}/tables/`.
- Hourly: solver reference diagnostics in `solver/{track}/history.json`, up to 720 points. No BB/100 label or NN baseline is reused.
- The evaluator measures four fixed independent sentinel roots against stronger computations. Candidates are refined by the worker, never by the evaluation request. Candidate refinement caps at 256 iterations, so flat curves after that cap are expected. This measures local convergence on fixed spots, not generalization from the growing dataset or full-game winnings.
- Training displays the separate solver series, connected street curves, actual computed counts and precision outcomes. HTTP/schema failures surface explicitly; no fallback to old history.

Existing ONNX exports remain auxiliary policies for consumers that explicitly use them. This cutover does **not** promote the new students, replace Overview's policy probabilities, or claim that a scoped table is an arbitrary-state live policy. Test Live's local solver remains the decision authority. New tables are published as reusable scoped artifacts, not silently injected into unrelated live states. Legacy NN history remains intact but is no longer the Training page's default.

## Experiments and failed approach

Initial full-tree multiway river generation was unsuitable: a deep 3-max artifact exceeded 500 MiB and exhausted its proposed storage allowance. Reducing only iteration counts would not fix the large tree/observation representation. The deployed alternative uses explicit depth-limited rollout leaves and keeps exact small-river validation separate.

Final bounded pilot: `runs/solver-production-bounded/{hu,3max}/preflight.json`.

| Track | Subgames / labels | Committed nodes | Total CPU including references/fit | Storage including students | Process peak RSS |
|---|---:|---:|---:|---:|---:|
| HU | 20 / 20 | 52,417 | 26.63 s | 31,614,469 bytes | 342,081,536 bytes |
| 3-max | 20 / 20 | 57,933 | 46.04 s | 32,439,575 bytes | 349,011,968 bytes |

All 40 response labels remain indeterminate at the requested 0.1 BB tolerance. All 16 students fitted on bounded data, none accepted. The process RSS is cumulative across this sequential local experiment, not a VPS RAM guarantee. Initial OVH read-only preflight: 57 GiB disk free and approximately 4 GiB available RAM while old services still ran.

Tests cover sampled resume equality, contract incompatibility, durable table recovery, independent Storage namespaces, read-only evaluation, evolving-dataset opt-in, optimizer checkpoint restore, and approximate-river continuation encoding. The GitHub deployment workflow now includes these solver tests before activation.

Next verification: observe both production workers producing new tables/labels, run the hourly publisher, read back both public solver histories, and confirm the Vercel Training page uses the new UI. General poker strength and useful neural amortization remain unvalidated.
