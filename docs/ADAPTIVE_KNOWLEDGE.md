# Adaptive knowledge generation

Implementation: COMPLETE for the bounded workflow. Strategic generalization: NOT VALIDATED. All work is local; production training, published artifacts, and network architectures are unchanged.

| Requirement | Status | Implementation and evidence |
|---|---|---|
| Adaptive paired action differences | DONE | `hybrid/adaptive_labels.py`; finite-look simultaneous intervals, paired samples, exact terminal checks |
| Persistent uncertainty and provenance | DONE | `hybrid/label_dataset.py`; versioned artifacts, source/configuration fingerprints, explicit unresolved labels and resume |
| Diverse scalable curriculum | DONE | `hybrid/knowledge_curriculum.py`; 4,000 planned addresses, whole-family splits, 40 generated legality cases |
| Quality and cost amortization | DONE | `hybrid/knowledge_benchmark.py`; direct CFR, computed cache, root NN prior plus CFR, upfront costs included |
| Bounded experiments and profiling | DONE | `examples/adaptive_knowledge.py`, `hybrid/adaptive_label_experiments.py`; eight broad labels, 36 exact checks, eight comparison spots |
| Large diverse dataset and neural benefit | NOT VALIDATED | Only eight broad labels calculated; no large run or model promotion |

## Label semantics

Each sample uses a compatible private deal and conditional future deck. All legal root actions share the same continuation seed. Existing poker rules settle terminal chip utilities; values are converted to current BB. Continuation policies are explicit, fixed, and must be pure for the independent-world assumptions to apply.

These are response values under declared continuation policies, not equilibrium values, cumulative CFR regrets, or average-strategy targets. They are not automatically injected into the strategy replay. `response_target` rejects unresolved estimates and returns relative action values, pair intervals, continuation identities and the selection-loss bound. Absolute action-EV precision must not be inferred from precise paired differences.

Let S be total conserved chips divided by BB, m the number of action pairs, L the number of predeclared looks, n the sample count, and v the empirical variance of a paired difference. With z = log(4mL/alpha), the radius is:

```
sqrt(2*v*z/n) + 14*S*z/(3*(n-1))
```

Differences lie in [-S, S]. Two sides, all pairs and all scheduled looks are covered by a union bound using [Maurer and Pontil, Theorem 4](https://arxiv.org/abs/0907.3740). This is per-label coverage under the sampling assumptions, not simultaneous coverage for all 4,000 labels or uncertainty about the behavior model.

A label stops when a unique action is separated or its upper selection-loss bound is below the requested tolerance. Otherwise it remains `indeterminate`. If a node limit interrupts sampling between scheduled looks, the last scheduled estimate is retained; incomplete paired rounds are discarded. No new confidence statement is computed at a cost-dependent stopping point. Raw paired samples and every scheduled interval remain available.

## Curriculum and persistence

The manifest contains 100 roots per family, track and street: 4,000 addresses, with 1,600 training, 800 validation and 1,600 test roots.

| Family | Split | Base stack BB | Flop texture | Raises | Range support/player | Continuation |
|---|---|---:|---|---:|---:|---|
| shallow-dry-open | train | 5 | dry | 0 | 8 | uniform |
| medium-two-tone-open | train | 15 | two-tone | 1 | 32 | passive |
| medium-paired-asymmetric | validation | 18 | paired | 1 | 48 | passive |
| deep-monotone-threebet | test | 35 | monotone | 2 | 128 | aggressive |
| shallow-paired-open | test | 7 | paired | 0 | 16 | uniform |

Seeds identify complete scenarios. Roots use legal hand transitions, jittered stacks, optional asymmetry, explicit nonuniform compatible ranges and actual raise histories. Texture refers to the flop, not the preflop decision. These are controlled synthetic families, not a realistic population claim. Entire families are withheld; no related-record random split is used.

The builder is single-writer, atomic and resumable. Its contract includes manifest, budgets, numerical source dependencies, Python and evaluator versions. It rejects incompatible destinations instead of guessing. Completed unresolved labels are also retained on resume. Changing a budget requires a distinct experiment. CPU limits are checked between labels; each label has explicit node/world limits. Storage limits are checked before generation and before writing. Atomic replacement can temporarily require one additional artifact's space.

## Retained measurements

Artifacts: `runs/adaptive-knowledge/` (local generated results, not production models).

- `planned-4000.json`: addresses only; the full dataset was not calculated.
- `pilot-manifest.json`, `labels/`: eight cases covering HU/3-max and all four streets. Looks 32/128/512, alpha 0.05, tolerance 0.1 BB, maximum 100,000 nodes per label, seed 141. All eight remain indeterminate at 512 worlds. Unprofiled generation: **12.863 CPU seconds**. Resume generated **zero** new labels.
- `exact-terminal-check.json`: 36 seeded terminal-river checks, three explicitly weighted ranges, looks 32/128/512/2,048, tolerance 0.1 BB, maximum 10,000 nodes. Zero interval violations across inspected looks. The clearest 12 cases resolved at 128 worlds. Eight close cases remained indeterminate at 2,048. This limited check does not establish broad empirical calibration.
- `amortization-unprofiled/report.json`: five timing repeats, all eight track/street routes. Teachers: 32 iterations, 32 worlds, depth 1, seed 100. Stronger references: 256 iterations, 256 worlds, seed 101. Both capped at one million nodes. All 16 students receive 64 optimizer updates on the separate, deliberately tiny 24-root reference curriculum. Every model remains unaccepted.

The amortization comparison uses tiny independently split roots, not the broad curriculum above. Non-river references remain sampled approximations. Loss below is mixed-strategy root response shortfall against reference action values, not exploitability or realized poker winnings.

| Track/street | Direct CPU ms | NN + CFR CPU ms | Cached CPU ms | Direct loss BB | NN + CFR loss BB |
|---|---:|---:|---:|---:|---:|
| HU preflop | 19.886 | 20.355 | 0.324 | 0.017985 | 0.001985 |
| HU flop | 20.188 | 20.862 | 0.414 | 0.012134 | 0.002649 |
| HU turn | 21.778 | 23.550 | 0.501 | 0.243040 | 0.235596 |
| HU river | 4.214 | 5.064 | 0.431 | 0.048600 | 0.042157 |
| 3-max preflop | 31.526 | 32.042 | 0.708 | 0.054840 | 0.045256 |
| 3-max flop | 28.985 | 28.937 | 0.948 | 0.719348 | 0.703851 |
| 3-max turn | 31.774 | 32.593 | 1.614 | 0.011426 | 0.002037 |
| 3-max river | 27.939 | 28.696 | 1.003 | 0.004419 | 0.005310 |

Offline teacher generation/persistence plus model initialization, fitting/evaluation and checkpoint serialization cost 1.577 seconds HU and 1.568 seconds 3-max. Reports include ranking errors, argmax EV losses, repeated timings and total CPU at 1/10/100/1,000 queries.

Computed cache reuse breaks even after 2 queries except HU river at 3. This requires the identical root, ranges and continuation assumptions; it is not generalization. Cache timing queries all feasible holdings and uses already computed resident values.

NN assistance reduces measured mixed loss on seven spots but usually adds CPU. The 3-max flop median saves only 0.048 ms: the arithmetic break-even is 32,665 queries, but the timing difference is within observed variability and is not reliable evidence of acceleration. The 3-max river fails the descriptive quality filter. No NN is accepted. Full-track training cost is conservatively charged to each spot's amortization calculation.

`cpu.profile` and `cpu.txt` retain the initial instrumented pilot. Rollout policy observation construction dominates: `observe` accumulates 21.7 instrumented seconds and rollout 24.7 seconds. These nested timings are not additive and must not be substituted for the final unprofiled measurements. No rule semantics were changed to optimize this cost.

## Validation and reproduction

Relevant suite: 32 tests covering adaptive generation, search, river solving, solver inputs and solver authority. Nine new tests exercise paired-noise cancellation, sequential-look penalties, physical bounds, exact payoffs, deterministic seeds, interrupted rounds, cancellation, target rejection, family ownership, legal generation, compatible deals, resume contracts, storage caps and amortization gates.

```python
from pathlib import Path
from examples.adaptive_knowledge import demonstrate

report = demonstrate(Path("runs/adaptive-knowledge-reproduction"))
```

Use a fresh destination. The default demonstration uses three timing repeats; the retained final benchmark uses five. No CLI, remote deployment, publication or expensive training is triggered.

## Remaining limitations and next executable action

The broad-family generator and precision-aware dataset workflow are implemented; the broad training dataset and strategic improvement are not established. Conservative whole-stack bounds need many samples for close decisions. Tight paired uncertainty cannot fix a wrong continuation policy. The tiny amortization study does not establish benefit on broad ranges.

Next: select close and separated cases from the frozen family manifest, compare larger scheduled sample budgets with independently stronger continuations, and inspect selection-loss bounds per CPU second before generating thousands of labels. Keep unresolved targets explicit. Profile observation construction before any optimization; preserve identical numerical outputs. Only then assess auxiliary models on whole unseen families at matched total CPU. Tables and direct solving remain the default source of calculated knowledge.
