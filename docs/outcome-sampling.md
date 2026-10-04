# Outcome-sampling estimator contract

The external-sampling traversal
remains the all-traverser-actions reference for finite hand/subgame validation.
Outcome sampling is an explicitly selected alternative for current-hand chip-EV work.
No neural leaf value, tournament-win proxy, artificial hand horizon,
importance clipping, or rejection of long trajectories is part of the estimator.

## Behavior policy and reaches

Freeze the target strategy sigma for an outer iteration. At every decision,
including opponents, sample exactly one legal action from

    q(a | h) = (1 - epsilon) * sigma(a | I(h)) + epsilon / |A(h)|,
    0 < epsilon <= 1.

The initial exploration setting is epsilon=0.6. Mixing all players rather than
only the updating player gives full support for the separate own-reach average
estimator too, even where an opponent's target strategy assigns zero probability.
This is exploration, not weight clipping. Chance is sampled from the engine's
true distribution, with injected seeds, so chance likelihood ratios cancel.

For updating player i, at a visited history h define decision-only prefixes:

    P_i(h) = product of sigma for i's preceding actions
    P_-i(h) = product of sigma for all other players' preceding actions
    Q(h) = product of q for every preceding player action

These exclude chance. The complete behavior reach is pi_c(h) Q(h), and the
complete counterfactual reach is pi_c(h) P_-i(h). Store logarithms of these
prefixes so products do not silently underflow. A true zero target probability
has log reach -infinity; a positive unrepresentable result is an explicit error.

## Regret target

Walk one trajectory to a terminal state or an exact settled payoff for player i
(e.g. a fold in this hand). Initialize V_hat(z)=u_i(z). In reverse order, if action b
was sampled at h, compute

    V_hat(h,a) = 1[a=b] * V_hat(hb) / q(b | h)
    V_hat(h) = sigma(b | I(h)) * V_hat(hb) / q(b | h).

At each visited decision of i emit, for every legal a,

    r_hat(h,a) = P_-i(h) / Q(h) * (V_hat(h,a) - V_hat(h)).

Illegal targets are zero. The regret sample's additional training weight is 1:
the prefix and continuation importance correction is already in its target.
Missing infosets contribute zero for that traversal, not an omitted denominator
when evaluating Monte Carlo means. Taking expectations over visits and suffixes
recovers the full counterfactual regret increment on finite trees. In particular,
future decisions of i on the sampled trajectory also receive regret samples.
The returned root value includes all continuation sigma/q ratios.

Signed values and positive ratios are combined in log space before conversion
to finite float64 targets; there is no cap. Float32 training must reject target
values it cannot represent rather than silently clamp them.

## Average-policy target

At visited i nodes, on the same trajectory, emit sigma(I(h)) with weight

    w_avg(h) = P_i(h) / Q(h).

If P_i is exactly zero, the average contribution is exactly zero and is omitted;
regret traversal continues because own zero reach does not imply zero regret.
In expectation visitation contributes pi_c(h) P_i(h) sigma(I(h)), independent
of opponent strategy reach. Under perfect recall the fixed chance mass cancels
in each infoset's normalized average. Tabular validation uses uniform iteration
weighting; the neural trainer additionally multiplies by the outer iteration,
matching its existing linear averaging convention. This is not a claim that a
finite neural replay produces an exact tabular equilibrium.

## Existing overflow repair, precisely

The previous training loss constructed raw iteration-weighted values
w_j = sample.weight * sample.iteration directly as float32. Finite float64
weights such as 1e200 became infinity before normalization. The repair computes
one training-population mean in Python float64:

    s = max_j w_j
    mean_w = s * (sum_j(w_j / s) / N),

then converts w_j / mean_w to float32. A minibatch uses that same population
mean, not its own mean. Thus the full objective is unchanged:

    mean_j[(w_j / mean_w) L_j] = sum_j(w_j L_j) / sum_j(w_j).

Dividing by the maximum is only a temporary way of evaluating the mean without
overflowing its sum. No individual weight is clipped, capped, winsorized, or
replaced. The existing repair still rejects a nonfinite float64 product or mean.
Extreme float32 underflow and unrepresentable regret targets must also fail
explicitly. This arithmetic repair does not reduce importance-sampling variance.

## Current hand boundary and budgets

Both samplers extract the existing HandState from a tournament input. They stop
at hand settlement (or an exact folded-player payoff), using after-minus-before
chips. Neither traversal calls start_hand, schedules blinds, or follows another
hand. See [the cEV architecture and diagnostics](hand-cev.md).

Budgets remain operational guards: a failed iteration publishes no partial
learning batch and is not retried until a short trajectory succeeds. The
persistent tournament simulator and its potentially unbounded sequence of hands
are separate from this finite per-hand learning problem. Non-winner-take-all
payouts require a future independent ICM/$EV objective and are rejected here.

## Reference sources and validation

- [Lanctot et al., 2009](https://www.cs.cmu.edu/~waugh/publications/nips09b.pdf):
  MCCFR and outcome-sampling estimators for finite extensive games.
- [OpenSpiel's outcome-sampling implementation](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/algorithms/outcome_sampling_mccfr.py):
  zero-baseline continuation estimates and prefix reach correction.

Validate exact expectations by enumerating all sampled paths in small poker
subgames; independently compare empirical means (zeros for unvisited infosets)
with full-tree and external-sampling increments. Test nonuniform policies,
zero-probability actions, deeper updating-player nodes, masks, average weights,
atomic budget failure, and explicit numerical errors. No Kuhn/Leduc surrogate.

## Implementation and historical tournament-length baseline

`outcome_sampling.OutcomeSamplingTraversal` implements this contract iteratively,
so sampled path length does not consume Python recursion depth. It retains
explicit operational node/depth guards. `DeepCFRSolver.run_iteration` selects it
with `traversal_mode="outcome_sampling", epsilon=0.6`; the default external mode
is the primary per-hand Deep CFR generator. One outcome path supplies both sample
kinds. Replay schema v4 records the estimator and rejects mixing; checkpoints
record its mode and exploration settings. Budget/numerical failures abort the
iteration without publishing replay, models, or RNG updates.

`tests/test_outcome_sampling.py` compares independent full-tree expectations
against all outcome paths, including zero-probability target actions. It also
compares 1,000 outcome trials in each of heads-up and three-player river states,
plus 1,000 external trials in the heads-up state, counting absent infosets as
zero over the entire trial population. Tests
use a six-empirical-standard-error tolerance, backed by exact path enumeration;
they do not claim finite-sample convergence of a learned neural policy.

The retained **historical tournament-long** raw report is
[outcome-diagnostics.json](outcome-diagnostics.json). It predates the current
hand-terminal objective and is not a measurement of the new cEV traversal.
The current measurements are [hand-diagnostics.json](hand-diagnostics.json).
Twelve seeds were used with each of two frozen, synthetic target policies
(uniform and mildly skewed), epsilon=0.6, and 300-node/300-depth guards. These
are paired seeds across policies, not 24 independent draws from one learned
policy. No neural training is part of the measurement.

| Measurement | Median | 95th empirical percentile | Maximum |
| --- | ---: | ---: | ---: |
| Actual terminal tournament hands, 24 paths | 5 | 12.7 | 22 |
| Average-policy weight, 644 visited updating-player nodes | 3.47e5 | 1.49e16 | 3.12e24 |
| Complete regret correction (prefix * suffix / chosen q), 644 nodes | 5.98e3 | 3.28e12 | 1.74e21 |

The 72 additional updating-player paths comprise 60 terminal paths and 12 exact
early-settled eliminations. All 96 normal probe paths completed, with no censored
or numerical failures, in approximately 0.67 seconds on this checkout. The
separate one-node guard probe kept all eight censored attempts, with zero
learning samples. Raw finite-weight quantiles and log quantiles are both saved;
structural zeros, conversion underflows, and overflows have distinct counters.

The largest absolute regret target was 7.76e20. Squaring a value this large in
the current float32 MSE can overflow (sqrt(float32 maximum) is about 1.84e19).
The finite-loss guard aborts; weights/targets are not clipped to make training
appear successful. Large gradient/optimizer moments are additional numerical
limits. The overflow repair described above preserves weight normalization; it
does not fix these separate limits or sampling variance.

This measurement is too small and its policies too restricted to characterize
rare long tournaments, learned-policy importance tails, or expected full-run
cost. The infinite-horizon question remains open. Status remains
**SMOKE-TRAIN READY**, not full-tournament training ready. No expensive training
has been launched.

Historical checks before the cEV migration: `.venv/bin/python -m unittest discover -s tests -p 'test_*.py'`
passed 74 tests in 7.482 seconds. Focused outcome tests passed 21 tests. Python
compilation and `git diff --check` passed. Existing PyTorch ONNX-export
deprecation/tracing warnings remain; no test failures were suppressed.
