# Phase 2 architecture

> Historical evidence: the old models and `artifacts/` were explicitly deleted during training finalization. Numerical results below describe previous bounded experiments, not currently deployed models. See [TRAINING.md](TRAINING.md) for the current configuration, empty catalog and launch/export workflow.

## Decision ownership

The canonical Python and TypeScript poker engines retain all legality, chip accounting, runout and side-pot responsibilities. The hybrid layer maintains explicit beliefs and computes action values. The average-policy network is an optional baseline/prior. No learned continuation is enabled by the delivered experiments.

`HybridDecisionEngine.analyze` keeps action probabilities and chip EVs separate. Small river public ranges use full information-set CFR unless `search_mode="response"` explicitly requests a behavior-conditioned response. `public_cfr` retains the shared-information-set earlier-street forest. Broad default continuation now uses `ReferencePolicy` with an explicit conservative computed-equity fallback. `adaptive=True` is opt-in, not automatically promoted by a favorable action-selection example.

## Persistent session and beliefs

`HybridSession` owns a cloned canonical hand and named `Hypothesis` objects. Each hypothesis contains a probability, compatible joint range factors and explicit per-seat policies. An action is first applied to a candidate canonical hand. Every candidate holding of the acting player gets an observable-only likelihood. Its factor is multiplied by that likelihood; new board blockers are applied. Only a completely compatible posterior commits the transition. Impossible observations leave the Python session unchanged.

A single component is a product of private-hand factors conditioned jointly on disjoint cards. This captures blocker dependence without pretending marginals are independent. A mixture retains a distinct factorization per latent behavior profile, preserving hand/profile correlation across successive actions. Exact joint enumeration computes profile evidence for bounded supports; larger supports use explicitly seeded, bounded evidence sampling. This representation cannot express every arbitrary within-profile private-hand correlation.

Actual Hero cards condition only private action-EV queries. Public profile probabilities and Hero's public range remain unconditioned on that hidden knowledge. Exploitative mixture evaluation privately reweights profile probabilities using the acting holding, then aggregates action EVs across hypotheses. Display marginals are exact when tractable and sampled otherwise; they are never fed back as independent factors. Sampling uncertainty, profile sensitivity, private-hand integration and unbounded continuation approximation are separate diagnostics.

Every action requires a stated likelihood source. A human action is not falsely recorded as a draw from the solver's strategy. Callers may supply an external policy describing that action's likelihood. Off-tree raises retain the actual amount and require explicit interpolation consent; unsupported extrapolation fails. `next_hand` requires settlement and an increasing hand number, fresh priors and explicitly routed policies, supporting elimination and 3-max/HU transitions.

Reference mode uses the reference policy, independently of the action-observation profile. It requires one explicit public reference-range hypothesis. A multi-profile mixture remains an exploitative assumption and is not silently factorized for CFR.

## Strategic references and reactions

`ReferencePolicy` queries a validated exact information-set table when available. `solve_reference` uses the existing river CFR and independent best-response implementation to accept finite-game tables under an explicit response-gain threshold. Saved tables include their measured scope. Public-range changes do not create safe-resolving guarantees.

Outside table coverage, `ProfilePolicy` computes a reproducible 24-deal showdown-equity proxy from candidate private cards and observable board only. Pot odds, legal sizes and explicit synthetic tightness/aggression parameters form an auditable action distribution. This is a conservative fallback population, not an equilibrium theorem, human calibration or direct action-EV calculation. The search still evaluates each legal action through actual chip settlements. Fixed scripted profiles, uniform behavior, public-history adaptive mixtures, published neural models and previous solver tables are retained as controlled opponents.

Behavior learning uses separate models per player count and profile. Exact teacher likelihood distributions label real complete self-play trajectories at varied stacks. Train/held-out separation groups whole trajectories; independent seeds provide log loss, Brier score and calibration. Unknown populations are explicitly OOD. Six bounded prediction variants pass their narrow profile gate; aggressive three-player variants fail. The downstream EV comparison does not justify making learned behavior the default.

## Search, uncertainty and budgets

The canonical search integrates compatible hidden worlds and true conditional future cards. Candidate root actions share random worlds and rollout streams. Terminal values always use engine payouts; unresolved branches use explicit continuation policies. Public Bayesian updates are propagated when learned range-conditioned leaves need them; ordinary branch integration multiplies the explicit behavior likelihoods directly.

Adaptive search uses a bounded independent pilot to allocate additional evaluation to close contenders. Pilot outcomes are excluded from final EV estimates, every action retains fresh samples, and paired errors use only shared evaluated worlds. The experiment trades lower node count against worse EV MSE, so fixed allocation remains the default. Ranking warnings identify insufficient Monte Carlo precision; they are not confidence in the behavior model.

Exact/sampled river and chance CFR remain finite belief-conditioned approximations. Independent exhaustive turn references test policies beyond their fitted boards. Upstream counterfactual values can be carried by the existing belief contract, but safe resolving is not implemented. No three-player Nash guarantee is claimed.

Compatible sampling now caches immutable cumulative proposal weights. It still rejection-samples the entire independent product; it does not sequentially renormalize seats. Seeded deals are unchanged. Tree nodes, chance worlds, iterations and behavior equity-proxy computations are distinguished. Evidence/display range sampling has its own explicit session budget.

## Offline data and models

Existing corrected HU and partial-enumeration 3-max collectors, root stratification, objectives and original checkpoint schemas remain. Coverage now reports additive weight moments/ESS and all-street class/combo counts separately for generated and retained memories. Raw generated card IDs and canonical retained identities are labeled distinctly.

`ml.stratified_memory.ProtectedReplay` is an opt-in opening/other stratified reservoir. It exposes inverse-inclusion-corrected samples for existing fit functions and complete deterministic persistence. Its total-loss estimator preserves the stream expectation; normalized weighted fitting is a finite-sample ratio estimator. Set `replay_opening_fraction` explicitly to integrate it into `TrainingRunner`; optional checkpoint records carry a tagged allocation, while original configs and reservoir checkpoint records remain unchanged. The runner checks allocation compatibility and resumes both stratum RNGs, uncorrected records and corrected fitting exactly. Outcome-sampling use is rejected.

Continuation experiments compare the previous 169-class features against exact 1326-combination factors per player, jointly canonicalized with candidate hand and board over suit permutations. Training uses independent public-root groups, diverse boards/stacks/positions/streets/contexts and separate continuation profiles. Labels come from independent terminal rollouts, with compute budgets and conservative sampling-error metadata; they are not called exact equilibrium values. Existing exact-terminal CFR labels remain available with finite-iteration caveats.

Capacity sweeps use 32/96 hidden units, bounded epochs and complete model/data/optimizer checkpoints. JSON and ONNX exports preserve status and numerical parity. Additional fitting invalidates acceptance until revalidated. Learned leaves require a passed gate, compatible player count/street/feature schema and physical chip bounds. All tested value models fail; rollouts and tables remain available.

## Browser execution

`behavior.ts` matches the Python observable policy proxy, including canonical suits and integer PRNG. `beliefs.ts` updates exact-combo public factors and checks engine transitions and compatible support. `hybrid.worker.ts` replays observed transitions from the session prior and performs the real solve; changing likelihood assumptions explicitly replays the public history. Continuation profiles and observation-likelihood profiles are separate inputs.

The existing Specific spot page can play complete hands, lock known cards once play begins, preserve exact unusual raises and maintain posterior ranges automatically. The existing tournament test table also embeds the same analysis component; bot actions are explicitly modeled as uniform because those test opponents are uniform. Its canonical current-hand history is reconstructed to include actions before the first Hero turn and batches of bot responses. Private/future simulator cards are used only by engine replay, never policy features.

The UI preserves budget selection, shows reference/exploitative modes, profile assumptions, baseline versus calculated probabilities, all legal-action EVs and Monte Carlo errors, 169-class/exact-combo ranges and explicit failures. Heavy computations stay in a reusable worker. Cancellation terminates it and never presents an incomplete result. There is no mandatory Python backend. Browser profile selection is explicit; latent multi-profile Bayesian mixtures are currently a Python analysis capability.

## Reproduction

```python
from examples.phase2_demo import demonstrate
result = demonstrate(2)  # Use 3 for a complete three-player hand.
print(result["utilities"])
```

Experiment owners are `hybrid.quality`, `phase2_learning`, `reference_experiments`, `matches`, `robustness` and `replay`. Their functions take explicit paths/budgets and have no argparse interface. Reports/checkpoints are under `artifacts/phase2`. The acceptance document provides exact completed budgets and remaining quality limitations.

## Opponent-seat range inspector

Test Live opponent seats open an interactive 13×13 posterior matrix on hover, focus or tap. Each cell is class probability mass, not a policy action frequency or a fraction of selected combos. Clicking a class exposes exact-combo posterior probabilities; all combos remain inspectable. The inspector uses the existing worker/session factors and selected action-likelihood profiles. It never feeds display marginals back into Bayesian factors or uses simulated hidden opponent cards.

For a fixed observer, at most two unknown hands remain. For holding `{a,b}`, compatible mass in the other factor is `total - card_mass[a] - card_mass[b] + pair_mass[a,b]`. Multiplying by its own factor and normalizing produces exact observer-conditioned marginal probabilities in linear range size. Board/Hero blockers and folded-player factors remain included. Snapshot state/session keys suppress stale distributions during actions and hand changes; errors and pending updates are explicit.

### Range and reaction views

The range inspector uses a fixed diverging log-ratio scale against uniform exact combos excluding Hero and board cards: orange at or below ¼×, neutral at 1×, cyan at or above 4×. Class percentages remain posterior masses. This removes the visual 12/6/4-combo multiplicity bias; a uniform opponent correctly looks neutral. Impossible classes have hatching, while possible but zero-weight classes use the low end of the scale. The uniform comparator is not a prior inferred from action history.

The optional reaction view groups the selected behavior profile's legal-action probabilities into fold, check/call and bet/raise segments, averaged within each class using its current posterior combo weights. It is available only for the currently acting opponent; no counterfactual turn or action set is invented. The worker receives only candidate hand/public observations for policy inference. The displayed percentages still mean holding probability, with reaction frequencies in cell tooltips.

Postflop composition uses mutually exclusive best-five-card categories, including the board: two pair or better, one pair, unpaired flush/straight draw before the river, and remaining high card. Paired draws stay in the pair group. These presentation quantities never replace the public factors or search values.
