"""Importable entry point for local solving and explicit behavior-conditioned search."""
from __future__ import annotations

from typing import Callable, Mapping
from actions import ACTION_IDS
from hybrid.beliefs import BeliefState
from hybrid.diagnostics import Analysis
from hybrid.policy_source import PolicySource, distribution
from hybrid.reference import ReferencePolicy, showdown_strength
from hybrid.ranges import combo
from hybrid.river_solver import RiverSolver
from hybrid.search import search
from hybrid.state import ComputeBudget
from poker_game_expresso import HandState


class HybridDecisionEngine:
    def __init__(self, policies: Mapping[int, PolicySource] | None = None,
                 prior: PolicySource | None = None, *, search_mode: str = "behavior",
                 root_raise_amounts: tuple[float, ...] = (), learned_continuation=None,
                 adaptive: bool = False) -> None:
        if search_mode not in ("behavior", "public_cfr", "response"):
            raise ValueError(f"Unknown hybrid search mode: {search_mode}")
        self.policies = policies
        self.prior = prior
        self.search_mode = search_mode
        self.root_raise_amounts = root_raise_amounts
        self.learned_continuation = learned_continuation
        self.adaptive = adaptive

    def analyze(self, game_state: HandState, belief_state: BeliefState, compute_budget: ComputeBudget,
                *, cancelled: Callable[[], bool] = lambda: False) -> Analysis:
        if game_state.terminal:
            raise ValueError("Analysis requires a live decision")
        proxy_before = showdown_strength.cache_info().misses
        hero = game_state.current_player
        cards = combo(game_state.actor.cards)
        private = belief_state.private_view(hero, cards)
        policies = (dict(self.policies) if self.policies is not None else
                    {p: ReferencePolicy() for p in game_state.players})
        if set(policies) != set(game_state.players):
            raise ValueError("Behavior policy seats must match the current hand")
        from hybrid.actions import local_actions
        actions = local_actions(game_state, self.root_raise_amounts)
        baseline = distribution(policies[hero], game_state)
        methods = {}
        strategy_by_hand = None
        if self.search_mode != "response" and not self.root_raise_amounts and ((game_state.street == "RIVER" and belief_state.public.candidate_count <= compute_budget.max_joint_candidates)
                                           or self.search_mode == "public_cfr"):
            solver = RiverSolver(self.prior)
            if game_state.street == "RIVER" and self.search_mode != "response":
                solved = (solver.solve(game_state, belief_state.public, compute_budget, cancelled)
                          if belief_state.public.candidate_count <= compute_budget.max_joint_candidates
                          else solver.solve_sampled(game_state, belief_state.public, compute_budget, cancelled))
            else:
                solved = solver.solve_chance(game_state, belief_state.public, compute_budget, cancelled)
            if cards not in solved.strategy_by_hand:
                from hybrid.state import BudgetExceeded
                raise BudgetExceeded("Public CFR sample did not cover Hero's query holding; increase samples or use explicit behavior search")
            probabilities = solved.strategy_by_hand[cards]
            ev = solved.action_ev_by_hand[cards]
            work = solved.work
            warnings = solved.warnings
            uncertainty = {"sampling_standard_error": None, "finite_iteration_error": "not bounded",
                           "opponent_model_uncertainty": "not quantified"}
            methods = {a: "exact-terminal/full-tree-CFR" if game_state.street == "RIVER"
                       else "public-information-set-CFR/bounded-chance-and-rollout-leaves" for a in ev}
            if game_state.street == "RIVER" and solved.work.samples:
                methods = {a: "sampled-private-deals/exact-terminal-CFR" for a in ev}
                uncertainty["private_deal_sampling_error"] = "not estimated; empirical finite-game solve"
            strategy_by_hand = solved.strategy_by_hand
        else:
            result = search(game_state, private, hero, policies, compute_budget, cancelled, self.root_raise_amounts,
                            public_beliefs=belief_state, learned_continuation=self.learned_continuation,
                            adaptive=self.adaptive)
            probabilities, ev, work, warnings = result.probabilities, result.action_ev, result.work, result.warnings
            uncertainty = {"sampling_standard_error": result.standard_errors,
                           "paired_difference_standard_error": result.paired_difference_errors,
                           "paired_reference_action": actions[0].action_id,
                           "opponent_model_uncertainty": "not quantified"}
            methods = {a: result.method for a in ev}
            if game_state.street == "RIVER":
                warnings.append("Public range product exceeds exact CFR limit; behavior-conditioned rollout mode selected.")
        versions = {p: policy.version for p, policy in policies.items()}
        if self.policies is None:
            warnings.append("Uncovered continuation information sets use the computed-equity conservative profile; this fallback is not a balanced-equilibrium guarantee.")
        if self.prior is not None:
            versions["local_prior"] = self.prior.version
        if self.learned_continuation is not None:
            versions["continuation"] = self.learned_continuation.version
        return Analysis(actions, probabilities, ev, "physical_chips/hand_delta",
                        {p: r.weights for p, r in private.ranges.items() if p != hero},
                        belief_state.public.ranges[hero].weights, methods, work.nodes, work.samples, work.iterations,
                        uncertainty, versions,
                        ["Displayed ranges are factors; joint card compatibility is enforced during evaluation.",
                         "Values include earlier contributions in this hand.", *belief_state.provenance], warnings,
                        {a.action_id: baseline[ACTION_IDS.index(a.action_id)] for a in actions if a.action_id in ACTION_IDS},
                        game_state.blinds.big, strategy_by_hand,
                        {"terminal_leaf_evaluations": work.terminal_leaves, "rollout_leaves": work.rollout_leaves,
                         "learned_leaves": work.learned_leaves,
                         "behavior_equity_proxy_deals": 24*(showdown_strength.cache_info().misses-proxy_before)})
