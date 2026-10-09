"""Full-tree simultaneous CFR on a finite belief-conditioned river subgame.

Chance weights enter counterfactual regret, but not own-reach averaging.
An information set is averaged once per iteration, independent of opponent
reach or the number of compatible hidden deals representing it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Callable

from actions import ACTION_IDS, SolverAction, legal_actions
from cfr_solver import regret_matching
from infoset import Observation, observe
from poker_game_expresso import HandState
from hybrid.policy_source import LocallySolvedPolicy, PolicySource, UniformLegalPolicy, distribution
from hybrid.ranges import Combo, JointDeal, JointRanges
from hybrid.state import BudgetExceeded, ComputeBudget, WorkCounter, instantiate


@dataclass
class TreeNode:
    player: int | None
    key: str | None
    actions: tuple[SolverAction, ...]
    children: tuple[TreeNode, ...]
    utilities: tuple[float, ...] | None = None


@dataclass
class RiverSolution:
    policy: LocallySolvedPolicy
    strategy_by_hand: dict[Combo, dict[str, float]]
    action_ev_by_hand: dict[Combo, dict[str, float]]
    regrets: dict[str, tuple[float, ...]]
    strategy_mass: dict[str, tuple[float, ...]]
    expected_utilities: dict[int, float]
    work: WorkCounter
    complete_iterations: int
    warnings: list[str] = field(default_factory=list)


class RiverSolver:
    def __init__(self, prior: PolicySource | None = None) -> None:
        self.prior = prior

    def solve(self, public: HandState, ranges: JointRanges, budget: ComputeBudget,
              cancelled: Callable[[], bool] = lambda: False) -> RiverSolution:
        if public.terminal or public.street != "RIVER":
            raise ValueError("River solving requires a live river decision")
        return self._solve(public, ranges, budget, cancelled, chance=False)

    def solve_chance(self, public: HandState, ranges: JointRanges, budget: ComputeBudget,
                     cancelled: Callable[[], bool] = lambda: False,
                     continuation: PolicySource | None = None) -> RiverSolution:
        """CFR on a bounded public private-deal/runout forest with shared infosets.

        Turn rivers are exhaustive when the sample budget covers all outcomes.
        Otherwise, fixed sampled chance and baseline leaf rollouts define the
        finite approximation being refined; they are not exact-game guarantees.
        """
        if public.terminal or public.street == "RIVER":
            raise ValueError("Chance search requires a live preflop, flop or turn decision")
        return self._solve(public, ranges, budget, cancelled, chance=True, continuation=continuation)

    def solve_sampled(self, public: HandState, ranges: JointRanges, budget: ComputeBudget,
                      cancelled: Callable[[], bool] = lambda: False) -> RiverSolution:
        """CFR on a seeded empirical joint-deal distribution, independent of Hero's query.

        Every sampled deal has weight 1/N after exact compatibility rejection.
        This returns encountered holdings only; it never invents a policy for an
        unsampled private hand or calls the empirical game the exact full game.
        """
        if public.terminal or public.street != "RIVER":
            raise ValueError("Sampled private-deal CFR currently requires a live river")
        rng = random.Random(budget.seed)
        deals = []
        for _ in range(budget.samples):
            if cancelled():
                raise BudgetExceeded("Analysis cancelled during private deal sampling")
            deals.append(JointDeal(ranges.sample(rng), 1 / budget.samples))
        return self._solve(public, ranges, budget, cancelled, chance=False, private_deals=tuple(deals))

    def _solve(self, public: HandState, ranges: JointRanges, budget: ComputeBudget,
               cancelled: Callable[[], bool], *, chance: bool,
               continuation: PolicySource | None = None,
               private_deals: tuple[JointDeal, ...] | None = None) -> RiverSolution:
        if set(public.players) != set(ranges.ranges) or tuple(public.board) != ranges.board:
            raise ValueError("Public state and ranges must have identical players and board")
        seats = tuple(public.players)
        seat_index = {p: i for i, p in enumerate(seats)}
        actor = public.current_player
        work = WorkCounter(budget, cancelled)
        observations: dict[str, Observation] = {}
        regrets: dict[str, list[float]] = {}
        masses: dict[str, list[float]] = {}
        rng = random.Random(budget.seed)
        baseline = continuation if continuation is not None else UniformLegalPolicy()

        def build(state: HandState, depth: int = 0) -> TreeNode:
            work.visit()
            if chance and depth >= budget.max_depth and not state.terminal:
                from cfr_solver import sample_index
                state = state.clone()
                work.rollout_leaves += 1
                while not state.terminal:
                    work.visit()
                    selected = ACTION_IDS[sample_index(distribution(baseline, state), rng)]
                    next(a for a in legal_actions(state) if a.action_id == selected).apply(state)
            if state.terminal:
                work.terminal_leaves += 1
                return TreeNode(None, None, (), (), tuple(state.utility(p) for p in seats))
            obs = observe(state)
            key = obs.key()
            observations[key] = obs
            regrets.setdefault(key, [0.0] * len(ACTION_IDS))
            masses.setdefault(key, [0.0] * len(ACTION_IDS))
            children = []
            actions = legal_actions(state)
            for action in actions:
                child = state.clone()
                action.apply(child)
                children.append(build(child, depth + 1))
            return TreeNode(state.current_player, key, actions, tuple(children))

        deals = ranges.enumerate(budget.max_joint_candidates) if private_deals is None else private_deals
        if private_deals is not None:
            work.samples = len(private_deals)
        if not chance:
            trees = [(deal, build(instantiate(public, deal.hands))) for deal in deals]
            chance_method = "exact river"
        else:
            from hybrid.ranges import JointDeal
            trees = []
            remaining = 52 - len(public.board) - 2 * len(seats)
            exhaustive = public.street == "TURN" and len(deals) * remaining <= budget.samples
            per_deal = remaining if exhaustive else budget.samples // len(deals)
            if per_deal < 1:
                raise BudgetExceeded(f"Need at least {len(deals)} chance samples to cover every joint private deal")
            for deal in deals:
                initial = instantiate(public, deal.hands)
                for index in range(per_deal):
                    state = initial.clone()
                    if exhaustive:
                        card = initial.deck[index]
                        state.deck = [c for c in initial.deck if c != card] + [card]
                    else:
                        rng.shuffle(state.deck)
                    trees.append((JointDeal(deal.hands, deal.probability / per_deal), build(state)))
                    work.samples += 1
            chance_method = "exhaustive turn rivers" if exhaustive else "fixed sampled chance forest"
        tree_nodes = work.nodes
        warnings = ["Belief-conditioned local solve; upstream safety constraints are not enforced."]
        if private_deals is not None:
            warnings.append("Private ranges approximated by a fixed seeded compatible-deal sample; only encountered holdings are solved.")
        if chance:
            warnings.append(f"Chance method: {chance_method}; depth={budget.max_depth}; continuation={baseline.version}.")
            if work.rollout_leaves:
                warnings.append("Unresolved branches use sampled baseline continuations; finite-forest optimization may overfit chance noise.")
        if len(seats) == 3:
            warnings.append("Multiplayer CFR has no general two-player Nash guarantee.")

        for iteration in range(budget.iterations):
            if work.nodes + 2 * tree_nodes > budget.max_nodes:
                if budget.strict or work.iterations == 0:
                    raise BudgetExceeded("Insufficient nodes for a complete CFR iteration and final EV evaluation")
                warnings.append("Iteration limit shortened to reserve final strategy EV evaluation.")
                break
            strategy = {}
            for key, obs in observations.items():
                if iteration == 0 and self.prior is not None:
                    from cfr_solver import validate_strategy
                    probabilities = self.prior.probabilities(obs)
                    validate_strategy(probabilities, obs.legal_mask)
                    strategy[key] = probabilities
                else:
                    strategy[key] = regret_matching(regrets[key], obs.legal_mask)
            changes = {key: [0.0] * len(ACTION_IDS) for key in observations}
            own_mass: dict[str, tuple[float, ...]] = {}

            def visit(node: TreeNode, reach: tuple[float, ...], chance: float) -> tuple[float, ...]:
                work.visit()
                if node.utilities is not None:
                    work.terminal_leaves += 1
                    return node.utilities
                p = seat_index[node.player]
                probabilities = strategy[node.key]
                values = []
                for action, child in zip(node.actions, node.children):
                    index = ACTION_IDS.index(action.action_id)
                    next_reach = list(reach)
                    next_reach[p] *= probabilities[index]
                    values.append(visit(child, tuple(next_reach), chance))
                expected = tuple(math.fsum(probabilities[ACTION_IDS.index(a.action_id)] * v[i]
                                          for a, v in zip(node.actions, values)) for i in range(len(seats)))
                counterfactual = chance * math.prod(r for i, r in enumerate(reach) if i != p)
                for action, value in zip(node.actions, values):
                    changes[node.key][ACTION_IDS.index(action.action_id)] += counterfactual * (value[p] - expected[p])
                contribution = tuple(reach[p] * probability for probability in probabilities)
                if node.key in own_mass and any(abs(a - b) > 1e-12 for a, b in zip(own_mass[node.key], contribution)):
                    raise ValueError("Information set has inconsistent own reach; perfect recall violated")
                own_mass[node.key] = contribution
                return expected

            try:
                for deal, tree in trees:
                    visit(tree, (1.0,) * len(seats), deal.probability)
            except BudgetExceeded:
                if budget.strict or work.iterations == 0:
                    raise
                warnings.append("Incomplete iteration discarded after budget exhaustion.")
                break
            for key in regrets:
                regrets[key] = [r + d for r, d in zip(regrets[key], changes[key])]
                masses[key] = [s + d for s, d in zip(masses[key], own_mass[key])]
            work.iterations += 1

        averages = {}
        for key, vector in masses.items():
            total = math.fsum(vector)
            # Unreached own prefixes have no defined average. Uniform is explicit.
            averages[key] = (tuple(v / total for v in vector) if total > 0
                             else regret_matching([0.0] * len(ACTION_IDS), observations[key].legal_mask))

        def evaluate(node: TreeNode) -> tuple[float, ...]:
            work.visit()
            if node.utilities is not None:
                return node.utilities
            values = [evaluate(child) for child in node.children]
            return tuple(math.fsum(averages[node.key][ACTION_IDS.index(a.action_id)] * v[i]
                                   for a, v in zip(node.actions, values)) for i in range(len(seats)))

        ev: dict[Combo, dict[str, float]] = {}
        hand_mass: dict[Combo, float] = {}
        strategies = {}
        utilities = {p: 0.0 for p in seats}
        for deal, tree in trees:
            hand = deal.hands[actor]
            hand_mass[hand] = hand_mass.get(hand, 0.0) + deal.probability
            strategies[hand] = {a.action_id: averages[tree.key][ACTION_IDS.index(a.action_id)] for a in tree.actions}
            destination = ev.setdefault(hand, {a.action_id: 0.0 for a in tree.actions})
            for action, child in zip(tree.actions, tree.children):
                values = evaluate(child)
                destination[action.action_id] += deal.probability * values[seat_index[actor]]
                for p in seats:
                    utilities[p] += deal.probability * strategies[hand][action.action_id] * values[seat_index[p]]
        for hand, values in ev.items():
            ev[hand] = {a: v / hand_mass[hand] for a, v in values.items()}
        return RiverSolution(LocallySolvedPolicy(averages, f"river-cfr-{work.iterations}"), strategies, ev,
                             {k: tuple(v) for k, v in regrets.items()}, {k: tuple(v) for k, v in masses.items()},
                             utilities, work, work.iterations, warnings)
