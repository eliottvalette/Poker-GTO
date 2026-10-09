"""Matched complete-hand evaluation; uncertainty is clustered by independent deal."""

import json
from pathlib import Path
import random
import time

from actions import ACTION_IDS, legal_actions
from cfr_solver import sample_index
from hybrid.beliefs import BeliefState
from hybrid.decision_engine import HybridDecisionEngine
from hybrid.policy_source import TabularPolicy, UniformLegalPolicy, distribution
from hybrid.quality import mean_interval
from hybrid.ranges import HandRange, JointRanges, combo
from hybrid.reference import ProfilePolicy
from hybrid.session import HybridSession, Hypothesis
from hybrid.state import BudgetExceeded, ComputeBudget, instantiate
from infoset import observe
from poker_game_expresso import HandState


def matched_hands(
    path: Path,
    *,
    deals: int = 16,
    samples: int = 8,
    profiles: tuple[str, ...] = (
        "tight_passive",
        "loose_passive",
        "tight_aggressive",
        "loose_aggressive",
        "push_fold",
    ),
) -> dict:
    rows = []
    for count in (2, 3):
        for profile in profiles:
            teacher = ProfilePolicy(profile)
            for seed in range(deals):
                rng = random.Random(81000 + seed)
                root = HandState.start(
                    {p: 3.0 + (seed + p) % 3 for p in range(count)}, seed % count, rng
                )
                deck = list(range(52))
                rng.shuffle(deck)
                factors = {
                    p: HandRange(
                        {
                            combo(tuple(deck[p * 4 : p * 4 + 2])): 1.0,
                            combo(tuple(deck[p * 4 + 2 : p * 4 + 4])): 1.0,
                        }
                    )
                    for p in root.players
                }
                belief = BeliefState(JointRanges(factors))
                root = instantiate(root, belief.public.sample(rng), rng)
                hero = seed % count
                for candidate in (
                    "uniform-search",
                    "reference-search",
                    "profile-search",
                ):
                    policies = dict.fromkeys(root.players, teacher)
                    session = HybridSession(
                        root,
                        (
                            Hypothesis(
                                "known evaluation population", 1.0, belief, policies
                            ),
                        ),
                    )
                    assumed = (
                        UniformLegalPolicy()
                        if candidate == "uniform-search"
                        else ProfilePolicy()
                        if candidate == "reference-search"
                        else teacher
                    )
                    continuation = dict.fromkeys(root.players, assumed)
                    continuation[hero] = ProfilePolicy()
                    engine = HybridDecisionEngine(continuation, search_mode="response")
                    budget = ComputeBudget(
                        samples=samples, max_depth=1, max_nodes=50000, seed=seed + 321
                    )
                    play_rng = random.Random(93000 + seed)
                    nodes = decisions = 0
                    started = time.perf_counter()
                    failure = None
                    try:
                        while not session.state.terminal:
                            if decisions >= 80:
                                raise BudgetExceeded("Complete-hand decision limit 80")
                            if session.state.current_player == hero:
                                table = {}
                                feasible = session.hypotheses[
                                    0
                                ].beliefs.public.marginals()[hero]
                                for hand in feasible.weights:
                                    view = session.state.clone()
                                    view.actor.cards = hand
                                    result = engine.analyze(
                                        view, session.hypotheses[0].beliefs, budget
                                    )
                                    nodes += result.search_nodes
                                    table[observe(view).key()] = tuple(
                                        result.action_probabilities.get(a, 0.0)
                                        for a in ACTION_IDS
                                    )
                                behavior = TabularPolicy(table, version=candidate)
                            else:
                                behavior = teacher
                            probabilities = distribution(behavior, session.state)
                            selected = ACTION_IDS[sample_index(probabilities, play_rng)]
                            action = next(
                                a
                                for a in legal_actions(session.state)
                                if a.action_id == selected
                            )
                            session.observe_action(
                                action,
                                likelihood_source=behavior.version,
                                external_policy=behavior,
                            )
                            decisions += 1
                    except (BudgetExceeded, ValueError, KeyError) as error:
                        failure = f"{type(error).__name__}: {error}"
                    rows.append(
                        {
                            "count": count,
                            "profile": profile,
                            "seed": seed,
                            "hero": hero,
                            "candidate": candidate,
                            "chip_ev_realized": None
                            if failure
                            else session.state.utility(hero),
                            "failure": failure,
                            "decisions": decisions,
                            "nodes": nodes,
                            "seconds": time.perf_counter() - started,
                        }
                    )
    paired = []
    for count in (2, 3):
        for profile in profiles:
            for candidate in ("reference-search", "profile-search"):
                differences = []
                for seed in range(deals):
                    group = {
                        r["candidate"]: r
                        for r in rows
                        if r["count"] == count
                        and r["profile"] == profile
                        and r["seed"] == seed
                    }
                    if (
                        not group[candidate]["failure"]
                        and not group["uniform-search"]["failure"]
                    ):
                        differences.append(
                            group[candidate]["chip_ev_realized"]
                            - group["uniform-search"]["chip_ev_realized"]
                        )
                paired.append(
                    {
                        "count": count,
                        "profile": profile,
                        "candidate": candidate,
                        "paired_chip_difference_vs_uniform": mean_interval(differences),
                    }
                )
    report = {
        "budget": {
            "deals_per_profile_count": deals,
            "samples_per_query": samples,
            "max_decisions": 80,
        },
        "scope": "Controlled disclosed narrow priors, matched private deals and action draws. Small samples do not certify poker strength.",
        "rows": rows,
        "paired": paired,
        "failure_rate": sum(bool(r["failure"]) for r in rows) / len(rows),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def tournament_matches(
    path: Path, *, tournaments: int = 16, max_hands: int = 80
) -> dict:
    from tournament import TournamentState

    rows = []
    for count in (2, 3):
        for seed in range(tournaments):
            game = TournamentState(
                stacks={p: 4.0 for p in range(count)},
                button=seed % count,
                rng=random.Random(seed + 611),
            )
            rng = random.Random(seed + 191)
            policies = {
                p: ProfilePolicy("conservative" if p == 0 else "loose_aggressive")
                for p in range(count)
            }
            hands = 0
            while not game.terminal and hands < max_hands:
                hand = game.start_hand()
                while not hand.terminal:
                    probabilities = distribution(policies[hand.current_player], hand)
                    game.act(ACTION_IDS[sample_index(probabilities, rng)])
                hands += 1
            rows.append(
                {
                    "count": count,
                    "seed": seed,
                    "hands": hands,
                    "winner": game.winner,
                    "censored": not game.terminal,
                    "final_stacks": {
                        p: player.stack for p, player in game.hand.players.items()
                    },
                }
            )
    report = {
        "budget": {"tournaments_per_count": tournaments, "max_hands": max_hands},
        "rows": rows,
        "objective": "reference remains hand_chip_delta; these tournament outcomes do not imply optimal tournament play",
        "win_rates": {
            count: mean_interval(
                [
                    float(r["winner"] == 0)
                    for r in rows
                    if r["count"] == count and not r["censored"]
                ]
            )
            for count in (2, 3)
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    return report
