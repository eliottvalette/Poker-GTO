"""Fixed public probes and deterministic conditional subgames; no full-game claims."""
from __future__ import annotations
import random
import torch
from actions import ACTION_IDS, legal_actions
from cfr_solver import sample_index, validate_strategy
from evaluation import subgame_best_response
from infoset import observe
from ml.deep_cfr import TraversalTask, generate_samples
from ml.model import encode_batch
from ml.train import evaluate_loss
from poker_game_expresso import BlindLevel, HandState
from scripts.benchmark_policy import BOT_NAMES, scripted_action

EVALUATION_VERSION = 1


def fixed_roots(count: int) -> list[tuple[str, HandState]]:
    cases = [("balanced_deep", [25, 25, 25] if count == 3 else [37.5, 37.5], 1, "PREFLOP"),
             ("asymmetric", [1, 9, 65] if count == 3 else [5, 70], 2, "PREFLOP"),
             ("shallow", [25, 25, 25] if count == 3 else [37.5, 37.5], 16, "PREFLOP"),
             ("postflop", [25, 25, 25] if count == 3 else [37.5, 37.5], 1, "FLOP"),
             ("turn", [25, 25, 25] if count == 3 else [37.5, 37.5], 1, "TURN"),
             ("river_tractable", [2] * count, 1, "RIVER")]
    roots = []
    for index, (name, stacks, big, street) in enumerate(cases):
        hand = HandState.start(dict(enumerate(stacks)), index % count, random.Random(907 + index), BlindLevel(big / 2, big))
        while hand.street != street and not hand.terminal:
            hand.act("CALL" if hand.to_call() else "CHECK")
        if hand.terminal:
            raise ValueError(f"Fixed evaluation fixture terminated unexpectedly: {name}")
        roots.append((name, hand))
    return roots


def model_policy(model, obs) -> tuple[float, ...]:
    return model.probabilities(obs)


def fixed_drift(previous, current, roots: list[tuple[str, HandState]]) -> float | None:
    if previous is None:
        return None
    with torch.no_grad():
        batch = encode_batch([observe(h) for _, h in roots])
        return float((current(batch) - previous(batch)).abs().sum(1).mean())


def evaluate_solver(solver, roots: list[tuple[str, HandState]], max_nodes: int, batch_size: int, reference_snapshot) -> dict:
    river = roots[-1][1]
    # Multiple hidden deals share hero's information. Remaining deck/runout is
    # fixed in this river suite; this is conditional bounded BR, not exploitability.
    hero = river.current_player
    deals = []
    for index in range(4):
        hand = river.clone()
        other = next(i for i in hand.players if i != hero)
        if index:
            player = hand.players[other]
            offset = (index - 1) * 2
            replacements = hand.deck[offset:offset + 2]
            hand.deck[offset:offset + 2] = list(player.cards)
            player.cards = tuple(replacements)
        hand.assert_invariants()
        deals.append((0.25, hand))
    strategy = lambda obs: model_policy(solver.average_model, obs)
    br = subgame_best_response(deals, strategy, hero, max_nodes=max_nodes)
    br["scope"] = "fixed_river_weighted_hidden_deals"
    # Independent seeds and fixed roots, never inserted into training reservoirs.
    heldout_advantages = {p: [] for p in solver.players}
    heldout_strategies = []
    for index, (_, hand) in enumerate(roots):
        for player in solver.players:
            result = generate_samples(reference_snapshot, TraversalTask(index * len(solver.players) + player, player,
                                      hand, 30101 + index * 10 + player, max_nodes, 64,
                                      traversal_mode=solver.traversal_mode,
                                      epsilon=solver.metrics[-1]["epsilon"] if solver.traversal_mode == "outcome_sampling" else 0.6))
            heldout_advantages[player].extend(result.advantages)
            heldout_strategies.extend(result.strategies)
    losses = {str(p): evaluate_loss(solver.advantage_models[p], rows, batch_size) for p, rows in heldout_advantages.items()}
    average_loss = evaluate_loss(solver.average_model, heldout_strategies, batch_size)
    scripted = {}
    for bot in BOT_NAMES:
        utilities = []
        for index, (_, root) in enumerate(roots[:3]):
            hand = root.clone()
            hero = index % len(solver.players)
            rng = random.Random(50001 + index)
            decisions = 0
            while not hand.terminal:
                decisions += 1
                if decisions > 128:
                    raise RuntimeError(f"Scripted hand budget exceeded: bot={bot}")
                if hand.current_player == hero:
                    action = ACTION_IDS[sample_index(strategy(observe(hand)), rng)]
                    next(a for a in legal_actions(hand) if a.action_id == action).apply(hand)
                else:
                    scripted_action(bot, hand, rng).apply(hand)
            utilities.append(hand.utility(hero))
        scripted[bot] = {"hand_chip_deltas": utilities, "mean": sum(utilities) / len(utilities), "scope": "three_fixed_hands_sanity_only"}
    return {"version": EVALUATION_VERSION, "bounded_best_response": br,
            "independent_advantage_loss": losses, "independent_average_policy_loss": average_loss,
            "scripted_opponents": scripted}
