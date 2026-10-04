"""Small explicit tournament smoke benchmarks; scripted win rates are not GTO evidence."""
from __future__ import annotations
import random
from typing import Callable
from actions import SolverAction, legal_actions
from cfr_solver import sample_index
from infoset import observe
from tournament import TournamentState

BOT_NAMES = ("random", "calling_station", "nit", "aggro", "shove_fold")


def scripted_action(name: str, hand, rng: random.Random) -> SolverAction:
    actions = legal_actions(hand)
    if name not in BOT_NAMES:
        raise ValueError(f"Unknown scripted bot {name}; expected {BOT_NAMES}")
    if name == "random":
        return rng.choice(actions)
    if name == "calling_station":
        preferences = ("CHECK", "CALL")
    elif name == "nit":
        preferences = ("CHECK", "FOLD", "CALL")
    elif name == "aggro":
        raises = [a for a in actions if a.category == "RAISE"]
        if raises:
            return rng.choice(raises)
        preferences = ("CHECK", "CALL", "FOLD")
    else:
        # All-in calls use CALL, including short-stack calls. A shove/fold bot
        # must sometimes call a jam rather than endlessly trading blind folds.
        names = {a.action_id for a in actions}
        if "ALL_IN" not in names and "CALL" in names:
            choice = "CALL" if rng.random() < 0.5 else "FOLD"
            return next(a for a in actions if a.action_id == choice)
        preferences = ("ALL_IN", "CHECK", "FOLD", "CALL")
    for name in preferences:
        for action in actions:
            if action.action_id == name:
                return action
    raise ValueError(f"Scripted preferences found no legal action: {actions}")


def benchmark_tournaments(policy: Callable, tournaments: int, seed: int,
                          opponents: tuple[str, str] = ("random", "calling_station"),
                          max_decisions: int = 10000) -> dict:
    if tournaments < 1 or max_decisions < 1:
        raise ValueError("Explicit positive tournament count and decision budget required")
    rng = random.Random(seed)
    wins = hands = decisions = 0
    for _ in range(tournaments):
        t = TournamentState(rng=random.Random(rng.randrange(2**31)))
        t.start_hand()
        count = 0
        while not t.terminal:
            if t.hand.terminal:
                t.start_hand()
                continue
            count += 1
            if count > max_decisions:
                raise RuntimeError(f"Tournament benchmark decision budget exceeded: {count}/{max_decisions}")
            if t.current_player == 0:
                from actions import ACTION_IDS
                from cfr_solver import validate_strategy
                o = observe(t)
                distribution = policy(o)
                validate_strategy(distribution, o.legal_mask)
                t.act(ACTION_IDS[sample_index(distribution, rng)])
            else:
                scripted_action(opponents[t.current_player - 1], t.hand, rng).apply(t.hand)
        wins += t.winner == 0
        hands += t.hand_number
        decisions += count
    return {"tournaments": tournaments, "hero_wins": wins, "hands": hands, "decisions": decisions,
            "starting_stack_bb": 25.0, "total_chips_bb": 75.0, "opponents": opponents}
