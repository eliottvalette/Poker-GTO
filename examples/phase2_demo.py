"""Persistent public beliefs and profile-sensitive computed decisions."""

from dataclasses import asdict
import json
from pathlib import Path
import random

from actions import SolverAction, legal_actions
from hybrid.beliefs import BeliefState
from hybrid.ranges import HandRange, JointRanges, combo
from hybrid.reference import ProfilePolicy
from hybrid.session import HybridSession, Hypothesis
from hybrid.state import ComputeBudget
from infoset import observe
from poker_game_expresso import HandState


def demonstrate(count: int = 2, *, fixture_path: Path | None = None) -> dict:
    rng = random.Random(171 + count)
    state = HandState.start({p: 5.0 + p for p in range(count)}, 0, rng)
    ranges = {}
    for p, player in state.players.items():
        rows = {combo(player.cards): 2.0}
        for _ in range(4):
            rows[combo(tuple(rng.sample(range(52), 2)))] = 1.0
        ranges[p] = HandRange(rows)
    profiles = dict.fromkeys(state.players, ProfilePolicy("loose_passive"))
    session = HybridSession(
        state,
        (Hypothesis("loose-passive", 1.0, BeliefState(JointRanges(ranges)), profiles),),
    )

    def transport(hand):
        raw = asdict(hand)
        raw["pending"] = sorted(hand.pending)
        return raw

    def range_rows(beliefs):
        return {
            p: [{"cards": h, "probability": w} for h, w in r.weights.items()]
            for p, r in beliefs.public.ranges.items()
        }

    result = {
        "player_count": count,
        "initial": transport(state),
        "ranges": range_rows(session.hypotheses[0].beliefs),
        "profiles": dict.fromkeys(state.players, "loose_passive"),
        "steps": [],
    }
    while not session.state.terminal:
        before = transport(session.state)
        likelihoods = profiles[session.state.current_player].probabilities(
            observe(session.state)
        )
        analysis = session.analyze(
            ComputeBudget(samples=8, max_depth=1, max_nodes=50000), mode="exploitative"
        )
        action = (
            SolverAction("actual", "RAISE", 2.25)
            if not session.events
            else next(
                a
                for a in legal_actions(session.state)
                if a.category == ("CALL" if session.state.to_call() else "CHECK")
            )
        )
        session.observe_action(
            action,
            likelihood_source="declared synthetic passive behavior",
            interpolate=True,
        )
        result["steps"].append(
            {
                "before": before,
                "after": transport(session.state),
                "action": asdict(action),
                "likelihoods": likelihoods,
                "posterior": range_rows(session.hypotheses[0].beliefs),
                "analysis": analysis.to_dict(),
            }
        )
    result["utilities"] = {p: session.state.utility(p) for p in state.players}
    if fixture_path:
        fixture_path.parent.mkdir(parents=True, exist_ok=True)
        fixture_path.write_text(json.dumps(result, indent=2) + "\n")
    return result
