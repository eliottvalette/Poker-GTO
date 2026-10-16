"""Exact terminal reference checks for bounded sequential response labels."""
from itertools import combinations
import json
from pathlib import Path

from actions import legal_actions
from hybrid.adaptive_labels import LabelBudget, generate
from hybrid.policy_source import UniformLegalPolicy
from hybrid.ranges import HandRange, JointRanges
from hybrid.solver_experiments import bounded_roots
from hybrid.state import instantiate
from training.checkpoint import atomic_bytes
from utils import rank7


def exact_terminal_check(path: Path, *, seeds: int = 12) -> dict:
    root, _ = list(bounded_roots(2, "test"))[-1]
    next(a for a in legal_actions(root) if a.action_id == "ALL_IN").apply(root)
    actor = root.current_player
    opponent = next(p for p in root.players if p != actor)
    hands = list(combinations([c for c in range(52) if c not in root.board], 2))
    selected = None
    for hero in hands:
        score = rank7((*hero, *root.board))
        strong = next((h for h in hands if not set(h).intersection(hero) and rank7((*h, *root.board)) > score), None)
        weak = next((h for h in hands if not set(h).intersection(hero) and rank7((*h, *root.board)) < score), None)
        if strong and weak:
            selected = hero, strong, weak
            break
    if selected is None:
        raise ValueError("Exact contrast needs winning and losing compatible holdings")
    hero, strong, weak = selected
    report = {"scope": "36 seeded terminal river checks; no broad statistical calibration claim", "cases": []}
    for win_probability in (.2, .5, .9):
        ranges = JointRanges({actor: HandRange({hero: 1.}), opponent: HandRange({strong: 1-win_probability, weak: win_probability})}, tuple(root.board))
        exact = {}
        for action in legal_actions(root):
            value = 0.
            for deal in ranges.enumerate():
                state = instantiate(root, deal.hands)
                action.apply(state)
                if not state.terminal:
                    raise ValueError("Reference requires a terminal action")
                value += deal.probability * state.utility(actor) / root.blinds.big
            exact[action.action_id] = value
        for seed in range(seeds):
            label = generate(root, ranges, hero, {p: UniformLegalPolicy() for p in root.players},
                             LabelBudget(looks=(32, 128, 512, 2048), tolerance_bb=.1, max_nodes=10000, seed=seed))
            violations = sum(not (pair["interval_bb"][0] <= exact[pair["actions"][0]] - exact[pair["actions"][1]] <= pair["interval_bb"][1])
                             for look in label["looks"] for pair in look["pairs"])
            estimate = label["estimate"]
            regret = max(exact.values()) - exact[estimate["selected_action"]]
            report["cases"].append({"win_probability": win_probability, "exact_ev_bb": exact,
                "interval_violations": violations, "selection_loss_bb": regret, "label": label})
    atomic_bytes(path, json.dumps(report, separators=(",", ":"), allow_nan=False).encode())
    return report
