"""Generate bounded deterministic Python fixtures for browser-engine parity.

Import ``generate_fixtures`` or ``save_fixtures`` from test tooling. No runtime
browser dependency on Python is introduced.
"""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import random

from actions import ACTION_IDS, legal_actions
from blind_schedule import BlindSchedule, BlindStage
from infoset import NUMERIC_NAMES, observe
from poker_game_expresso import BlindLevel, HandState
from tournament import TournamentState
from utils import rank7


def snapshot(state: HandState | TournamentState) -> dict:
    hand = state if isinstance(state, HandState) else state.hand
    if hand is None:
        raise ValueError("Parity snapshot requires a started hand")
    result = {
        "players": [asdict(p) for p in hand.players.values()],
        "button": hand.button, "hand_number": hand.hand_number,
        "blind_level_index": hand.blind_level_index, "blinds": asdict(hand.blinds), "board": hand.board.copy(), "street": hand.street,
        "pot": hand.pot, "highest": hand.highest, "last_full_raise": hand.last_full_raise,
        "pending": sorted(hand.pending), "current_player": hand.current_player,
        "terminal": hand.terminal, "showdown": hand.showdown,
        "awards": dict(hand.awards), "history": [asdict(e) for e in hand.history],
        "legal_actions": [asdict(a) for a in legal_actions(hand)],
        "observation": None if hand.terminal else asdict(observe(state)),
    }
    if isinstance(state, TournamentState):
        result["tournament"] = {"button": state.button, "hand_number": state.hand_number,
                                "active": list(state.active), "terminal": state.terminal,
                                "winner": state.winner, "total_chips": state.total_chips,
                                "blind_level_index": state.blind_level_index, "blinds": asdict(state.blinds)}
    return result


def arranged_deck(private: list[int], board: list[int]) -> list[int]:
    return ([c for c in range(52) if c not in private + board]
            + list(reversed(board)) + list(reversed(private)))


def fixture(name: str, stacks: list[float], button: int = 0,
            actions: list[tuple[str, float | None]] | None = None,
            deck: list[int] | None = None, seed: int = 7,
            random_actions: bool = False, blinds: BlindLevel = BlindLevel(),
            hand_number: int = 1, blind_level_index: int = 0) -> dict:
    rng = random.Random(seed)
    if deck is None:
        deck = list(range(52))
        rng.shuffle(deck)
    h = HandState.start(dict(enumerate(stacks)), button, rng, blinds, deck=deck,
                        hand_number=hand_number, blind_level_index=blind_level_index)
    initial = snapshot(h)
    steps = []
    for category, amount in actions or []:
        h.act(category, amount)
        steps.append({"action": {"category": category, "amount_to": amount}, "expected": snapshot(h)})
    for _ in range(120):
        if h.terminal:
            break
        selected = rng.choice(legal_actions(h)) if random_actions else None
        category = selected.category if selected else "CALL" if h.to_call() else "CHECK"
        amount = selected.amount_to if selected else None
        h.act(category, amount)
        steps.append({"action": {"category": category, "amount_to": amount}, "expected": snapshot(h)})
    if not h.terminal:
        raise ValueError(f"Parity fixture {name} exceeded its 120-action budget")
    return {"name": name, "stacks": stacks, "button": button, "deck": deck,
            "blinds": asdict(blinds), "hand_number": hand_number, "blind_level_index": blind_level_index,
            "initial": initial, "steps": steps}


def tournament_fixture(button: int, eliminated: int) -> dict:
    board = [8, 17, 26, 35, 40]
    survivors = [i for i in range(3) if i != eliminated]
    holes = {eliminated: [0, 1], survivors[0]: [48, 49], survivors[1]: [44, 45]}
    private = [c for i in range(3) for c in holes[i]]
    stacks = [1.0 if i == eliminated else 25.0 for i in range(3)]
    t = TournamentState(dict(enumerate(stacks)), button=button, rng=random.Random(0),
                        blind_schedule=BlindSchedule.fixed())
    decks = [arranged_deck(private, board)]
    steps = []
    for hand_no in range(3):
        if t.terminal:
            break
        if hand_no:
            deck = list(range(52))
            random.Random(70 + hand_no).shuffle(deck)
            decks.append(deck)
        t.start_hand(deck=decks[-1])
        steps.append({"operation": "start", "deck": decks[-1], "expected": snapshot(t)})
        for _ in range(50):
            if t.hand.terminal:
                break
            category = "CALL" if t.hand.to_call() else "CHECK"
            t.hand.act(category)
            steps.append({"operation": "action", "action": {"category": category, "amount_to": None},
                          "expected": snapshot(t)})
        if not t.hand.terminal:
            raise ValueError("Tournament parity fixture exceeded its hand action budget")
    return {"name": f"hu_button_{button}_eliminated_{eliminated}", "stacks": stacks,
            "button": button, "schedule": [{"first_hand": 1, "blinds": asdict(BlindLevel())}], "steps": steps}


def progression_fixture() -> dict:
    schedule = BlindSchedule((BlindStage(1, BlindLevel()), BlindStage(2, BlindLevel(1, 2)),
                              BlindStage(3, BlindLevel(2, 4))))
    t = TournamentState(rng=random.Random(0), blind_schedule=schedule)
    steps = []
    for number in range(1, 5):
        deck = list(range(52))
        random.Random(number).shuffle(deck)
        t.start_hand(deck=deck)
        steps.append({"operation": "start", "deck": deck, "expected": snapshot(t)})
        t.hand.act("FOLD")
        steps.append({"operation": "action", "action": {"category": "FOLD", "amount_to": None},
                      "expected": snapshot(t)})
        t.hand.act("FOLD")
        steps.append({"operation": "action", "action": {"category": "FOLD", "amount_to": None},
                      "expected": snapshot(t)})
    return {"name": "progressive_blinds_current_hand_context", "stacks": [25, 25, 25], "button": 0,
            "schedule": [asdict(stage) for stage in schedule.stages], "steps": steps}


def generate_fixtures() -> dict:
    cases = [
        fixture("arbitrary_reraises_and_folds", [25, 25, 25], actions=[("RAISE", 2), ("RAISE", 3.7), ("RAISE", 5.4), ("FOLD", None), ("FOLD", None)]),
        fixture("short_allin_closed", [25, 3, 25], actions=[("RAISE", 2.5), ("RAISE", 3), ("CALL", None)]),
        fixture("full_allin_reopens", [25, 5, 25], actions=[("RAISE", 2.5), ("RAISE", 5), ("CALL", None)]),
        fixture("unacted_short_allin_rights", [25, 1.4, 25], actions=[("CALL", None), ("RAISE", 1.4), ("RAISE", 2.4)]),
        fixture("uncalled_refund", [25, 2, 3], actions=[("RAISE", 25), ("CALL", None), ("CALL", None)]),
        fixture("sidepot_exact_awards", [20, 10, 5], actions=[("RAISE", 20), ("CALL", None), ("CALL", None)],
                deck=arranged_deck([40, 41, 44, 45, 48, 49], [0, 5, 22, 31, 36])),
        fixture("folded_contribution", [25, 25, 25], actions=[("RAISE", 3), ("CALL", None), ("RAISE", 8), ("FOLD", None), ("CALL", None)]),
        fixture("royal_board_split", [1, 1, 1], deck=arranged_deck([0, 1, 4, 5, 8, 9], [32, 36, 40, 44, 48])),
        fixture("short_call", [1, 25]),
        fixture("hu_short_allin_bb_below_sb", [25, 0.25]),
        fixture("hu_dry_pot_nominal_blind_cap", [25, 20], blinds=BlindLevel(16, 32),
                hand_number=51, blind_level_index=5),
        fixture("three_handed_short_bb_nominal_call", [25, 25, 0.25]),
        fixture("raised_blind_current_bb", [25, 25, 25], blinds=BlindLevel(1, 2),
                hand_number=11, blind_level_index=1),
        fixture("checked_short_allin", [25, 25, 1.4], actions=[("CALL", None), ("CALL", None), ("CHECK", None), ("CHECK", None), ("RAISE", 0.4), ("CALL", None)]),
    ]
    for count in (2, 3):
        for button in range(count):
            cases.append(fixture(f"order_{count}_{button}", [25] * count, button))
    for seed in range(18):
        rng = random.Random(seed)
        count = 2 + seed % 2
        cases.append(fixture(f"random_{seed}", [round(rng.uniform(0.2, 25), 4) for _ in range(count)],
                             button=seed % count, seed=seed, random_actions=True))
    rng = random.Random(101)
    evaluations = []
    for _ in range(100):
        cards = rng.sample(range(52), 9)
        a, b = cards[:7], cards[2:7] + cards[7:]
        ra, rb = rank7(tuple(a)), rank7(tuple(b))
        evaluations.append({"a": a, "b": b, "comparison": (ra > rb) - (ra < rb)})
    # One explicit representative of every 7-card rank category, strongest first.
    category_cards = [
        [32, 36, 40, 44, 48, 1, 6],
        [48, 49, 50, 51, 44, 4, 9],
        [48, 49, 50, 44, 45, 4, 9],
        [48, 40, 32, 20, 4, 9, 14],
        [0, 5, 10, 15, 16, 40, 45],
        [48, 49, 50, 40, 33, 4, 9],
        [48, 49, 44, 45, 32, 4, 9],
        [48, 49, 40, 33, 22, 4, 9],
        [48, 41, 34, 23, 16, 5, 10],
    ]
    ranks = [rank7(tuple(cards)) for cards in category_cards]
    if not all(a > b for a, b in zip(ranks, ranks[1:])):
        raise ValueError(f"Category parity representatives out of order: {ranks}")
    return {"version": 1, "category_cards": category_cards, "actions": list(ACTION_IDS), "numeric_names": list(NUMERIC_NAMES),
            "hands": cases, "tournaments": [tournament_fixture(b, e) for b in range(3) for e in range(3)] + [progression_fixture()],
            "evaluations": evaluations}


def save_fixtures(path: Path = Path(__file__).parent / "fixtures" / "browser_parity.json") -> None:
    data = generate_fixtures()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, separators=(",", ":"), allow_nan=False) + "\n")
