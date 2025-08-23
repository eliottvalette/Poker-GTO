# utils.py
# ------------------------------------------------------------
# Utilitaires pour l'évaluation des mains de poker
# ------------------------------------------------------------
from __future__ import annotations
from classes import card_rank, card_suit
import json
from typing import Dict

from treys import Card as TCard, Evaluator as TEvaluator
_TREYS_EVAL = TEvaluator()
_RANK_CHARS = "23456789TJQKA"   # 2..A
_SUIT_CHARS = "shdc"            # ♠, ♥, ♦, ♣ (0..3)

def _to_treys_int(c: int) -> int:
    """Convertit ton entier (0..51) en int Treys."""
    r = card_rank(c)  # 2..14
    s = card_suit(c)  # 0..3
    return TCard.new(_RANK_CHARS[r-2] + _SUIT_CHARS[s])

def rank7(cards7: tuple[int, ...]) -> int:
    """
    Retourne un score utilisable pour comparer deux mains.
    Plus GRAND = meilleure main.
    """
    h1, h2, *board = cards7
    t_board = [_to_treys_int(c) for c in board]
    t_hand  = [_to_treys_int(h1), _to_treys_int(h2)]
    score = _TREYS_EVAL.evaluate(t_board, t_hand)  # plus PETIT = meilleur
    return -score  # inversé pour garder compatibilité avec tes comparaisons


def save_ranges_json(path: str, ranges: Dict[str, list]):
    """
    Sauvegarde les ranges au format JSON.
    Les combos sont stockés comme des listes de tuples (int, int).
    """

    ranges_for_json = {
        "BTN_shove": list(ranges["BTN shove"]),
        "SB_call_vs_BTN": list(ranges["SB call vs BTN"]),
        "BB_call_vs_BTN": list(ranges["BB call vs BTN"]),
        "SB_shove": list(ranges["SB shove"]),
        "BB_call_vs_SB": list(ranges["BB call vs SB"])
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(ranges_for_json, f, indent=4)

def load_ranges_json(path: str) -> Dict[str, set]:
    """
    Charge les ranges depuis un fichier JSON.
    Retourne un dictionnaire avec les noms des ranges et leurs combos convertis en sets.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Convertir les listes en sets de tuples
    ranges = {}
    for name, combo_list in data.items():
        ranges[name] = set(tuple(combo) for combo in combo_list)
    
    return ranges
