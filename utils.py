# utils.py
# ------------------------------------------------------------
# Utilitaires pour l'évaluation des mains de poker (version LUT Treys)
# ------------------------------------------------------------
from __future__ import annotations
import json
from typing import Dict, Tuple

# --- Treys (évaluateur poker ultra-rapide) ---
from treys import Card as TCard, Evaluator as TEvaluator

# Évaluateur global (objet Treys)
_TREYS_EVAL = TEvaluator()

# LUT (Lookup Table) 52 -> int Treys
# Construite une seule fois au chargement du module.
_RANK_CHARS = "23456789TJQKA"   # 2..A
_SUIT_CHARS = "shdc"            # ♠, ♥, ♦, ♣ (0..3)

def build_treys_lut() -> Tuple[int, ...]:
    lut = [0] * 52
    for c in range(52):
        r = (c // 4) + 2      # 2..14
        s = c % 4             # 0..3
        lut[c] = TCard.new(_RANK_CHARS[r-2] + _SUIT_CHARS[s])
    return tuple(lut)

TREYS_INT_LUT: Tuple[int, ...] = build_treys_lut()

# --------- API d'évaluation ----------
def rank7(cards7: tuple[int, ...]) -> int:
    """
    Évalue 7 cartes (2 main + 5 board) via Treys.
    Retourne un entier où *plus GRAND = meilleur* (on inverse le score Treys).
    """
    # Déplie sans allocations inutiles
    h1, h2, b0, b1, b2, b3, b4 = cards7

    # Conversion via LUT (aucun appel à Card.new / card_rank / card_suit)
    t_hand  = [TREYS_INT_LUT[h1], TREYS_INT_LUT[h2]]
    t_board = [TREYS_INT_LUT[b0], TREYS_INT_LUT[b1], TREYS_INT_LUT[b2],
               TREYS_INT_LUT[b3], TREYS_INT_LUT[b4]]

    # Treys: plus PETIT = meilleur → on renvoie l'opposé pour rester compatible
    score = _TREYS_EVAL.evaluate(t_board, t_hand)
    return -score



# --------- Sauvegarde / chargement des ranges ----------
def save_ranges_json(path: str, ranges: Dict[str, list]):
    """
    Sauvegarde les ranges au format JSON.
    Les combos sont stockés comme des listes de tuples (int, int).
    """
    ranges_for_json = {
        "BTN_shove":       list(ranges["BTN_shove"]),
        "SB_call_vs_BTN":  list(ranges["SB_call_vs_BTN"]),
        "BB_call_vs_BTN":  list(ranges["BB_call_vs_BTN"]),
        "SB_shove":        list(ranges["SB_shove"]),
        "BB_call_vs_SB":   list(ranges["BB_call_vs_SB"]),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(ranges_for_json, f, indent=4)


def load_ranges_json(path: str) -> Dict[str, set]:
    """
    Charge les ranges depuis un fichier JSON.
    Retourne un dict {nom_range: set[(int,int), ...]}.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ranges = {}
    for name, combo_list in data.items():
        ranges[name] = set(tuple(combo) for combo in combo_list)
    return ranges
