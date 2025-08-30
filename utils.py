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

# Listes globales réutilisées pour éviter les allocations
_EVAL_HAND_BUFFER = [0, 0]          # buffer réutilisé pour 2 cartes
_EVAL_BOARD_BUFFER = [0, 0, 0, 0, 0]  # buffer réutilisé pour 5 cartes

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
    hero1, hero2, board0, board1, board2, board3, board4 = cards7

    # Conversion via LUT dans buffers réutilisés
    _EVAL_HAND_BUFFER[0] = TREYS_INT_LUT[hero1]
    _EVAL_HAND_BUFFER[1] = TREYS_INT_LUT[hero2]

    _EVAL_BOARD_BUFFER[0] = TREYS_INT_LUT[board0]
    _EVAL_BOARD_BUFFER[1] = TREYS_INT_LUT[board1]
    _EVAL_BOARD_BUFFER[2] = TREYS_INT_LUT[board2]
    _EVAL_BOARD_BUFFER[3] = TREYS_INT_LUT[board3]
    _EVAL_BOARD_BUFFER[4] = TREYS_INT_LUT[board4]

    # Treys: plus PETIT = meilleur → on renvoie l’opposé
    return -_TREYS_EVAL.evaluate(_EVAL_BOARD_BUFFER, _EVAL_HAND_BUFFER)

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
