# utils.py
# ------------------------------------------------------------
# Utilitaires pour l'évaluation des mains de poker
# ------------------------------------------------------------
from __future__ import annotations
from typing import Tuple
from collections import Counter
import itertools
from classes import card_rank, card_suit


def hand_rank_5(cards5: Tuple[int, int, int, int, int]) -> Tuple:
    """
    Retourne un tuple comparable (catégorie, bris d'égalité…) pour une main de 5 cartes.
    Catégories : 8=quinte flush, 7=carré, 6=full, 5=couleur, 4=quinte, 3=brelan, 2=deux paires, 1=une paire, 0=hauteur
    """
    ranks = sorted([card_rank(c) for c in cards5], reverse=True)
    suits = [card_suit(c) for c in cards5]
    rank_counts = Counter(ranks)
    counts_sorted = sorted(rank_counts.items(), key=lambda x: (x[1], x[0]), reverse=True)  # d'abord par multiplicité puis par rang
    is_flush = max(Counter(suits).values()) == 5

    # Détection de la quinte (wheel A-5 gérée)
    uniq = sorted(set(ranks), reverse=True)
    def straight_high(uniq_ranks):
        if 14 in uniq_ranks:
            uniq_ranks = uniq_ranks + [1]  # A traité aussi comme 1
        for i in range(len(uniq_ranks)-4):
            window = uniq_ranks[i:i+5]
            if all(window[k] - 1 == window[k+1] for k in range(4)):
                return window[0] if window[0] != 1 else 5
        return None
    straight_hi = straight_high(uniq)
    is_straight = straight_hi is not None

    if is_straight and is_flush:
        return (8, straight_hi)  # quinte flush
    # Carré
    if counts_sorted[0][1] == 4:
        four = counts_sorted[0][0]
        kicker = max([r for r in ranks if r != four])
        return (7, four, kicker)
    # Full
    if counts_sorted[0][1] == 3 and counts_sorted[1][1] == 2:
        trips = counts_sorted[0][0]
        pair = counts_sorted[1][0]
        return (6, trips, pair)
    # Couleur
    if is_flush:
        return (5, ) + tuple(ranks)
    # Quinte
    if is_straight:
        return (4, straight_hi)
    # Brelan
    if counts_sorted[0][1] == 3:
        trips = counts_sorted[0][0]
        kickers = [r for r in ranks if r != trips][:2]
        return (3, trips) + tuple(kickers)
    # Deux paires
    if counts_sorted[0][1] == 2 and counts_sorted[1][1] == 2:
        hi_pair = max(counts_sorted[0][0], counts_sorted[1][0])
        lo_pair = min(counts_sorted[0][0], counts_sorted[1][0])
        kicker = max([r for r in ranks if r != hi_pair and r != lo_pair])
        return (2, hi_pair, lo_pair, kicker)
    # Une paire
    if counts_sorted[0][1] == 2:
        pair = counts_sorted[0][0]
        kickers = [r for r in ranks if r != pair][:3]
        return (1, pair) + tuple(kickers)
    # Hauteur
    return (0, ) + tuple(ranks)


def best5_of7_rank(cards7: Tuple[int, ...]) -> Tuple:
    """
    Trouve la meilleure main de 5 cartes parmi 7 cartes et retourne son rang.
    Utilise itertools.combinations pour tester toutes les combinaisons possibles.
    """
    best = None
    for comb in itertools.combinations(cards7, 5):
        r = hand_rank_5(comb)
        if (best is None) or (r > best):
            best = r
    return best
