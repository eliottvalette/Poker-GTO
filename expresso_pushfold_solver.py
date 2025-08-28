# expresso_pushfold_solver.py

"""
GTO : Expresso Push/Fold Solver
"""

from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple, Dict, Set, Iterable, Optional
import tqdm
import time
import os
import cProfile
import random
import math

# =========================
# Profiler
# =========================
PROFILE = False

# =========================
# Import des classes et utilitaires
# =========================
from classes import (
    DECK, ALL_COMBOS, combo_to_169
)

from utils import rank7, save_ranges_json, load_ranges_json

from visualisation_push_fold import visualise_ranges

# ==== Précompute pour filtrage et comptage rapide ====
ALL_COMBOS_SET: Set[Tuple[int,int]] = set(ALL_COMBOS)  # Ensemble de tous les combos possibles
COMBOS_BY_CARD: Dict[int, frozenset[Tuple[int,int]]] = {}  # Dictionnaire carte -> combos contenant cette carte
for card in range(52):  # Pour chaque carte du deck
    list_of_combos_with_card = []  # Liste des combos contenant la carte
    for card_1,card_2 in ALL_COMBOS:  # Pour chaque combo possible
        if card_1 == card or card_2 == card:  # Si la carte est dans le combo
            list_of_combos_with_card.append((card_1,card_2))  # Ajouter le combo à la liste
    COMBOS_BY_CARD[card] = frozenset((min(card_1,card_2), max(card_1,card_2)) for card_1,card_2 in list_of_combos_with_card)  # Ensemble des combos contenant la carte

TOTAL_COMBOS_NO_BLOCKERS = 1225  # C(50,2) = 1225 combos sans blockers

# Statistiques pour l'approche adaptative
ADAPTIVE_STATS = {
    'total_ev_calculations': 0,
    'total_samples_used': 0,
    'total_samples_saved': 0,
    'early_stops': 0,
    'max_samples_reached': 0
}

# Paramètres d'hystérésis pour stabilité
ADD_EPS = 0.02   # ajoute si EV > +0.02 bb
DROP_EPS = 0.01  # retire si EV < -0.01 bb

def keep_or_flip(prev_has: bool, ev: float) -> bool:
    """Hystérésis pour éviter les changements dus au bruit Monte-Carlo"""
    if ev > ADD_EPS:  return True
    if ev < -DROP_EPS: return False
    return prev_has

def fast_filter_range(range_set, blocked: Set[int]) -> List[Tuple[int,int]]:  # Filtrer les combos contenant des cartes bloquées
    card_1,card_2 = list(blocked)  # Obtenir les cartes bloquées (les deux cartes du hero)
    range_set = set(range_set) if not isinstance(range_set, set) else range_set  # Convertir la range en set si nécessaire
    return list(range_set - COMBOS_BY_CARD[card_1] - COMBOS_BY_CARD[card_2])  # Retourner la range filtrée (sans les combos contenant les cartes bloquées) (en soustrayant les sets on est en O(1))

# =======================
# Monte Carlo d'équités adaptatif avec arrêt précoce
# =======================
class EquityCache:  # Cache pour les calculs d'équité
    def __init__(self):
        self.cache2 = {}  # Cache: ((h1a,h1b), frozenset(villain_range_ids), eff), mc_samples -> (p_win,p_tie)

    @staticmethod
    def combo_norm(combo: Tuple[int,int]) -> Tuple[int,int]:  # Normaliser l'ordre des cartes dans un combo
        card_1,card_2 = combo  # Extraire les deux cartes
        return (card_1,card_2) if card_1<card_2 else (card_2,card_1)  # Retourner dans l'ordre croissant

def sample_board(excluded: set[int], rng: random.Random) -> tuple[int,int,int,int,int]:
    used = set(excluded) # Ensemble des cartes utilisées
    b0 = []
    while len(b0) < 5: # Tirage par rejet :
        c = rng.randrange(52)
        if c in used:
            continue
        used.add(c)
        b0.append(c)
    return tuple(b0)

def q_adaptive(
    hero_combo: tuple[int,int],
    villain_list: list[tuple[int,int]],
    rng: random.Random,
    tau: float,                 # seuil = E / (pot + 2E)
    batch: int,
    alpha: float,        # ~99% de confiance
    max_samples: int,
) -> tuple[float, int]:
    """
    Monte-Carlo adaptatif (early-stop) pour estimer si l'EV d'un all-in 2-way est positive ou négative.

    Idée clé :
    L'EV d'un all-in 2-way s'écrit :
        EV = -E + (pot + 2E) * q
    où q = p_win + 0.5 * p_tie, et le signe de l'EV ne dépend que de q et du seuil tau = E / (pot + 2E).

    On veut donc savoir si q > tau (EV>0) ou q < tau (EV<0).

    Plutôt que d'échantillonner un nombre fixe de boards, on procède par paquets (batch) et on s'arrête dès qu'on sait de quel côté du seuil tau on est, avec une confiance 1-alpha.
    À chaque board, la variable X vaut 1 (win), 0.5 (tie), ou 0 (loss), donc bornée dans [0,1].
    Notons q_hat l'estimation empirique de q.
    Par l'inégalité de Hoeffding, avec n échantillons, l'intervalle de confiance est :
        |q_hat - q| <= sqrt(ln(2/alpha)/(2n))
    Dès que [q_hat - eps_n, q_hat + eps_n] ne recoupe plus tau, on a la décision à confiance >= 1-alpha.

    Retourne (q_hat, n)
        - q_hat : estimation empirique de q
        - n     : nombre d'échantillons utilisés
    """
    card_1, card_2 = hero_combo
    wins = ties = losses = 0
    nb_samples = 0
    log_term = math.log(2.0/alpha)

    # On suppose villain_list déjà filtrée 
    if not villain_list:
        return 1.0, 0  # personne ne call → q=1

    while nb_samples < max_samples:
        # Échantillonner un batch
        for _ in range(batch):
            villain_card_1, villain_card_2 = rng.choice(villain_list)
            board = sample_board({card_1,card_2,villain_card_1,villain_card_2}, rng)
            hero_rank = rank7((card_1, card_2) + board)
            villain_rank = rank7((villain_card_1, villain_card_2) + board)
            if hero_rank > villain_rank: 
                wins += 1
            elif hero_rank == villain_rank: 
                ties += 1
            else: 
                losses += 1
        nb_samples = wins+ties+losses

        q_hat = (wins + 0.5*ties)/nb_samples
        eps = math.sqrt(log_term/(2.0*nb_samples))   # Hoeffding

        if q_hat - eps > tau:
            # Arrêt précoce - décision prise
            ADAPTIVE_STATS['early_stops'] += 1
            ADAPTIVE_STATS['total_samples_saved'] += max_samples - nb_samples
            return q_hat, nb_samples
        if q_hat + eps < tau:
            # Arrêt précoce - décision prise
            ADAPTIVE_STATS['early_stops'] += 1
            ADAPTIVE_STATS['total_samples_saved'] += max_samples - nb_samples
            return q_hat, nb_samples

    # Pas de décision tranchée -> max_samples atteint
    ADAPTIVE_STATS['max_samples_reached'] += 1
    q_hat = (wins + 0.5*ties)/nb_samples
    return q_hat, nb_samples

# ===========================
# Configuration Expresso
# ===========================
@dataclass
class ExpressoConfig:  # Configuration du solveur push/fold
    sb: float = 0.5  # Petite blinde en BB
    bb: float = 1.0  # Grosse blinde en BB
    stacks_bb: Tuple[float,float,float] = (25.0, 25.0, 25.0)  # Stacks (BTN,SB,BB) en BB
    mc_samples: int = 1600  # Nombre d'échantillons Monte Carlo
    seed: int = 42  # Graine pour la reproductibilité
    mc_batch: int = 80  # Nombre de boards par échantillon
    mc_alpha: float = 0.01  # Seuil de confiance pour l'arrêt précoce

# ===========================
# Ranges (ensembles de combos)
# ===========================
def combos_to_set(combos: Iterable[Tuple[int,int]]) -> Set[Tuple[int,int]]:  # Convertir une liste de combos en set
    return set((a,b) if a<b else (b,a) for a,b in combos)  # Normaliser l'ordre des cartes

def all_combos_set() -> Set[Tuple[int,int]]:  # Obtenir l'ensemble de tous les combos possibles
    return combos_to_set(ALL_COMBOS)  # Convertir ALL_COMBOS en set normalisé

# ===========================
# EV all-in / nœuds de jeu
# ===========================
class NodeEV:
    def __init__(self, config: ExpressoConfig):
        self.config = config
        self.rng = random.Random(config.seed)
        self.context = None

    def set_context(self, stacks: Tuple[float,float,float]):
        self.context = self.pot_and_behind(stacks)

    def context_pot_and_behind(self):
        return self.context

    def pot_and_behind(self, stacks: Tuple[float,float,float]) -> Tuple[float, Tuple[float,float,float]]:  # Calculer le pot et les stacks après blindes
        # Stacks en BB (avant blindes). On passe en "behind" après blindes postées.
        BTN_STACK, SB_STACK, BB_STACK = stacks  # Extraire les stacks des trois positions
        pot = self.config.sb + self.config.bb  # Calculer le pot total (SB + BB)
        behind = (BTN_STACK, SB_STACK - self.config.sb, BB_STACK - self.config.bb)  # Stacks après avoir posté les blindes

        if min(behind) < 0.0:
            raise ValueError("Stacks négatifs")
        
        return pot, behind  # Retourner le pot et les stacks "behind"

    def ev_allin_heads_up(self, hero_combo: Tuple[int,int], villain_list: List[Tuple[int,int]],
                       behind_hero: float, behind_vill: float, pot: float) -> float:
        """
        cEV pour HERO sur un all-in à 2 joueurs (post-blinds).
        Utilise l'estimateur adaptatif q_adaptive pour un arrêt précoce.
        """
        effective_stack = min(behind_hero, behind_vill)
        if effective_stack <= 0.0 :
            raise ValueError("Un des deux joueurs a un stack négatif ou nul")
        if not villain_list:
            return 0.0
        
        pot_final = pot + 2.0 * effective_stack
        tau = effective_stack / pot_final  # Seuil critique q = E/(pot + 2E)
        
        # Utiliser l'estimateur adaptatif au lieu de Monte Carlo fixe
        q_hat, nb_samples = q_adaptive(
            hero_combo=hero_combo,
            villain_list=villain_list,
            rng=self.rng,
            tau=tau,
            batch=self.config.mc_batch,
            alpha=self.config.mc_alpha,
            max_samples=self.config.mc_samples,
        )
        
        # Collecter les statistiques
        ADAPTIVE_STATS['total_ev_calculations'] += 1
        ADAPTIVE_STATS['total_samples_used'] += nb_samples
        
        # EV = -E + pot_final * q
        ev = -effective_stack + pot_final * q_hat
        return ev

    # ---- EV des décisions ----
    def ev_btn_shove(self, hero_combo: Tuple[int,int],
                     sb_call_range: Set[Tuple[int,int]],
                     bb_call_range: Set[Tuple[int,int]]) -> float:
        """
        EV (cEV) du shove BTN. Référence fold=0 (post-blinds).
        Branches : SB call / SB fold & BB call / SB fold & BB fold.
        """
        pot, (bBTN, bSB, bBB) = self.context_pot_and_behind()
        blocked = {hero_combo[0], hero_combo[1]}

        sb_call = fast_filter_range(sb_call_range, blocked) # Liste des combos que la SB pourrait call
        bb_call = fast_filter_range(bb_call_range, blocked) # Liste des combos que la BB pourrait call

        p_sb_call = len(sb_call) / TOTAL_COMBOS_NO_BLOCKERS # Probabilité que la SB call
        p_bb_call = len(bb_call) / TOTAL_COMBOS_NO_BLOCKERS # Probabilité que la BB call

        ev_vs_sb = self.ev_allin_heads_up(hero_combo, sb_call, bBTN, bSB, pot) # EV vs la SB
        ev_vs_bb = self.ev_allin_heads_up(hero_combo, bb_call, bBTN, bBB, pot) # EV vs la BB
        ev_steal = pot # EV si le BTN fold

        # EV Total = probabilité que la SB call * EV vs la SB + probabilité que la SB fold * (probabilité que la BB call * EV vs la BB + probabilité que la BB fold * EV si le BTN fold)
        ev = p_sb_call * ev_vs_sb + (1 - p_sb_call) * (p_bb_call * ev_vs_bb + (1 - p_bb_call) * ev_steal)
        return ev

    def ev_call_vs_btn(self, hero_combo: Tuple[int,int], btn_shove_range: Set[Tuple[int,int]], hero_pos: str) -> float:
        """
        EV pour CALL face au shove du BTN (hero_pos ∈ {"SB","BB"}). Fold = 0.
        """
        pot, (bBTN, bSB, bBB) = self.context_pot_and_behind()
        blocked = {hero_combo[0], hero_combo[1]}

        btn_range = fast_filter_range(btn_shove_range, blocked) # Liste des combos que le BTN pourrait shove

        if not btn_range: # Si aucun combo compatible
            return 0.0
        
        hero_stack = bSB if hero_pos == "SB" else bBB
        return self.ev_allin_heads_up(hero_combo, btn_range, hero_stack, bBTN, pot) # EV vs le BTN
    
    def ev_sb_shove(self, hero_combo: Tuple[int,int], bb_call_range: Set[Tuple[int,int]]) -> float:
        """
        BTN a fold ; SB décide shove vs BB. Fold = 0.
        """
        pot, (_, bSB, bBB) = self.context_pot_and_behind()
        blocked = {hero_combo[0], hero_combo[1]}

        bb_call = fast_filter_range(bb_call_range, blocked) # Liste des combos que la BB pourrait call
        p_bb_call = len(bb_call) / TOTAL_COMBOS_NO_BLOCKERS # Probabilité que la BB call

        ev_vs_bb = self.ev_allin_heads_up(hero_combo, bb_call, bSB, bBB, pot) # EV vs la BB
        ev_steal = pot # EV si le SB fold

        # EV Total = probabilité que la BB call * EV vs la BB + probabilité que la BB fold * EV si le SB fold
        ev = p_bb_call * ev_vs_bb + (1 - p_bb_call) * ev_steal
        return ev

    def ev_call_vs_sb(self, hero_combo: Tuple[int,int], sb_shove_range: Set[Tuple[int,int]]) -> float:
        """
        BB face au shove du SB (BTN a fold). Fold = 0.
        """
        pot, (_, bSB, bBB) = self.context_pot_and_behind()
        blocked = {hero_combo[0], hero_combo[1]}
        sb_range = fast_filter_range(sb_shove_range, blocked) # Liste des combos que le SB pourrait shove
        if not sb_range: # Si aucun combo compatible
            return 0.0
        return self.ev_allin_heads_up(hero_combo, sb_range, bBB, bSB, pot) # EV vs le SB

# ======================
# Solveur push/fold 3-max
# ======================
class SpinGoPushFoldSolver:
    def __init__(self, config: ExpressoConfig, saved_ranges: Dict[str, Set[Tuple[int,int]]] = None):
        self.config = config
        self.node = NodeEV(config)
        self.BTN_shove: Set[Tuple[int,int]] = saved_ranges.get("BTN_shove", set())
        self.SB_call_vs_BTN: Set[Tuple[int,int]] = saved_ranges.get("SB_call_vs_BTN", set())
        self.BB_call_vs_BTN: Set[Tuple[int,int]] = saved_ranges.get("BB_call_vs_BTN", set())
        self.SB_shove: Set[Tuple[int,int]] = saved_ranges.get("SB_shove", set())
        self.BB_call_vs_SB: Set[Tuple[int,int]] = saved_ranges.get("BB_call_vs_SB", set())

        self.all_combos = all_combos_set()
        self.rng = random.Random(config.seed)

    def iterate(self, n_iters: int = 8) -> None:
        stacks = self.config.stacks_bb
        print(f"\nDÉMARRAGE DES ITÉRATIONS ({n_iters})")
        print(f"Stacks : BTN={stacks[0]}bb, SB={stacks[1]}bb, BB={stacks[2]}bb")
        print(f"Samples Monte Carlo : {self.config.mc_samples}")
        print(f"Total des combos évalués : {len(self.all_combos)}")
        
        # Stocker l'évolution pour le graphique
        self.evolution_data = {
            'BTN_shove': [], 'SB_call_vs_BTN': [], 'BB_call_vs_BTN': [],
            'SB_shove': [], 'BB_call_vs_SB': []
        }
        
        # Ajouter l'état initial (itération 0)
        self.evolution_data['BTN_shove'].append(len(self.BTN_shove))
        self.evolution_data['SB_call_vs_BTN'].append(len(self.SB_call_vs_BTN))
        self.evolution_data['BB_call_vs_BTN'].append(len(self.BB_call_vs_BTN))
        self.evolution_data['SB_shove'].append(len(self.SB_shove))
        self.evolution_data['BB_call_vs_SB'].append(len(self.BB_call_vs_SB))
        
        # Définir le contexte une seule fois par itération
        self.node.set_context(stacks)
        
        for it in range(1, n_iters+1):
            print(f"\n{'='*70}")
            print(f"ITÉRATION {it}/{n_iters}")
            print(f"{'='*70}")

            start_time = time.time()

            # Snapshot de départ (état avant modifications)
            snap_BTN = self.BTN_shove.copy()
            snap_SBc = self.SB_call_vs_BTN.copy()
            snap_BBc = self.BB_call_vs_BTN.copy()
            snap_SBs = self.SB_shove.copy()
            snap_BBvsSB = self.BB_call_vs_SB.copy()

            # Garde pour l'affichage des changements
            self.previous_ranges = {
                "BTN_shove": snap_BTN,
                "SB_call_vs_BTN": snap_SBc,
                "BB_call_vs_BTN": snap_BBc,
                "SB_shove": snap_SBs,
                "BB_call_vs_SB": snap_BBvsSB,
            }

            # Compute-only depuis le snapshot (pas d'écriture pendant le calcul)
            print(f"\nCalcul des nouvelles ranges...")
            new_SBc = self.compute_sb_call_vs_btn(snap_BTN)
            new_BBc = self.compute_bb_call_vs_btn(snap_BTN)
            new_BBvsSB = self.compute_bb_call_vs_sb(snap_SBs)
            new_BTN = self.compute_btn_shove(snap_SBc, snap_BBc)
            new_SBs = self.compute_sb_shove(snap_BBvsSB)

            # Commit en bloc (mises à jour synchrones)
            print(f"Application des changements...")
            self.SB_call_vs_BTN = new_SBc
            self.BB_call_vs_BTN = new_BBc
            self.BB_call_vs_SB = new_BBvsSB
            self.BTN_shove = new_BTN
            self.SB_shove = new_SBs
            
            dt = time.time() - start_time
            
            # Stocker les données d'évolution
            self.evolution_data['BTN_shove'].append(len(self.BTN_shove))
            self.evolution_data['SB_call_vs_BTN'].append(len(self.SB_call_vs_BTN))
            self.evolution_data['BB_call_vs_BTN'].append(len(self.BB_call_vs_BTN))
            self.evolution_data['SB_shove'].append(len(self.SB_shove))
            self.evolution_data['BB_call_vs_SB'].append(len(self.BB_call_vs_SB))
            
            print(f"\nRÉSUMÉ ITÉRATION {it} :")
            print(f"Durée : {dt:.2f}s")
            print(f"Modifications :")
            c1 = len(new_SBc) != len(snap_SBc)
            c2 = len(new_BBc) != len(snap_BBc)
            c3 = len(new_BBvsSB) != len(snap_BBvsSB)
            c4 = len(new_BTN) != len(snap_BTN)
            c5 = len(new_SBs) != len(snap_SBs)
            print(f"   SB_call_vs_BTN: {'OUI' if c1 else 'NON'}")
            print(f"   BB_call_vs_BTN: {'OUI' if c2 else 'NON'}")
            print(f"   BB_call_vs_SB:  {'OUI' if c3 else 'NON'}")
            print(f"   BTN_shove:      {'OUI' if c4 else 'NON'}")
            print(f"   SB_shove:       {'OUI' if c5 else 'NON'}")
            
            print(f"\nTAILLES ACTUELLES :")
            print(f"   BTN shove      : {len(self.BTN_shove):4d} combos ({self.coverage_pct(self.BTN_shove):5.1f}%)")
            print(f"   SB call vs BTN : {len(self.SB_call_vs_BTN):4d} combos ({self.coverage_pct(self.SB_call_vs_BTN):5.1f}%)")
            print(f"   BB call vs BTN : {len(self.BB_call_vs_BTN):4d} combos ({self.coverage_pct(self.BB_call_vs_BTN):5.1f}%)")
            print(f"   SB shove       : {len(self.SB_shove):4d} combos ({self.coverage_pct(self.SB_shove):5.1f}%)")
            print(f"   BB call vs SB  : {len(self.BB_call_vs_SB):4d} combos ({self.coverage_pct(self.BB_call_vs_SB):5.1f}%)")
            
            total_changes = sum([c1, c2, c3, c4, c5])
            if total_changes == 0:
                print(f"\nCONVERGENCE atteinte à l'itération {it} !")
                break

            self.display_summary(iter_num=it)

    # ----- Affichage / résumé en 169 -----
    @staticmethod
    def summarize_169(combos_set: Set[Tuple[int,int]]) -> Dict[str, int]:
        d = Counter()
        for a,b in combos_set:
            lab = combo_to_169(a,b)
            d[lab] += 1
        return dict(sorted(d.items(), key=lambda x: (-x[1], x[0])))

    @staticmethod
    def coverage_pct(combos_set: Set[Tuple[int,int]]) -> float:
        return 100.0 * len(combos_set) / len(ALL_COMBOS)

    # Méthodes compute-only pour mises à jour synchrones
    def compute_sb_call_vs_btn(self, prev_BTN: Set[Tuple[int, int]]) -> Set[Tuple[int, int]]:
        """Calcule la nouvelle range SB call vs BTN sans modifier l'état"""
        new_set = set()
        for hero_combo in tqdm.tqdm(self.all_combos, desc="SB call vs BTN", leave=False):
            ev = self.node.ev_call_vs_btn(hero_combo, prev_BTN, "SB")
            has = hero_combo in self.SB_call_vs_BTN
            if keep_or_flip(has, ev):
                new_set.add(hero_combo)
        return new_set

    def compute_bb_call_vs_btn(self, prev_BTN: Set[Tuple[int, int]]) -> Set[Tuple[int, int]]:
        """Calcule la nouvelle range BB call vs BTN sans modifier l'état"""
        new_set = set()
        for hero_combo in tqdm.tqdm(self.all_combos, desc="BB call vs BTN", leave=False):
            ev = self.node.ev_call_vs_btn(hero_combo, prev_BTN, "BB")
            has = hero_combo in self.BB_call_vs_BTN
            if keep_or_flip(has, ev):
                new_set.add(hero_combo)
        return new_set

    def compute_bb_call_vs_sb(self, prev_SB: Set[Tuple[int, int]]) -> Set[Tuple[int, int]]:
        """Calcule la nouvelle range BB call vs SB sans modifier l'état"""
        new_set = set()
        for hero_combo in tqdm.tqdm(self.all_combos, desc="BB call vs SB", leave=False):
            ev = self.node.ev_call_vs_sb(hero_combo, prev_SB)
            has = hero_combo in self.BB_call_vs_SB
            if keep_or_flip(has, ev):
                new_set.add(hero_combo)
        return new_set

    def compute_btn_shove(self, prev_SBc: Set[Tuple[int, int]], prev_BBc: Set[Tuple[int, int]]) -> Set[Tuple[int, int]]:
        """Calcule la nouvelle range BTN shove sans modifier l'état"""
        new_set = set()
        for hero_combo in tqdm.tqdm(self.all_combos, desc="BTN shove", leave=False):
            ev = self.node.ev_btn_shove(hero_combo, prev_SBc, prev_BBc)
            has = hero_combo in self.BTN_shove
            if keep_or_flip(has, ev):
                new_set.add(hero_combo)
        return new_set

    def compute_sb_shove(self, prev_BB: Set[Tuple[int, int]]) -> Set[Tuple[int, int]]:
        """Calcule la nouvelle range SB shove sans modifier l'état"""
        new_set = set()
        for hero_combo in tqdm.tqdm(self.all_combos, desc="SB shove", leave=False):
            ev = self.node.ev_sb_shove(hero_combo, prev_BB)
            has = hero_combo in self.SB_shove
            if keep_or_flip(has, ev):
                new_set.add(hero_combo)
        return new_set

    def display_summary(self, iter_num: int) -> None:
        """Affiche un résumé clair avec visualisations et sauvegarde des PNGs"""
        # Données pour les graphiques
        ranges_data = {
            "BTN_shove": self.BTN_shove,
            "SB_call_vs_BTN": self.SB_call_vs_BTN,
            "BB_call_vs_BTN": self.BB_call_vs_BTN,
            "SB_shove": self.SB_shove,
            "BB_call_vs_SB": self.BB_call_vs_SB
        }

        # Visualisations
        visualise_ranges(ranges_data, self.coverage_pct, iter_num, self.evolution_data)

        # Sauvegarde des ranges
        if iter_num == 0:
            os.makedirs('ranges', exist_ok=True)
            save_ranges_json('ranges/ranges.json', ranges_data)
        else:
            os.makedirs(f'ranges', exist_ok=True)
            save_ranges_json(f'ranges/ranges_{iter_num}.json', ranges_data)

        # Print texte clair
        print("\n" + "="*80)
        print("RÉSUMÉ FINAL DES RANGES PUSH/FOLD")
        print("="*80)
        
        print(f"\nCOUVERTURES DES RANGES:")
        print("-" * 50)
        for name, combo_set in ranges_data.items():
            coverage = self.coverage_pct(combo_set)
            print(f"{name:20} : {len(combo_set):4d} combos ({coverage:5.1f}%)")
        
        if iter_num > 0:
            print(f"\nCHANGEMENTS PAR RANGE (vs itération précédente):")
            print("-" * 50)
            for name, combo_set in ranges_data.items():
                prev_set = self.previous_ranges[name]
                before = len(prev_set)
                after = len(combo_set)
                difference = after - before
                print(f"{name:20} : {before:4d} → {after:4d} ({difference:+4d})")
        
        # Créer un fichier de données pour analyse
        with open('ranges/ranges_data.txt', 'w') as f:
            f.write("DONNÉES DÉTAILLÉES DES RANGES PUSH/FOLD\n")
            f.write(f"="*50 + "\n\n")
            for name, combo_set in ranges_data.items():
                f.write(f"{name}:\n")
                f.write(f"  Couverture: {self.coverage_pct(combo_set):.1f}%\n")
                f.write(f"  Nombre de combos: {len(combo_set)}\n")
                f.write("  Top combos:\n")
                top_combos = list(self.summarize_169(combo_set).items())[:10]
                for combo, count in top_combos:
                    f.write(f"    {combo}: {count}\n")
                f.write("\n")
        
        # Afficher les statistiques de l'approche adaptative
        if ADAPTIVE_STATS['total_ev_calculations'] > 0:
            print(f"\nSTATISTIQUES APPROCHE ADAPTATIVE:")
            print("-" * 50)
            total_fixed_samples = ADAPTIVE_STATS['total_ev_calculations'] * self.config.mc_samples
            efficiency = 100.0 * ADAPTIVE_STATS['total_samples_used'] / total_fixed_samples
            print(f"Calculs EV effectués     : {ADAPTIVE_STATS['total_ev_calculations']}")
            print(f"Échantillons utilisés    : {ADAPTIVE_STATS['total_samples_used']}")
            print(f"Échantillons économisés  : {ADAPTIVE_STATS['total_samples_saved']}")
            print(f"Arrêts précoces          : {ADAPTIVE_STATS['early_stops']}")
            print(f"Max samples atteint      : {ADAPTIVE_STATS['max_samples_reached']}")
            print(f"Efficacité               : {efficiency:.1f}%")
            print(f"Accélération             : {100.0/efficiency:.1f}x")

# ======================
# Démonstration
# ======================
if __name__ == "__main__":
    config = ExpressoConfig(
        sb=0.5, bb=1.0,
        stacks_bb=(25.0, 25.0, 25.0),  # (BTN, SB, BB)
        mc_samples=400,
        seed=42
    )

    saved_ranges = load_ranges_json("ranges/ranges.json")
    solver = SpinGoPushFoldSolver(config, saved_ranges)

    if PROFILE:
        profiler = cProfile.Profile()
        profiler.enable()

    solver.iterate(n_iters=15)

    if PROFILE:
        profiler.disable()
        profiler.dump_stats("profiling/training_profile.prof")

    solver.display_summary(iter_num=0)

    