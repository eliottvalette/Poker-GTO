# expresso_pushfold_solver.py
# ------------------------------------------------------------
# Spin&Go (3-handed) — Solveur PUSH/FOLD
#
# Objet : calculer des ranges préflop 3-max en mode push/fold, en cEV
# (pas d’ICM dans le moteur), sans overcalls (pas de all-in à 3).
#
# Positions : BTN (pas de blinde), SB (poste SB), BB (poste BB)
# Séquence :
#   - BTN : {FOLD, SHOVE}
#   - Si BTN shove : SB {FOLD, CALL} ; si SB fold → BB {FOLD, CALL}
#   - Si BTN fold : SB {FOLD, SHOVE} ; si SB shove → BB {FOLD, CALL}
#
# Méthode :
#   - Itérations de meilleures réponses sur 5 ranges :
#       BTN_shove, SB_call_vs_BTN, BB_call_vs_BTN, SB_shove, BB_call_vs_SB
#   - Équités all-in via Monte Carlo (éval 7 cartes -> meilleure main 5 cartes)
#   - EV en jetons (cEV) en se plaçant "post-blinds" (fold = 0 EV local)
#
# Hypothèses/simplifications :
#   - Pas d’overcall quand BTN shove et SB call (pas de 3-way all-in)
#   - Échantillonnage uniforme des combos dans les ranges adverses
#   - Blockers pris en compte en filtrant les combos incompatibles
#   - cEV uniquement (ICM en post-traitement si souhaité)
#
# Sorties :
#   - Ranges push/call stables pour la profondeur donnée
#   - Résumés en notation 169 (A2..AKs) et couverture en %
#
# ------------------------------------------------------------
from __future__ import annotations
import random, math, itertools
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import List, Tuple, Dict, Set, Iterable, Optional
import tqdm
import time


# =========================
# Import des classes et utilitaires
# =========================
from classes import (
    Card, Deck, DECK, ALL_COMBOS, 
    card, card_rank, card_suit, combo_to_169, filter_combos_excluding
)

# ================
# Éval mains 7->5
# ================
def hand_rank_5(cards5: Tuple[int,int,int,int,int]) -> Tuple:
    # Retourne un tuple comparable (catégorie, bris d’égalité…), cat: 8..0
    ranks = sorted([card_rank(c) for c in cards5], reverse=True)
    suits = [card_suit(c) for c in cards5]
    rank_counts = Counter(ranks)
    counts_sorted = sorted(rank_counts.items(), key=lambda x: (x[1], x[0]), reverse=True)  # d’abord par multiplicité puis par rang
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

def best5_of7_rank(cards7: Tuple[int,...]) -> Tuple:
    best = None
    for comb in itertools.combinations(cards7, 5):
        r = hand_rank_5(comb)
        if (best is None) or (r > best):
            best = r
    return best

# =======================
# Monte Carlo d’équités
# =======================
class EquityCache:
    def __init__(self):
        self.cache2 = {}  # ((h1a,h1b), frozenset(villain_range_ids), eff), mc_samples -> (p_win,p_tie)

    @staticmethod
    def combo_norm(c):
        a,b = c
        return (a,b) if a<b else (b,a)

def sample_board(excluded: Set[int], rng: random.Random) -> Tuple[int,int,int,int,int]:
    pool = [c for c in DECK if c not in excluded]
    rng.shuffle(pool)
    return tuple(pool[:5])

def estimate_equity_vs_range(hero_combo: Tuple[int,int],
                             villain_range: List[Tuple[int,int]],
                             mc_samples: int = 400,
                             rng: Optional[random.Random] = None) -> Tuple[float,float,float]:
    """
    Retourne (p_win, p_tie, p_lose) pour hero vs *une* main tirée uniformément de villain_range.
    Monte Carlo sur boards.
    """
    rng = rng or random.Random(0)
    wins = ties = losses = 0
    ha, hb = hero_combo
    blocked = {ha,hb}
    # Pré-sélection pour tenir compte des blockers
    vill_list = [c for c in villain_range if c[0] not in blocked and c[1] not in blocked]
    if not vill_list:
        return (1.0, 0.0, 0.0)  # Personne ne peut call → cas traité dans l’EV macro
    for _ in range(mc_samples):
        va, vb = rng.choice(vill_list)
        used = {ha,hb,va,vb}
        board = sample_board(used, rng)
        hero_rank = best5_of7_rank((ha,hb)+board)
        vill_rank = best5_of7_rank((va,vb)+board)
        if hero_rank > vill_rank:
            wins += 1
        elif hero_rank == vill_rank:
            ties += 1
        else:
            losses += 1
    n = wins+ties+losses
    return (wins/n, ties/n, losses/n)

# ===========================
# Configuration Expresso
# ===========================
@dataclass
class ExpressoConfig:
    sb: float = 0.5
    bb: float = 1.0
    stacks_bb: Tuple[float,float,float] = (25.0, 25.0, 25.0)  # (BTN,SB,BB) en BB
    mc_samples: int = 400
    seed: int = 42

# ===========================
# Ranges (ensembles de combos)
# ===========================
def combos_to_set(combos: Iterable[Tuple[int,int]]) -> Set[Tuple[int,int]]:
    return set((a,b) if a<b else (b,a) for a,b in combos)

def all_combos_set() -> Set[Tuple[int,int]]:
    return combos_to_set(ALL_COMBOS)

# ===========================
# EV all-in / nœuds de jeu
# ===========================
class NodeEV:
    def __init__(self, cfg: ExpressoConfig):
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)

    def _pot_and_behind(self, stacks: Tuple[float,float,float]) -> Tuple[float, Tuple[float,float,float]]:
        # Stacks en BB (avant blindes). On passe en "behind" après blindes postées.
        BTN, SB, BB = stacks
        pot = self.cfg.sb + self.cfg.bb
        behind = (BTN, SB - self.cfg.sb, BB - self.cfg.bb)
        return pot, behind

    @staticmethod
    def _eff(behind_hero: float, behind_vill: float) -> float:
        return max(0.0, min(behind_hero, behind_vill))

    def _ev_allin_2way(self, hero_combo: Tuple[int,int], villain_range: List[Tuple[int,int]],
                       behind_hero: float, behind_vill: float, pot: float) -> float:
        """
        cEV pour HERO sur un all-in à 2 joueurs (post-blinds).
        """
        E = self._eff(behind_hero, behind_vill)
        if E <= 0.0:
            return 0.0
        p_win, p_tie, p_lose = estimate_equity_vs_range(hero_combo, villain_range, self.cfg.mc_samples, self.rng)
        # pot_final = pot + 2E ; EV = p_win*(-E + pot_final) + p_tie*(-E + 0.5*pot_final) + p_lose*(-E)
        pot_final = pot + 2.0 * E
        ev = p_win * (-E + pot_final) + p_tie * (-E + 0.5 * pot_final) + p_lose * (-E)
        return ev

    # ---- EV des décisions ----
    def ev_btn_shove(self, hero_combo: Tuple[int,int],
                     stacks: Tuple[float,float,float],
                     sb_call_range: Set[Tuple[int,int]],
                     bb_call_range: Set[Tuple[int,int]]) -> float:
        """
        EV (cEV) du shove BTN. Référence fold=0 (post-blinds).
        Branches : SB call / SB fold & BB call / SB fold & BB fold.
        """
        pot, (bBTN, bSB, bBB) = self._pot_and_behind(stacks)
        blocked = {hero_combo[0], hero_combo[1]}
        sb_all = filter_combos_excluding(ALL_COMBOS, blocked)
        sb_call = filter_combos_excluding(list(sb_call_range), blocked)
        bb_call = filter_combos_excluding(list(bb_call_range), blocked)

        p_sb_call = len(sb_call) / len(sb_all) if sb_all else 0.0
        bb_all = filter_combos_excluding(ALL_COMBOS, blocked)
        p_bb_call = len(bb_call) / len(bb_all) if bb_all else 0.0

        ev_vs_sb = self._ev_allin_2way(hero_combo, sb_call, bBTN, bSB, pot)
        ev_vs_bb = self._ev_allin_2way(hero_combo, bb_call, bBTN, bBB, pot)
        ev_steal = pot

        ev = p_sb_call * ev_vs_sb + (1 - p_sb_call) * (p_bb_call * ev_vs_bb + (1 - p_bb_call) * ev_steal)
        return ev

    def ev_call_vs_btn(self, hero_combo: Tuple[int,int], stacks: Tuple[float,float,float],
                        btn_shove_range: Set[Tuple[int,int]], hero_pos: str) -> float:
        """
        EV pour CALL face au shove du BTN (hero_pos ∈ {"SB","BB"}). Fold = 0.
        """
        assert hero_pos in ("SB","BB")
        pot, (bBTN, bSB, bBB) = self._pot_and_behind(stacks)
        blocked = {hero_combo[0], hero_combo[1]}
        btn_range = filter_combos_excluding(list(btn_shove_range), blocked)
        if not btn_range:
            return 0.0
        if hero_pos == "SB":
            return self._ev_allin_2way(hero_combo, btn_range, bSB, bBTN, pot)
        else:
            # BB : atteint seulement si SB a fold
            return self._ev_allin_2way(hero_combo, btn_range, bBB, bBTN, pot)

    def ev_sb_shove(self, hero_combo: Tuple[int,int], stacks: Tuple[float,float,float],
                    bb_call_range: Set[Tuple[int,int]]) -> float:
        """
        BTN a fold ; SB décide shove vs BB. Fold = 0.
        """
        pot, (bBTN, bSB, bBB) = self._pot_and_behind(stacks)
        blocked = {hero_combo[0], hero_combo[1]}
        bb_all = filter_combos_excluding(ALL_COMBOS, blocked)
        bb_call = filter_combos_excluding(list(bb_call_range), blocked)
        p_bb_call = len(bb_call) / len(bb_all) if bb_all else 0.0

        ev_vs_bb = self._ev_allin_2way(hero_combo, bb_call, bSB, bBB, pot)
        ev_steal = pot
        ev = p_bb_call * ev_vs_bb + (1 - p_bb_call) * ev_steal
        return ev

    def ev_call_vs_sb(self, hero_combo: Tuple[int,int], stacks: Tuple[float,float,float],
                      sb_shove_range: Set[Tuple[int,int]]) -> float:
        """
        BB face au shove du SB (BTN a fold). Fold = 0.
        """
        pot, (bBTN, bSB, bBB) = self._pot_and_behind(stacks)
        blocked = {hero_combo[0], hero_combo[1]}
        sb_range = filter_combos_excluding(list(sb_shove_range), blocked)
        if not sb_range:
            return 0.0
        return self._ev_allin_2way(hero_combo, sb_range, bBB, bSB, pot)

# ======================
# Solveur push/fold 3-max
# ======================
class SpinGoPushFoldSolver:
    def __init__(self, cfg: ExpressoConfig):
        self.cfg = cfg
        self.node = NodeEV(cfg)
        # Ranges (ensembles de combos)
        self.BTN_shove: Set[Tuple[int,int]] = set()
        self.SB_call_vs_BTN: Set[Tuple[int,int]] = set()
        self.BB_call_vs_BTN: Set[Tuple[int,int]] = set()
        self.SB_shove: Set[Tuple[int,int]] = set()
        self.BB_call_vs_SB: Set[Tuple[int,int]] = set()

        self._all = all_combos_set()
        self.rng = random.Random(cfg.seed)

    def _update_btn_shove(self, stacks):
        print(f"\nMise à jour : range BTN shove…")
        new_set = set()
        for c in tqdm.tqdm(self._all, desc="BTN shove", leave=False):
            ev = self.node.ev_btn_shove(c, stacks, self.SB_call_vs_BTN, self.BB_call_vs_BTN)
            if ev > 0.0:
                new_set.add(c)
        changed = (len(new_set ^ self.BTN_shove) > 0)
        self.BTN_shove = new_set
        print(f"BTN shove : {len(new_set)} combos (modifié : {changed})")
        return changed

    def _update_sb_call_vs_btn(self, stacks):
        print(f"\nMise à jour : range SB call vs BTN…")
        new_set = set()
        for c in tqdm.tqdm(self._all, desc="SB call vs BTN", leave=False):
            ev_call = self.node.ev_call_vs_btn(c, stacks, self.BTN_shove, "SB")
            if ev_call > 0.0:
                new_set.add(c)
        changed = (len(new_set ^ self.SB_call_vs_BTN) > 0)
        self.SB_call_vs_BTN = new_set
        print(f"SB call vs BTN : {len(new_set)} combos (modifié : {changed})")
        return changed

    def _update_bb_call_vs_btn(self, stacks):
        print(f"\nMise à jour : range BB call vs BTN…")
        new_set = set()
        for c in tqdm.tqdm(self._all, desc="BB call vs BTN", leave=False):
            ev_call = self.node.ev_call_vs_btn(c, stacks, self.BTN_shove, "BB")
            if ev_call > 0.0:
                new_set.add(c)
        changed = (len(new_set ^ self.BB_call_vs_BTN) > 0)
        self.BB_call_vs_BTN = new_set
        print(f"BB call vs BTN : {len(new_set)} combos (modifié : {changed})")
        return changed

    def _update_sb_shove(self, stacks):
        print(f"\nMise à jour : range SB shove…")
        new_set = set()
        for c in tqdm.tqdm(self._all, desc="SB shove", leave=False):
            ev = self.node.ev_sb_shove(c, stacks, self.BB_call_vs_SB)
            if ev > 0.0:
                new_set.add(c)
        changed = (len(new_set ^ self.SB_shove) > 0)
        self.SB_shove = new_set
        print(f"SB shove : {len(new_set)} combos (modifié : {changed})")
        return changed

    def _update_bb_call_vs_sb(self, stacks):
        print(f"\nMise à jour : range BB call vs SB…")
        new_set = set()
        for c in tqdm.tqdm(self._all, desc="BB call vs SB", leave=False):
            ev_call = self.node.ev_call_vs_sb(c, stacks, self.SB_shove)
            if ev_call > 0.0:
                new_set.add(c)
        changed = (len(new_set ^ self.BB_call_vs_SB) > 0)
        self.BB_call_vs_SB = new_set
        print(f"BB call vs SB : {len(new_set)} combos (modifié : {changed})")
        return changed

    def iterate(self, n_iters: int = 8) -> None:
        stacks = self.cfg.stacks_bb
        print(f"\nDémarrage des itérations ({n_iters})")
        print(f"Stacks : BTN={stacks[0]}bb, SB={stacks[1]}bb, BB={stacks[2]}bb")
        print(f"Samples Monte Carlo : {self.cfg.mc_samples}")
        print(f"Total des combos évalués : {len(self._all)}")
        
        for it in range(1, n_iters+1):
            print(f"\n{'='*60}")
            print(f"ITÉRATION {it}/{n_iters}")
            print(f"{'='*60}")
            start_time = time.time()
            
            c1 = self._update_sb_call_vs_btn(stacks)
            c2 = self._update_bb_call_vs_btn(stacks)
            c3 = self._update_bb_call_vs_sb(stacks)
            c4 = self._update_btn_shove(stacks)
            c5 = self._update_sb_shove(stacks)
            
            dt = time.time() - start_time
            
            print(f"\nRésumé itération {it} :")
            print(f"Durée : {dt:.2f}s")
            print(f"Modifs : SB_call_vs_BTN={c1}, BB_call_vs_BTN={c2}, BB_call_vs_SB={c3}, BTN_shove={c4}, SB_shove={c5}")
            print(f"Tailles actuelles :")
            print(f"   BTN shove : {len(self.BTN_shove)} combos ({self.coverage_pct(self.BTN_shove):.1f}%)")
            print(f"   SB call vs BTN : {len(self.SB_call_vs_BTN)} combos ({self.coverage_pct(self.SB_call_vs_BTN):.1f}%)")
            print(f"   BB call vs BTN : {len(self.BB_call_vs_BTN)} combos ({self.coverage_pct(self.BB_call_vs_BTN):.1f}%)")
            print(f"   SB shove : {len(self.SB_shove)} combos ({self.coverage_pct(self.SB_shove):.1f}%)")
            print(f"   BB call vs SB : {len(self.BB_call_vs_SB)} combos ({self.coverage_pct(self.BB_call_vs_SB):.1f}%)")
            
            total_changes = sum([c1, c2, c3, c4, c5])
            if total_changes == 0:
                print(f"\nCONVERGENCE atteinte à l’itération {it}.")
                break

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

    def print_summary(self) -> None:
        def show(name, s):
            cov = self.coverage_pct(s)
            top = list(self.summarize_169(s).items())[:20]
            print(f"\n{name}: {len(s)} combos ({cov:.1f}%)")
            print("Top (par nombre de combos) :", ", ".join([f"{k}:{v}" for k,v in top]))
        show("BTN shove", self.BTN_shove)
        show("SB call vs BTN", self.SB_call_vs_BTN)
        show("BB call vs BTN", self.BB_call_vs_BTN)
        show("SB shove", self.SB_shove)
        show("BB call vs SB", self.BB_call_vs_SB)

# ======================
# Démonstration
# ======================
if __name__ == "__main__":
    cfg = ExpressoConfig(
        sb=0.5, bb=1.0,
        stacks_bb=(25.0, 25.0, 25.0),  # (BTN, SB, BB)
        mc_samples=400,
        seed=123
    )
    solver = SpinGoPushFoldSolver(cfg)
    solver.iterate(n_iters=8)  # augmenter pour plus de stabilité (coût CPU ↑)
    solver.print_summary()
