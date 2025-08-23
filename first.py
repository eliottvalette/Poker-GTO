# expresso_pushfold_solver.py
# ------------------------------------------------------------
# Spin&Go (3-handed) PUSH/FOLD solver — from scratch, pure Python.
# Focus: preflop 3-max with push/fold only, cEV model (no overcalls).
# - Positions: BTN (no blind), SB (posts sb), BB (posts bb).
# - Actions:
#     * BTN: {FOLD, SHOVE}
#     * If BTN shoves: SB: {FOLD, CALL}; if SB folds → BB: {FOLD, CALL}
#       (Pas d’overcall: si SB call, BB ne peut pas overcall)
#     * Si BTN fold: SB: {FOLD, SHOVE}; si SB shoves → BB: {FOLD, CALL}
# - Décisions via best-response itératif sur ranges de shove/call.
# - Équités all-in via Monte Carlo avec éval 7 cartes (best 5).
# - EV en jetons (cEV) en se plaçant "post-blinds" (fold = 0 EV local).
#
# Hypothèses/simplifications (à lever plus tard si besoin) :
# - Pas d’overcall (3-way all-in) quand BTN shove et SB call.
# - Décisions locales en cEV ; l’ICM ne sert qu’en reporting éventuel.
# - Échantillonnage uniforme des combos dans une range adverse donnée.
# - Pas de block effects dynamiques au-delà de la suppression des cartes héros.
#
# Sorties :
# - Solver.iterate() met à jour 5 ranges :
#     BTN_shove, SB_call_vs_BTN, BB_call_vs_BTN, SB_shove, BB_call_vs_SB
# - Utils pour afficher les ranges en 169 (A2..AKs) avec coverages.
#
# ------------------------------------------------------------
from __future__ import annotations
import random, math, itertools
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import List, Tuple, Dict, Set, Iterable, Optional

# =========================
# Cartes, combos, utilités
# =========================

RANKS = list(range(2, 15))  # 2..14 (A=14)
SUITS = list(range(4))       # 0..3

def card(rank: int, suit: int) -> int:
    return (rank - 2) * 4 + suit  # 0..51

def card_rank(c: int) -> int:
    return (c // 4) + 2

def card_suit(c: int) -> int:
    return c % 4

DECK = [card(r, s) for r in RANKS for s in SUITS]  # 52 cartes

def all_starting_combos() -> List[Tuple[int,int]]:
    combos = []
    for i in range(52):
        for j in range(i+1, 52):
            combos.append((i, j))
    return combos

ALL_COMBOS = all_starting_combos()  # 1326 combos

# 169 labelling (AKs, AKo, A5s, etc.)
RANK_TO_STR = {14:'A', 13:'K', 12:'Q', 11:'J', 10:'T', 9:'9', 8:'8', 7:'7', 6:'6', 5:'5', 4:'4', 3:'3', 2:'2'}
def combo_to_169(c1: int, c2: int) -> str:
    r1, r2 = card_rank(c1), card_rank(c2)
    s1, s2 = card_suit(c1), card_suit(c2)
    hi, lo = max(r1, r2), min(r1, r2)
    if r1 == r2:
        return f"{RANK_TO_STR[hi]}{RANK_TO_STR[lo]}"
    suited = (s1 == s2)
    if hi == r1:
        top, bot = r1, r2
    else:
        top, bot = r2, r1
    return f"{RANK_TO_STR[top]}{RANK_TO_STR[bot]}{'s' if suited else 'o'}"

def filter_combos_excluding(combos: Iterable[Tuple[int,int]], blocked: Set[int]) -> List[Tuple[int,int]]:
    out = []
    for a,b in combos:
        if a in blocked or b in blocked: 
            continue
        out.append((a,b))
    return out

# ================
# Éval mains 7->5
# ================
def hand_rank_5(cards5: Tuple[int,int,int,int,int]) -> Tuple:
    # Retourne un tuple comparable (cat, tiebreakers...), cat: 8..0
    ranks = sorted([card_rank(c) for c in cards5], reverse=True)
    suits = [card_suit(c) for c in cards5]
    rank_counts = Counter(ranks)
    counts_sorted = sorted(rank_counts.items(), key=lambda x: (x[1], x[0]), reverse=True)  # by count then rank
    is_flush = max(Counter(suits).values()) == 5

    # Straight check (handle wheel A-5)
    uniq = sorted(set(ranks), reverse=True)
    def straight_high(uniq_ranks):
        # returns high card of straight or None
        # handle A-5 (A=14, 5..A)
        if 14 in uniq_ranks:
            uniq_ranks = uniq_ranks + [1]  # treat A as 1
        for i in range(len(uniq_ranks)-4):
            window = uniq_ranks[i:i+5]
            if all(window[k] - 1 == window[k+1] for k in range(4)):
                return window[0] if window[0] != 1 else 5
        return None
    straight_hi = straight_high(uniq)
    is_straight = straight_hi is not None

    if is_straight and is_flush:
        return (8, straight_hi)  # straight flush
    # Four of a kind
    if counts_sorted[0][1] == 4:
        four = counts_sorted[0][0]
        kicker = max([r for r in ranks if r != four])
        return (7, four, kicker)
    # Full house
    if counts_sorted[0][1] == 3 and counts_sorted[1][1] == 2:
        trips = counts_sorted[0][0]
        pair = counts_sorted[1][0]
        return (6, trips, pair)
    # Flush
    if is_flush:
        return (5, ) + tuple(ranks)
    # Straight
    if is_straight:
        return (4, straight_hi)
    # Trips
    if counts_sorted[0][1] == 3:
        trips = counts_sorted[0][0]
        kickers = [r for r in ranks if r != trips][:2]
        return (3, trips) + tuple(kickers)
    # Two pair
    if counts_sorted[0][1] == 2 and counts_sorted[1][1] == 2:
        hi_pair = max(counts_sorted[0][0], counts_sorted[1][0])
        lo_pair = min(counts_sorted[0][0], counts_sorted[1][0])
        kicker = max([r for r in ranks if r != hi_pair and r != lo_pair])
        return (2, hi_pair, lo_pair, kicker)
    # One pair
    if counts_sorted[0][1] == 2:
        pair = counts_sorted[0][0]
        kickers = [r for r in ranks if r != pair][:3]
        return (1, pair) + tuple(kickers)
    # High card
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
        self.cache2 = {}  # ((h1a,h1b), frozenset(villain_combos_ids), eff), mc_samples -> (p_win,p_tie)
    @staticmethod
    def combo_norm(c):
        a,b = c
        return (a,b) if a<b else (b,a)

def sample_board(excluded: Set[int], rng: random.Random) -> Tuple[int,int,int,int,int]:
    pool = [c for c in DECK if c not in excluded]
    rng.shuffle(pool)
    return tuple(pool[:5])

def sample_combo_from_range(range_combos: List[Tuple[int,int]], excluded: Set[int], rng: random.Random) -> Optional[Tuple[int,int]]:
    candidates = [c for c in range_combos if (c[0] not in excluded and c[1] not in excluded)]
    if not candidates:
        return None
    return rng.choice(candidates)

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
    # Pré-sélection rapide
    vill_list = [c for c in villain_range if c[0] not in blocked and c[1] not in blocked]
    if not vill_list:
        return (1.0, 0.0, 0.0)  # personne ne peut caller -> on traite à part dans EV
    for _ in range(mc_samples):
        vill = rng.choice(vill_list)
        va, vb = vill
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
# Config Spin&Go / Push-Fold
# ===========================
@dataclass
class ExpressoConfig:
    sb: float = 0.5
    bb: float = 1.0
    stacks_bb: Tuple[float,float,float] = (25.0, 25.0, 25.0)  # (BTN,SB,BB) en BB
    mc_samples: int = 400
    seed: int = 42

# ===========================
# Ranges (sets de combos)
# ===========================
def combos_to_set(combos: Iterable[Tuple[int,int]]) -> Set[Tuple[int,int]]:
    return set((a,b) if a<b else (b,a) for a,b in combos)

def all_combos_set() -> Set[Tuple[int,int]]:
    return combos_to_set(ALL_COMBOS)

# ===========================
# EV all-in / noeuds de jeu
# ===========================
class NodeEV:
    def __init__(self, cfg: ExpressoConfig):
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)

    def _pot_and_behind(self, stacks: Tuple[float,float,float]) -> Tuple[float, Tuple[float,float,float]]:
        # stacks en BB (avant blinds). On passe en "behind" après blinds postées.
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
        cEV pour HERO sur un all-in 2-way (post blinds).
        """
        E = self._eff(behind_hero, behind_vill)
        if E <= 0.0:
            return 0.0
        p_win, p_tie, p_lose = estimate_equity_vs_range(hero_combo, villain_range, self.cfg.mc_samples, self.rng)
        # pot final = pot + 2E ; EV hero = p_win*( -E + pot+2E ) + p_tie*( -E + 0.5*(pot+2E) ) + p_lose*(-E)
        pot_final = pot + 2.0 * E
        ev = p_win * (-E + pot_final) + p_tie * (-E + 0.5 * pot_final) + p_lose * (-E)
        return ev

    # ---- EV pour boutons décisionnels ----
    def ev_btn_shove(self, hero_combo: Tuple[int,int],
                     stacks: Tuple[float,float,float],
                     sb_call_range: Set[Tuple[int,int]],
                     bb_call_range: Set[Tuple[int,int]]) -> float:
        """
        EV (en cEV) du shove du BTN. Baseline fold=0 (post blinds).
        Branches:
          - SB call -> all-in BTN vs SB
          - SB fold & BB call -> all-in BTN vs BB
          - SB fold & BB fold -> BTN gagne le pot
        """
        pot, (bBTN, bSB, bBB) = self._pot_and_behind(stacks)
        blocked = {hero_combo[0], hero_combo[1]}
        sb_all = filter_combos_excluding(ALL_COMBOS, blocked)  # toutes mains possibles SB (posté sb déjà)
        sb_call = filter_combos_excluding(list(sb_call_range), blocked)
        bb_call = filter_combos_excluding(list(bb_call_range), blocked)

        # Probas "combinatoires" d'appel
        p_sb_call = len(sb_call) / len(sb_all) if sb_all else 0.0
        # Si SB fold, BB a l’option de call
        bb_all = filter_combos_excluding(ALL_COMBOS, blocked)
        p_bb_call = len(bb_call) / len(bb_all) if bb_all else 0.0

        # EV si SB call :
        ev_vs_sb = self._ev_allin_2way(hero_combo, sb_call, bBTN, bSB, pot)
        # EV si SB fold et BB call :
        ev_vs_bb = self._ev_allin_2way(hero_combo, bb_call, bBTN, bBB, pot)
        # EV si tout le monde fold :
        ev_steal = pot  # gain du pot actuel

        # EV totale (sans overcall) :
        ev = p_sb_call * ev_vs_sb + (1 - p_sb_call) * (p_bb_call * ev_vs_bb + (1 - p_bb_call) * ev_steal)
        return ev

    def ev_call_vs_btn(self, hero_combo: Tuple[int,int], stacks: Tuple[float,float,float],
                        btn_shove_range: Set[Tuple[int,int]], hero_pos: str) -> float:
        """
        EV pour CALL face au shove du BTN :
          hero_pos ∈ {"SB","BB"} (pas d’overcall dans ce modèle)
        Baseline fold = 0 (post blinds).
        """
        assert hero_pos in ("SB","BB")
        pot, (bBTN, bSB, bBB) = self._pot_and_behind(stacks)
        blocked = {hero_combo[0], hero_combo[1]}
        # BTN a déjà choisit de shove → sa range est conditionnée : on échantillonne dedans.
        btn_range = filter_combos_excluding(list(btn_shove_range), blocked)
        if not btn_range:
            return 0.0  # Personne ne shove en pratique -> call pas atteint

        if hero_pos == "SB":
            return self._ev_allin_2way(hero_combo, btn_range, bSB, bBTN, pot)
        else:
            # BB : SB a fold pour que BB décide (pas d’overcall)
            return self._ev_allin_2way(hero_combo, btn_range, bBB, bBTN, pot)

    def ev_sb_shove(self, hero_combo: Tuple[int,int], stacks: Tuple[float,float,float],
                    bb_call_range: Set[Tuple[int,int]]) -> float:
        """
        BTN a fold ; SB décide shove vs BB (ou steal). Baseline fold=0 (post blinds).
        Branches:
          - BB call -> all-in SB vs BB
          - BB fold -> SB gagne le pot
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
        BB face au shove du SB (BTN a fold). Baseline fold=0 (post blinds).
        """
        pot, (bBTN, bSB, bBB) = self._pot_and_behind(stacks)
        blocked = {hero_combo[0], hero_combo[1]}
        sb_range = filter_combos_excluding(list(sb_shove_range), blocked)
        if not sb_range:
            return 0.0
        return self._ev_allin_2way(hero_combo, sb_range, bBB, bSB, pot)

# ======================
# Solver push/fold 3-max
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

        # Initialisation simple : aucune main (solver va remplir)
        self._all = all_combos_set()
        self.rng = random.Random(cfg.seed)

    def _update_btn_shove(self, stacks):
        new_set = set()
        for c in self._all:
            ev = self.node.ev_btn_shove(c, stacks, self.SB_call_vs_BTN, self.BB_call_vs_BTN)
            if ev > 0.0:
                new_set.add(c)
        changed = (len(new_set ^ self.BTN_shove) > 0)
        self.BTN_shove = new_set
        return changed

    def _update_sb_call_vs_btn(self, stacks):
        new_set = set()
        for c in self._all:
            ev_call = self.node.ev_call_vs_btn(c, stacks, self.BTN_shove, "SB")
            if ev_call > 0.0:
                new_set.add(c)
        changed = (len(new_set ^ self.SB_call_vs_BTN) > 0)
        self.SB_call_vs_BTN = new_set
        return changed

    def _update_bb_call_vs_btn(self, stacks):
        new_set = set()
        for c in self._all:
            ev_call = self.node.ev_call_vs_btn(c, stacks, self.BTN_shove, "BB")
            if ev_call > 0.0:
                new_set.add(c)
        changed = (len(new_set ^ self.BB_call_vs_BTN) > 0)
        self.BB_call_vs_BTN = new_set
        return changed

    def _update_sb_shove(self, stacks):
        new_set = set()
        for c in self._all:
            ev = self.node.ev_sb_shove(c, stacks, self.BB_call_vs_SB)
            if ev > 0.0:
                new_set.add(c)
        changed = (len(new_set ^ self.SB_shove) > 0)
        self.SB_shove = new_set
        return changed

    def _update_bb_call_vs_sb(self, stacks):
        new_set = set()
        for c in self._all:
            ev_call = self.node.ev_call_vs_sb(c, stacks, self.SB_shove)
            if ev_call > 0.0:
                new_set.add(c)
        changed = (len(new_set ^ self.BB_call_vs_SB) > 0)
        self.BB_call_vs_SB = new_set
        return changed

    def iterate(self, n_iters: int = 8) -> None:
        stacks = self.cfg.stacks_bb
        for it in range(1, n_iters+1):
            c1 = self._update_sb_call_vs_btn(stacks)
            c2 = self._update_bb_call_vs_btn(stacks)
            c3 = self._update_bb_call_vs_sb(stacks)
            c4 = self._update_btn_shove(stacks)
            c5 = self._update_sb_shove(stacks)
            # Optionnel: petit logging concis
            print(f"[Iter {it}] BTN shove:{len(self.BTN_shove)} | SB call vs BTN:{len(self.SB_call_vs_BTN)} | "
                  f"BB call vs BTN:{len(self.BB_call_vs_BTN)} | SB shove:{len(self.SB_shove)} | "
                  f"BB call vs SB:{len(self.BB_call_vs_SB)}")

    # ------------- Affichage / résumé en 169 -------------
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
            print("Top (by combos count):", ", ".join([f"{k}:{v}" for k,v in top]))
        show("BTN shove", self.BTN_shove)
        show("SB call vs BTN", self.SB_call_vs_BTN)
        show("BB call vs BTN", self.BB_call_vs_BTN)
        show("SB shove", self.SB_shove)
        show("BB call vs SB", self.BB_call_vs_SB)

# ======================
# Demo d’utilisation
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
