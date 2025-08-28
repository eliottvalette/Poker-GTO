# infoset.py
from __future__ import annotations
from typing import List, Tuple
from classes import Card, Player

# --- 169 map (lisible <-> index) ---
_R2S = {14:"A",13:"K",12:"Q",11:"J",10:"T",9:"9",8:"8",7:"7",6:"6",5:"5",4:"4",3:"3",2:"2"}
_RANKS_DESC = [14,13,12,11,10,9,8,7,6,5,4,3,2]  # A..2 (lignes/colonnes)

def combo_label_169(c1: Card, c2: Card) -> str:
    if c1.rank == c2.rank:
        s = _R2S[c1.rank]
        return f"{s}{s}"
    hi, lo = (c1, c2) if c1.rank >= c2.rank else (c2, c1)
    suited = hi.suit == lo.suit
    return f"{_R2S[hi.rank]}{_R2S[lo.rank]}{'s' if suited else 'o'}"

# 13x13 labels: diagonale = paires, triangle haut = suited, triangle bas = offsuit
_LABELS_169 = []
for i, r1 in enumerate(_RANKS_DESC):
    for j, r2 in enumerate(_RANKS_DESC):
        if i == j:
            _LABELS_169.append(f"{_R2S[r1]}{_R2S[r2]}")
        elif i < j:
            _LABELS_169.append(f"{_R2S[r1]}{_R2S[r2]}s")  # top-right
        else:
            _LABELS_169.append(f"{_R2S[r2]}{_R2S[r1]}o")  # bottom-left

LABEL_TO_169IDX = {lab: i for i, lab in enumerate(_LABELS_169)}

# -------- helpers --------
PHASE_TO_ID = {"PREFLOP":0,"FLOP":1,"TURN":2,"RIVER":3,"SHOWDOWN":4}
ROLE_LABELS = ["SB", "BB", "BTN"]

def hand169_idx(c1: Card, c2: Card) -> tuple[int, str]:
    lab = combo_label_169(c1, c2)
    idx = LABEL_TO_169IDX.get(lab)
    if idx is None:
        raise KeyError(f"unknown 169 label: {lab}")
    return idx, lab

def board_bucket(board: List[Card]) -> Tuple[int, str]:
    n = len(board)
    if n == 0:
        return 0, "PF"  # préflop: bucket 0

    ranks = [c.rank for c in board]
    suits = [c.suit for c in board]

    # suit texture
    max_suit = max(suits.count(s) for s in set(suits))
    if   (n == 3 and max_suit == 3) or (n == 4 and max_suit >= 4) or (n == 5 and max_suit >= 5):
        suit_tex = 2  # monotone
    elif (n >= 3 and max_suit == n-1):  # 2 on flop, 3 on turn, 4 on river
        suit_tex = 1  # two-tone / flushy
    else:
        suit_tex = 0  # rainbow

    # paired
    paired = 1 if any(ranks.count(r) >= 2 for r in set(ranks)) else 0

    # high class by highest rank
    hi = max(ranks)
    if hi >= 12: hi_cls = 2          # Q+
    elif hi >= 10: hi_cls = 1        # T/J
    else: hi_cls = 0                  # ≤9

    idx = suit_tex*6 + paired*3 + hi_cls  # 0..17
    name = ["RB","TT","MONO"][suit_tex] + ("_PR" if paired else "_NP") + ["_LO","_MID","_HI"][hi_cls]
    return idx, name

# -------- Buckets sizing --------
# Layout (44 bits, en BB entiers) — on *réutilise* les champs:
# [ phase:3 | role:2 | hand169:8 | board:5 | potQ:8 | ratioQ:8 | sprQ:8 | raises:2 ]
_POS = {
    "RAISES": 0,
    "LASTH":  2,   # 8 bits  (réutilisé pour SPR bucket)
    "TOCALL": 10,  # 8 bits  (réutilisé pour ratio bucket)
    "POT":    18,  # 8 bits  (qlog pot)
    "BOARD":  26,  # 5 bits
    "HAND":   31,  # 8 bits
    "ROLE":   39,  # 2 bits
    "PHASE":  41,  # 3 bits
}
_MASK = {
    "RAISES": (1<<2)-1,
    "LASTH":  (1<<8)-1,
    "TOCALL": (1<<8)-1,
    "POT":    (1<<8)-1,
    "BOARD":  (1<<5)-1,
    "HAND":   (1<<8)-1,
    "ROLE":   (1<<2)-1,
    "PHASE":  (1<<3)-1,
}

def pack_u64(**f) -> int:
    v = 0
    v |= (f["raises"] & _MASK["RAISES"]) << _POS["RAISES"]
    v |= (f["last_h"] & _MASK["LASTH"])  << _POS["LASTH"]    # SPR bucket
    v |= (f["tocall"] & _MASK["TOCALL"]) << _POS["TOCALL"]   # ratio bucket
    v |= (f["pot"]    & _MASK["POT"])    << _POS["POT"]      # qlog pot
    v |= (f["board"]  & _MASK["BOARD"])  << _POS["BOARD"]
    v |= (f["hand"]   & _MASK["HAND"])   << _POS["HAND"]
    v |= (f["role"]   & _MASK["ROLE"])   << _POS["ROLE"]
    v |= (f["phase"]  & _MASK["PHASE"])  << _POS["PHASE"]
    return v

def unpack_infoset_key_dense(k: int) -> dict:
    return {
        "phase":  (k >> _POS["PHASE"])  & _MASK["PHASE"],
        "role":   (k >> _POS["ROLE"])   & _MASK["ROLE"],
        "hand":   (k >> _POS["HAND"])   & _MASK["HAND"],
        "board":  (k >> _POS["BOARD"])  & _MASK["BOARD"],
        "pot":    (k >> _POS["POT"])    & _MASK["POT"],     # qlog pot bucket id
        "tocall": (k >> _POS["TOCALL"]) & _MASK["TOCALL"],  # ratio bucket id
        "last":   (k >> _POS["LASTH"])  & _MASK["LASTH"],   # SPR bucket id
        "raises": (k >> _POS["RAISES"]) & _MASK["RAISES"],
    }

# --- Edges pour les buckets ---
# Pot quasi-log (BB). ~24 niveaux, dense en micro/low, plus large ensuite.
_POT_EDGES_BB = [
    0,1,2,3,4,5,6,8,10,12,16,20,24,32,40,48,64,80,96,128,160,192,256,320, float("inf")
]  # -> indices 0..23

# Ratio toCall/pot (invariant d’échelle)
_RATIO_EDGES = [0.00, 0.05, 0.125, 0.25, 0.5, 1.0, 2.0, float("inf")]  # -> 0..6

# SPR = stack_effectif / pot (profondeur de tapis)
_SPR_EDGES   = [0.00, 0.75, 1.25, 2.0, 3.5, 6.0, 10.0, float("inf")]   # -> 0..6

def _bucket_from_edges(x: float, edges: list[float]) -> int:
    # edges croissants, dernier = +inf ; renvoie 0..len(edges)-2
    for i in range(len(edges)-1):
        if x <= edges[i+1]:
            return i
    return len(edges)-2

def qlog_bb(pot_bb: float) -> int:
    # map BB -> bucket id (0..23)
    x = max(0.0, float(pot_bb))
    return _bucket_from_edges(x, _POT_EDGES_BB)

def ratio_bucket(to_call_bb: float, pot_bb: float) -> int:
    denom = max(1.0, float(pot_bb))
    r = max(0.0, float(to_call_bb)) / denom
    return _bucket_from_edges(r, _RATIO_EDGES)

def spr_bucket(eff_stack_bb: float, pot_bb: float) -> int:
    denom = max(1.0, float(pot_bb))
    spr = max(0.0, float(eff_stack_bb)) / denom
    return _bucket_from_edges(spr, _SPR_EDGES)

# -------- public API --------
def build_infoset_key_fast(game, hero) -> int:
    phase_id = PHASE_TO_ID[game.current_phase]
    role_id  = hero.role

    # hand 169 id (sans string)
    c1, c2 = hero.cards
    rmap = {14:12,13:11,12:10,11:9,10:8,9:7,8:6,7:5,6:4,5:3,4:2,3:1,2:0}
    i = rmap[c1.rank]; j = rmap[c2.rank]
    suited = (c1.suit == c2.suit)
    if i == j:
        hidx = i*13 + i
    elif suited:
        hidx = max(i,j)*13 + min(i,j)
    else:
        hidx = min(i,j)*13 + max(i,j)

    # board bucket compact
    b = game.community_cards
    n = len(b)
    if n == 0:
        bidx = 0
    else:
        ranks = [c.rank for c in b]
        suits = [c.suit for c in b]
        max_suit = max(suits.count(s) for s in set(suits))
        if   (n == 3 and max_suit == 3) or (n == 4 and max_suit >= 4) or (n == 5 and max_suit >= 5):
            suit_tex = 2
        elif (n >= 3 and max_suit == n-1):
            suit_tex = 1
        else:
            suit_tex = 0
        paired = 1 if any(ranks.count(r) >= 2 for r in set(ranks)) else 0
        hi = max(ranks)
        hi_cls = 2 if hi >= 12 else (1 if hi >= 10 else 0)
        bidx = suit_tex*6 + paired*3 + hi_cls

    # sizing originaux en BB
    pot_bb    = float(game.main_pot)
    tocall_bb = max(0.0, float(game.current_maximum_bet - hero.current_player_bet))
    # stack effectif côté héros vs tapis des vilains encore “live”
    live_opponents = [p for p in game.players if p.is_active and (not p.has_folded)]
    eff_vs_each = [min(hero.stack, op.stack) for op in live_opponents if op is not hero]
    eff_stack   = min(eff_vs_each) if eff_vs_each else hero.stack

    # -> buckets compacts
    pot_q   = qlog_bb(pot_bb)                      # 0..23 (on reste sur 8 bits)
    ratio_q = ratio_bucket(tocall_bb, pot_bb)      # 0..6
    spr_q   = spr_bucket(eff_stack, pot_bb)        # 0..6
    raises  = min(3, int(getattr(game, "number_raise_this_game_phase", 0)))

    return pack_u64(
        phase=phase_id, role=role_id, hand=hidx, board=bidx,
        pot=pot_q, tocall=ratio_q, last_h=spr_q, raises=raises
    )

def build_infoset_key(game, hero: Player) -> Tuple[str, int]:
    phase_id = PHASE_TO_ID[game.current_phase]
    role_id  = hero.role

    # hand 169 (lisible)
    hidx, hlab = hand169_idx(hero.cards[0], hero.cards[1])

    # board bucket (lisible)
    bidx, bname = board_bucket(game.community_cards)

    # sizing en BB
    pot_bb    = float(game.main_pot)
    tocall_bb = max(0.0, float(game.current_maximum_bet - hero.current_player_bet))
    live_opponents = [p for p in game.players if p.is_active and (not p.has_folded)]
    eff_vs_each = [min(hero.stack, op.stack) for op in live_opponents if op is not hero]
    eff_stack   = min(eff_vs_each) if eff_vs_each else hero.stack

    # buckets
    pot_q   = qlog_bb(pot_bb)
    ratio_q = ratio_bucket(tocall_bb, pot_bb)
    spr_q   = spr_bucket(eff_stack, pot_bb)
    raises  = min(3, int(getattr(game, "number_raise_this_game_phase", 0)))

    dense = pack_u64(
        phase=phase_id, role=role_id, hand=hidx, board=bidx,
        pot=pot_q, tocall=ratio_q, last_h=spr_q, raises=raises
    )

    readable = (
        f"ph={game.current_phase} role={hero.role} "
        f"hand={hlab} bkt={bname} "
        f"potQ={pot_q} ratioQ={ratio_q} sprQ={spr_q} "
        f"raises={raises}"
    )
    return readable, dense
