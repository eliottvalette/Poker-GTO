# infoset.py
from typing import List, Tuple
from classes import Card, Player

# --- 169 map (lisible <-> index) ---
_R2S = {14:"A",13:"K",12:"Q",11:"J",10:"T",9:"9",8:"8",7:"7",6:"6",5:"5",4:"4",3:"3",2:"2"}
_RANKS_DESC = [14,13,12,11,10,9,8,7,6,5,4,3,2]  # A..2 (lignes/colonnes)

def _combo_label_169(c1: Card, c2: Card) -> str:
    if c1.rank == c2.rank:
        s = _R2S[c1.rank]
        return f"{s}{s}"
    hi, lo = (c1, c2) if c1.rank >= c2.rank else (c2, c1)
    suited = hi.suit == lo.suit
    return f"{_R2S[hi.rank]}{_R2S[lo.rank]}{'s' if suited else 'o'}"

# 13x13 labels: diagonale = paires, triangle haut = suited, triangle bas = offsuit
_LABELS_169 = []
for i, hi in enumerate(_RANKS_DESC):
    for j, lo in enumerate(_RANKS_DESC):
        if i == j:
            _LABELS_169.append(f"{_R2S[hi]}{_R2S[lo]}")
        elif i < j:
            _LABELS_169.append(f"{_R2S[hi]}{_R2S[lo]}s")
        else:
            _LABELS_169.append(f"{_R2S[hi]}{_R2S[lo]}o")

LABEL_TO_169IDX = {lab: i for i, lab in enumerate(_LABELS_169)}

# -------- helpers --------
PHASE_TO_ID = {"PREFLOP":0,"FLOP":1,"TURN":2,"RIVER":3,"SHOWDOWN":4}
ROLE_LABELS = ["SB", "BB", "BTN"]

def _hand169_idx(c1: Card, c2: Card) -> tuple[int, str]:
    """
    Retourne (index_169, label) pour la main (c1,c2).
    - label via _combo_label_169 (ex: 'J7o')
    - index via table 13x13 (row*13 + col). Fallback si le label n'est pas dans le dict.
    """
    lab = _combo_label_169(c1, c2)  # 'AKs', 'QJo', 'TT', ...
    # Chemin rapide si présent
    idx = LABEL_TO_169IDX.get(lab)
    if idx is None:
        raise KeyError(f"unknown 169 label: {lab}")
    return idx, lab

def _board_bucket(board: List[Card]) -> Tuple[int, str]:
    """
    18 buckets = suit_texture(3) × paired(2) × highclass(3)
      - suit_texture: 0=rainbow, 1=twotone, 2=monotone(ish)
      - paired: 0=no pair on board, 1=paired (any rank count≥2)
      - highclass (via highest rank on board):
           0=≤9, 1= T/J, 2= Q+ (Q/K/A)
    Turn/River: 'monotone' si max suit count ≥ 4 (turn) ou ≥ 5 (river)
    """
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
        suit_tex = 1  # two-tone (flushy)
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

# -------- pack/unpack (uint64) --------
# Layout (38 bits utilisés, sans IP/OOP):
# [ phase:3 | role:2 | hand169:8 | board:5 | pot½:6 | tocall½:6 | last½:6 | raises:2 ]
# bit positions (LSB -> MSB):
_POS = {
    "RAISES": 0,
    "LASTH":  2,
    "TOCALL": 8,
    "POT":   14,
    "BOARD": 20,
    "HAND":  25,
    "ROLE":  33,
    "PHASE": 35,
}
_MASK = {
    "RAISES": (1<<2)-1,
    "LASTH":  (1<<6)-1,
    "TOCALL": (1<<6)-1,
    "POT":    (1<<6)-1,
    "BOARD":  (1<<5)-1,
    "HAND":   (1<<8)-1,
    "ROLE":   (1<<2)-1,
    "PHASE":  (1<<3)-1,
}

def _pack_u64(**f) -> int:
    v = 0
    v |= (f["raises"] & _MASK["RAISES"]) << _POS["RAISES"]
    v |= (f["last_h"] & _MASK["LASTH"])  << _POS["LASTH"]
    v |= (f["tocall"] & _MASK["TOCALL"]) << _POS["TOCALL"]
    v |= (f["pot"]    & _MASK["POT"])    << _POS["POT"]
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
        "pot½":   (k >> _POS["POT"])    & _MASK["POT"],
        "tocall½":(k >> _POS["TOCALL"]) & _MASK["TOCALL"],
        "last½":  (k >> _POS["LASTH"])  & _MASK["LASTH"],
        "raises": (k >> _POS["RAISES"]) & _MASK["RAISES"],
    }

# -------- public API --------
def build_infoset_key(game, hero: Player) -> Tuple[str, int]:
    """
    Retourne (lisible, dense_uint64)
    - lisible: string courte pour debug
    - dense:   uint64 compacté (à utiliser en prod)
    """
    phase_id = PHASE_TO_ID[game.current_phase]
    role_id  = hero.role

    # hand 169
    hidx, hlab = _hand169_idx(hero.cards[0], hero.cards[1])

    # board bucket
    bidx, bname = _board_bucket(game.community_cards)

    # sizing (en BB) -> 1/2BB
    pot_h     = game.main_pot
    tocall_bb = max(0.0, game.current_maximum_bet - hero.current_player_bet)
    tocall_h  = tocall_bb
    last_h    = getattr(game, "last_raise_amount", 0.0)
    raises    = min(3, int(getattr(game, "number_raise_this_game_phase", 0)))

    dense = _pack_u64(
        phase=phase_id, role=role_id, hand=hidx, board=bidx,
        pot=pot_h, tocall=tocall_h, last_h=last_h, raises=raises
    )

    readable = (
        f"ph={game.current_phase} role={hero.role} "
        f"hand={hlab} bkt={bname} "
        f"pot={pot_h/2:.1f} tocall={tocall_h/2:.1f} last={last_h/2:.1f} "
        f"raises={raises}"
    )
    return readable, dense


