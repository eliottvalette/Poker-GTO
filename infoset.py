# infoset.py
from __future__ import annotations
from typing import List, Tuple
from classes import Card, Player
import math

# ============================================================
# --- 169 map (lisible <-> index)
# ============================================================

_R2S = {14:"A",13:"K",12:"Q",11:"J",10:"T",
        9:"9",8:"8",7:"7",6:"6",5:"5",4:"4",3:"3",2:"2"}
_RANKS_DESC = [14,13,12,11,10,9,8,7,6,5,4,3,2]  # A..2

def combo_label_169(card_1: Card, card_2: Card) -> str:
    if card_1.rank == card_2.rank:
        s = _R2S[card_1.rank]
        return f"{s}{s}"
    high_card, low_card = (card_1, card_2) if card_1.rank >= card_2.rank else (card_2, card_1)
    suited = high_card.suit == low_card.suit
    return f"{_R2S[high_card.rank]}{_R2S[low_card.rank]}{'s' if suited else 'o'}"

# 13x13 grid
_LABELS_169 = []
for i, rank_1 in enumerate(_RANKS_DESC):
    for j, rank_2 in enumerate(_RANKS_DESC):
        if i == j:
            _LABELS_169.append(f"{_R2S[rank_1]}{_R2S[rank_2]}")
        elif i < j:
            _LABELS_169.append(f"{_R2S[rank_1]}{_R2S[rank_2]}s")  # suited
        else:
            _LABELS_169.append(f"{_R2S[rank_2]}{_R2S[rank_1]}o")  # offsuit

LABEL_TO_169IDX = {label: i for i, label in enumerate(_LABELS_169)}

# ============================================================
# --- Helpers
# ============================================================

PHASE_TO_ID = {"PREFLOP":0,"FLOP":1,"TURN":2,"RIVER":3,"SHOWDOWN":4}
ROLE_LABELS = ["SB", "BB", "BTN"]

def hand169_idx(card_1: Card, card_2: Card) -> tuple[int, str]:
    label = combo_label_169(card_1, card_2)
    return LABEL_TO_169IDX[label], label

def board_bucket(board: List[Card]) -> Tuple[int, str]:
    num_cards = len(board)
    if num_cards == 0:
        return 0, "PF"

    rank_counts = [card.rank for card in board]
    suit_counts = [card.suit for card in board]

    # suit texture
    max_suit = max(suit_counts.count(suit) for suit in set(suit_counts))
    if   (num_cards == 3 and max_suit == 3) or \
         (num_cards == 4 and max_suit >= 4) or \
         (num_cards == 5 and max_suit >= 5):
        suit_tex = 2  # monotone
    elif (num_cards >= 3 and max_suit == num_cards-1):
        suit_tex = 1  # two-tone
    else:
        suit_tex = 0  # rainbow

    # paired
    paired = 1 if any(rank_counts.count(rank) >= 2 for rank in set(rank_counts)) else 0

    # high card class
    high_card_rank = max(rank_counts)
    if high_card_rank >= 12: high_card_class = 2     # Q+
    elif high_card_rank >= 10: high_card_class = 1   # T/J
    else: high_card_class = 0            # ≤9

    idx = suit_tex*6 + paired*3 + high_card_class
    name = ["RB","TT","MONO"][suit_tex] + \
           ("_PR" if paired else "_NP") + \
           ["_LO","_MID","_HI"][high_card_class]

    return idx, name

# ============================================================
# --- Hero vs Board relation bucket
# ============================================================

def hero_vs_board_bucket(hero: Player, board: List[Card]) -> int:
    """Retourne un bucket 0..11 indiquant la relation directe entre main et board."""

    if not board:
        return 0  # préflop

    ranks = [card.rank for card in board]
    suits = [card.suit for card in board]
    hero_ranks = [card.rank for card in hero.cards]
    hero_suits = [card.suit for card in hero.cards]

    # ---- Pairing ----
    pair_type = 0
    if any(r in ranks for r in hero_ranks):
        hi = max(hero_ranks)
        if hi in ranks:
            if hi >= max(ranks): 
                pair_type = 3  # top pair+
            elif hi >= sorted(ranks)[-2]:
                pair_type = 2  # middle
            else:
                pair_type = 1  # low pair
    elif hero_ranks[0] == hero_ranks[1]:
        # pocket pair
        if hero_ranks[0] > max(ranks):
            pair_type = 4  # overpair
        else:
            pair_type = 1

    # ---- Flush draw ----
    flush_draw = 0
    for suit in set(hero_suits):
        need = 5 - suits.count(suit)
        if need <= 2:
            flush_draw = 2 if need == 1 else 1

    # ---- Straight draw (approx) ----
    straight_draw = 0
    all_ranks = sorted(set(ranks + hero_ranks))
    for start in range(2, 11):
        window = set(range(start, start+5))
        overlap = len(window & set(all_ranks))
        if overlap == 5:
            straight_draw = 3  # straight
        elif overlap == 4:
            straight_draw = max(straight_draw, 2)  # OESD
        elif overlap == 3:
            straight_draw = max(straight_draw, 1)  # gutshot

    # ---- Aggregate ----
    if flush_draw == 0 and straight_draw == 0 and pair_type == 0:
        return 0  # air
    if pair_type >= 3 and (flush_draw or straight_draw):
        return 7  # strong pair + draw
    if flush_draw == 2 and straight_draw >= 2:
        return 8  # combo draw
    if straight_draw == 3 or flush_draw == 2:
        return 9  # made straight/flush
    if pair_type == 4:
        return 6  # overpair
    if pair_type > 0:
        return 5  # some pair
    if straight_draw > 0 or flush_draw > 0:
        return 4  # some draw
    return 0

# ============================================================
# --- Bitfield layout (≤64 bits)
# ============================================================

_POS = {
    "HEROBOARD": 0,   # 4 bits (0..11)
    "SPR":       4,   # 8 bits
    "RATIO":    12,   # 8 bits
    "POT":      20,   # 8 bits
    "BOARD":    28,   # 5 bits
    "HAND":     33,   # 8 bits
    "ROLE":     41,   # 2 bits
    "PHASE":    43,   # 3 bits
}

_MASK = {k:(1<<bits)-1 for k,bits in {
    "HEROBOARD":4, "SPR":8, "RATIO":8, "POT":8,
    "BOARD":5, "HAND":8, "ROLE":2, "PHASE":3}.items()}

def pack_u64(**fields) -> int:
    value = 0
    for field in fields:
        value |= (fields[field] & _MASK[field]) << _POS[field]
    return value

def unpack_infoset_key_dense(k: int) -> dict:
    return {field: (k >> _POS[field]) & _MASK[field] for field in _POS}

# ============================================================
# --- Bucketing fonctions
# ============================================================

_POT_EDGES_BB = [0,1,2,3,4,5,6,8,10,12,16,20,24,32,40,48,
                 64,80,96,128,160,192,256,320,float("inf")]
_RATIO_EDGES  = [0.00,0.05,0.125,0.25,0.5,1.0,2.0,float("inf")]
_SPR_EDGES    = [0.00,0.75,1.25,2.0,3.5,6.0,10.0,float("inf")]

def _bucket_from_edges(x: float, edges: list[float]) -> int:
    for i in range(len(edges)-1):
        if x <= edges[i+1]:
            return i
    return len(edges)-2

def qlog_bb(pot_bb: float) -> int:
    return _bucket_from_edges(max(0.0, pot_bb), _POT_EDGES_BB)

def ratio_bucket(to_call_bb: float, pot_bb: float) -> int:
    ratio = max(0.0, to_call_bb) / max(1.0, pot_bb)
    return _bucket_from_edges(ratio, _RATIO_EDGES)

def spr_bucket(eff_stack_bb: float, pot_bb: float) -> int:
    spr_ratio = max(0.0, eff_stack_bb) / max(1.0, pot_bb)
    return _bucket_from_edges(spr_ratio, _SPR_EDGES)

# ============================================================
# --- API
# ============================================================

def build_infoset_key_fast(game, hero) -> int:
    phase_id = PHASE_TO_ID[game.current_phase]
    role_id  = hero.role

    # Hand 169 idx
    card_1, card_2 = hero.cards
    rank_to_index = {14:12,13:11,12:10,11:9,10:8,9:7,8:6,7:5,6:4,5:3,4:2,3:1,2:0}
    i, j = rank_to_index[card_1.rank], rank_to_index[card_2.rank]
    suited = (card_1.suit == card_2.suit)
    if i == j: hand_index = i*13+i
    elif suited: hand_index = max(i,j)*13+min(i,j)
    else: hand_index = min(i,j)*13+max(i,j)

    # Board bucket
    bidx, _ = board_bucket(game.community_cards)

    # Sizing
    pot_bb    = float(game.main_pot)
    tocall_bb = max(0.0, float(game.current_maximum_bet - hero.current_player_bet))
    live      = [p for p in game.players if p.is_active and not p.has_folded]
    eff       = min([min(hero.stack, op.stack) for op in live if op is not hero],
                    default=hero.stack)

    # Buckets
    pot_q   = qlog_bb(pot_bb)
    ratio_q = ratio_bucket(tocall_bb, pot_bb)
    spr_q   = spr_bucket(eff, pot_bb)
    hb      = hero_vs_board_bucket(hero, game.community_cards)

    return pack_u64(PHASE=phase_id, ROLE=role_id,
                    HAND=hand_index, BOARD=bidx,
                    POT=pot_q, RATIO=ratio_q,
                    SPR=spr_q, HEROBOARD=hb)

def build_infoset_key(game, hero: Player) -> Tuple[str, int]:
    hand_index, hand_label = hand169_idx(hero.cards[0], hero.cards[1])
    bidx, bname = board_bucket(game.community_cards)

    pot_bb    = float(game.main_pot)
    tocall_bb = max(0.0, float(game.current_maximum_bet - hero.current_player_bet))
    live      = [p for p in game.players if p.is_active and not p.has_folded]
    eff       = min([min(hero.stack, op.stack) for op in live if op is not hero],
                    default=hero.stack)

    pot_q   = qlog_bb(pot_bb)
    ratio_q = ratio_bucket(tocall_bb, pot_bb)
    spr_q   = spr_bucket(eff, pot_bb)
    hb      = hero_vs_board_bucket(hero, game.community_cards)

    dense = pack_u64(PHASE=PHASE_TO_ID[game.current_phase], ROLE=hero.role,
                     HAND=hand_index, BOARD=bidx,
                     POT=pot_q, RATIO=ratio_q,
                     SPR=spr_q, HEROBOARD=hb)

    readable = (
        f"ph={game.current_phase} role={hero.role} "
        f"hand={hand_label} bkt={bname} "
        f"potQ={pot_q} ratioQ={ratio_q} sprQ={spr_q} "
        f"heroBoard={hb}"
    )

    return readable, dense
