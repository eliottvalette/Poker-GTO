# artificial_policy.py
from __future__ import annotations
import json
import gzip
from typing import Dict
from infoset import unpack_infoset_key_dense, _LABELS_169

def _decode_compact_entry(entry: list[int]) -> Dict[str, float]:
    mask = entry[0]
    qs = entry[1:]
    total = sum(qs)
    if total <= 0:
        return {}
    dist = {}
    idx_q = 0
    for i, a in enumerate(ACTIONS):
        if (mask >> i) & 1:
            q = qs[idx_q]
            dist[a] = q / total
            idx_q += 1
    return dist

def _encode_compact(dist: Dict[str, float], keep_top_k: int = 3) -> list[int]:
    # normalise
    s = sum(dist.values())
    if s <= 0:
        return [0]
    norm = {a: dist[a]/s for a in dist}
    items = [(i, norm.get(a, 0.0)) for i, a in enumerate(ACTIONS) if norm.get(a, 0.0) > 0.0]
    items.sort(key=lambda x: x[1], reverse=True)
    items = items[:keep_top_k]
    ps = [p for _, p in items]
    s2 = sum(ps)
    ps = [p/s2 for p in ps]
    qs = [int(round(p*255)) for p in ps]
    diff = 255 - sum(qs)
    if diff != 0 and qs:
        j = max(range(len(qs)), key=lambda k: qs[k])
        qs[j] = max(0, min(255, qs[j] + diff))
    mask = 0
    for i, _ in items:
        mask |= (1 << i)
    return [mask] + qs

ACTIONS = ["FOLD", "CHECK", "CALL", "RAISE", "ALL-IN"]
ID_TO_PHASE = {0:"PREFLOP",1:"FLOP",2:"TURN",3:"RIVER",4:"SHOWDOWN"}
ROLE_NAMES   = ["SB","BB","BTN"]
_169_LABEL   = {i: _LABELS_169[i] for i in range(len(_LABELS_169))}

# --- Paramètres à ajuster ---
SPR_ALLIN_MAX_BUCKET = 2      # si SPR bucket <= 2 -> on autorise un peu d'ALL-IN
ALLIN_HANDS = set(["AA","KK","QQ","JJ","AKs","AQs","AKo"])
ALLIN_FRACTION_OF_RAISE = 0.25  # % du RAISE converti en ALL-IN quand SPR faible
LIMP_ALLOWED = False            # si True, déplacer une partie de RAISE vers CALL
LIMP_FRACTION = 0.0

def load_open_range(path="ranges/global_matrix.json") -> Dict[str, float]:
    with open(path, "r") as f:
        return json.load(f)  # ex: {"AA":1.0,"AKs":1.0,...} valeurs 0..1

def main():
    open_range = load_open_range()

    # lecture policy gz compact
    with gzip.open("policy/avg_policy.json.gz", "rt", encoding="utf-8") as f:
        raw = json.load(f)

    base_policy: Dict[str, Dict[str, float]] = {}
    for k, v in raw.items():
        if isinstance(v, list) and v and isinstance(v[0], int):
            dist = _decode_compact_entry(v)
            if dist:
                base_policy[k] = dist

    new_policy_compact = {}

    touched = 0
    for k_str, dist in base_policy.items():
        k = int(k_str)
        fields = unpack_infoset_key_dense(k)
        phase = ID_TO_PHASE.get(fields["PHASE"], str(fields["PHASE"]))
        role  = fields["ROLE"]
        hand_idx = fields["HAND"]
        spr_q = fields["SPR"]

        label_169 = _169_LABEL[hand_idx]
        p_open = float(open_range.get(label_169, 0.0))

        p_raise = max(0.0, min(1.0, p_open))
        p_fold  = 1.0 - p_raise
        p_call  = 0.0
        p_allin = 0.0

        if LIMP_ALLOWED and LIMP_FRACTION > 0.0:
            move = min(p_raise, LIMP_FRACTION)
            p_raise -= move
            p_call  += move

        if spr_q <= SPR_ALLIN_MAX_BUCKET and label_169 in ALLIN_HANDS and p_raise > 0.0:
            move = min(p_raise, p_raise * ALLIN_FRACTION_OF_RAISE)
            p_raise -= move
            p_allin += move

        out_float = {}
        if p_fold > 0:  out_float["FOLD"] = p_fold
        if p_call > 0:  out_float["CALL"] = p_call
        if p_raise > 0: out_float["RAISE"] = p_raise
        if p_allin > 0: out_float["ALL-IN"] = p_allin

        compact = _encode_compact(out_float, keep_top_k=3)
        if compact[0] != 0:
            new_policy_compact[k_str] = compact
            touched += 1

    out_path = "policy/avg_policy_artificial.json.gz"
    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        json.dump(new_policy_compact, f, separators=(",",":"), ensure_ascii=False)

    print(f"[OK] Infosets modifiés PREFLOP: {touched} / {len(base_policy)}")
    print(f"[SAVE] {out_path}")

if __name__ == "__main__":
    main()
