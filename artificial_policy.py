# artificial_policy.py
from __future__ import annotations
import json
import gzip
from typing import Dict, Tuple
from infoset import unpack_infoset_key_dense, _LABELS_169
import os

ACTIONS = ["FOLD", "CHECK", "CALL", "RAISE", "ALL-IN"]
ID_TO_PHASE = {0:"PREFLOP",1:"FLOP",2:"TURN",3:"RIVER",4:"SHOWDOWN"}
ROLE_NAMES   = ["SB","BB","BTN"]
_169_LABEL   = {i: _LABELS_169[i] for i in range(len(_LABELS_169))}

# --- Paramètres optionnels (non utilisés si tu ne t'en sers pas) ---
SPR_ALLIN_MAX_BUCKET = 2
ALLIN_HANDS = set(["AA","KK","QQ","JJ","AKs","AQs","AKo"])
ALLIN_FRACTION_OF_RAISE = 0.25
LIMP_ALLOWED = False
LIMP_FRACTION = 0.0

def _decode_compact_entry(entry_list: list[int]) -> Dict[str, float]:
    """Decode [mask, q...] -> {action: prob}"""
    mask = entry_list[0]
    qs = entry_list[1:]
    total = sum(qs)
    if not mask or total <= 0:
        return {}
    dist: Dict[str, float] = {}
    qi = 0
    for i, a in enumerate(ACTIONS):
        if (mask >> i) & 1:
            q = qs[qi] if qi < len(qs) else 0
            qi += 1
            dist[a] = q / total
    return dist

def _encode_compact(dist: Dict[str, float], keep_top_k: int = 3) -> list[int]:
    """Encode {action: prob} -> [mask, q...] avec quantif sur 255"""
    s = sum(dist.values())
    if s <= 0:
        return [0]
    norm = {a: dist.get(a, 0.0) / s for a in ACTIONS if dist.get(a, 0.0) > 0.0}
    items = [(i, norm.get(a, 0.0)) for i, a in enumerate(ACTIONS) if norm.get(a, 0.0) > 0.0]
    items.sort(key=lambda x: x[1], reverse=True)
    items = items[:keep_top_k]

    probs = [p for _, p in items]
    s2 = sum(probs)
    probs = [p / s2 for p in probs] if s2 > 0 else []

    quantized = [int(round(p * 255)) for p in probs]
    diff = 255 - sum(quantized)
    if diff != 0 and quantized:
        j = max(range(len(quantized)), key=lambda k: quantized[k])
        quantized[j] = max(0, min(255, quantized[j] + diff))

    mask = 0
    for i, _ in items:
        mask |= (1 << i)
    return [mask] + quantized

def load_open_range(path="ranges/global_matrix.json") -> Dict[str, float]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def decode_entry_any_format(value) -> Tuple[Dict[str, float], int]:
    """
    Accepte ancien format (list) ou nouveau (dict {"policy":[...], "visits":N}).
    Renvoie (dist, visits).
    """
    if isinstance(value, list):
        return _decode_compact_entry(value), 1
    if isinstance(value, dict):
        entry = value
        plist = entry.get("policy")
        visits = int(entry.get("visits", 1))
        if isinstance(plist, list):
            return _decode_compact_entry(plist), visits
    return {}, 0

def main():
    open_range = load_open_range()

    # lecture policy gz (ancien ou nouveau format)
    with gzip.open("policy/avg_policy.json.gz", "rt", encoding="utf-8") as f:
        raw = json.load(f)

    base_policy: Dict[str, Dict[str, float]] = {}
    base_visits: Dict[str, int] = {}

    for k_str, v in raw.items():
        dist, visits = decode_entry_any_format(v)
        if dist:
            base_policy[k_str] = dist
            base_visits[k_str] = max(1, visits)

    touched = 0
    new_policy_compact = {}

    for k_str, dist in base_policy.items():
        k = int(k_str)
        fields = unpack_infoset_key_dense(k)
        phase_name = ID_TO_PHASE.get(fields["PHASE"], str(fields["PHASE"]))
        hand_idx = fields["HAND"]
        label_169 = _169_LABEL.get(hand_idx, "??")

        # proba d'open souhaitée (par main) - 0 si range absente
        proba_open = float(open_range.get(label_169, 0.0))

        legal_actions = list(dist.keys())
        artificial = {a: 0.0 for a in legal_actions}

        if phase_name == "PREFLOP":
            # simple règle: open = RAISE, sinon CHECK/FOLD selon légalité
            if "RAISE" in legal_actions:
                artificial["RAISE"] = proba_open
            if "ALL-IN" in legal_actions:
                artificial["ALL-IN"] = 0.0
            if "CALL" in legal_actions:
                artificial["CALL"] = 0.0
            if "CHECK" in legal_actions:
                artificial["CHECK"] = 1.0 - proba_open
            if "FOLD" in legal_actions and "CHECK" not in legal_actions:
                artificial["FOLD"] = 1.0 - proba_open
        else:
            # postflop: uniforme sur actions légales
            n = max(1, len(legal_actions))
            for a in legal_actions:
                artificial[a] = 1.0 / n

        compact = _encode_compact(artificial, keep_top_k=3)
        if compact[0] != 0:
            new_policy_compact[k_str] = {
                "policy": compact,
                "visits": 10
            }
            touched += 1

    out_path = "policy/avg_policy_artificial.json.gz"
    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        json.dump(new_policy_compact, f, separators=(",",":"), ensure_ascii=False)

    total_in = len(base_policy)
    print(f"[OK] Infosets modifiés PREFLOP: {touched} / {total_in}")
    print(f"[SAVE] {out_path}")

if __name__ == "__main__":
    main()
