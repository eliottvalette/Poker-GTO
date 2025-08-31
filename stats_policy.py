# stats_policy.py
from __future__ import annotations
import json
import gzip
from typing import Dict
from infoset import unpack_infoset_key_dense
from infoset import _LABELS_169
import pandas as pd 
from tqdm import tqdm

def _decode_compact_entry(entry) -> Dict[str, float]:
    # si c'est l'ancien format (liste brute)
    visits = entry["visits"]
    mask = entry["policy"][0]
    quantized = entry["policy"][1:]

    total = sum(quantized)
    if total <= 0:
        raise ValueError(f"[DECODE] Total <= 0: {total}")

    dist = {}
    idx_quantized = 0
    for action_index, action_name in enumerate(ACTIONS):
        if (mask >> action_index) & 1:
            q = quantized[idx_quantized]
            dist[action_name] = q / total
            idx_quantized += 1
    return dist, visits

# Actions canon
ACTIONS = ["FOLD", "CHECK", "CALL", "RAISE", "ALL-IN"]
ID_TO_PHASE = {0:"PREFLOP",1:"FLOP",2:"TURN",3:"RIVER",4:"SHOWDOWN"}
ROLE_LABELS = ["SB", "BB", "BTN"]
_169_TO_LABEL = {i: _LABELS_169[i] for i in range(len(_LABELS_169))}

def _decode_fields(key: int, unpacked: Dict[str, int], policy_dist: Dict[str, float] = None, visits: int = None) -> Dict[str, str]:
    decoded: Dict[str, str] = {}
    decoded["KEY"] = key
    decoded["PHASE"] = ID_TO_PHASE.get(unpacked["PHASE"], str(unpacked["PHASE"]))
    decoded["ROLE"] = ROLE_LABELS[unpacked["ROLE"]] if 0 <= unpacked["ROLE"] < len(ROLE_LABELS) else str(unpacked["ROLE"]) 
    decoded["HAND"] = _169_TO_LABEL.get(unpacked["HAND"], str(unpacked["HAND"]))
    # Keep technical buckets numeric for now (concise)
    decoded["BOARD"] = str(unpacked["BOARD"]) 
    decoded["POT"] = str(unpacked["POT"]) 
    decoded["RATIO"] = str(unpacked["RATIO"]) 
    decoded["SPR"] = str(unpacked["SPR"]) 
    decoded["HEROBOARD"] = str(unpacked["HEROBOARD"]) 
    decoded["VISITS"] = visits
    
    # Add policy values for all actions, defaulting to 0.0 if not present
    if policy_dist is not None:
        for action in ACTIONS:
            decoded[f"PROB_{action}"] = str(policy_dist.get(action, 0.0))
    
    return decoded

def mix_actions_by_phase(policy_json):
    phases = ["PREFLOP","FLOP","TURN","RIVER"]
    mix = {ph:{a:0.0 for a in ACTIONS} for ph in phases}
    count = {ph:0 for ph in phases}

    for k, dist in policy_json.items():
        f = unpack_infoset_key_dense(int(k))
        ph = ID_TO_PHASE.get(f["PHASE"], str(f["PHASE"]))
        if ph not in mix: 
            raise ValueError(f"[MIX] Phase not in mix: {ph}")

        # normalise la dist de l’infoset (sur actions présentes)
        s = sum(dist.values())
        if s <= 0: 
            raise ValueError(f"[MIX] Sum <= 0: {s}")    
        norm = {a: (dist.get(a,0.0)/s) for a in ACTIONS}

        # moyenne "macro" : somme des vecteurs puis / nb d'infosets
        for a in ACTIONS:
            mix[ph][a] += norm[a]
        count[ph] += 1

    for ph in phases:
        if count[ph] == 0:
            print(f"\n== {ph} ==\n(no data)")
            raise ValueError(f"[MIX] Count == 0: {count[ph]}")
        print(f"\n== {ph} ==")
        for a in ACTIONS:
            pct = 100.0 * mix[ph][a] / count[ph]
            print(f"{a}: {pct:.2f}%")


def build_dataframe(policy_json):
    # Collect all data in a list first for much better performance
    data_rows = []
    
    for k in tqdm(policy_json.keys()):
        unpacked_key = unpack_infoset_key_dense(int(k))
        decoded = _decode_fields(int(k), unpacked_key, policy_json[k][0], policy_json[k][1]) # Pass policy_dist here
        data_rows.append(decoded)
    
    # Create DataFrame from all collected data at once
    df = pd.DataFrame(data_rows)
    return df

def extraction_policy_data(src_path="policy/avg_policy.json.gz"):
    with gzip.open(src_path, "rt", encoding="utf-8") as f:
        raw = json.load(f)

    policy_json = {}
    for idx, (k, v) in enumerate(raw.items()):
        distribution, visits = _decode_compact_entry(v)
        policy_json[k] = [distribution, visits]
        
        if idx < 10:
            print(f"{k}: [{distribution} / {visits}]")

    df = build_dataframe(policy_json)
    df.to_csv("policy/avg_policy.csv", index=False)

if __name__ == "__main__":
    extraction_policy_data()