# make_synthetic_policy.py
from __future__ import annotations
import json
from typing import Dict
from infoset import unpack_infoset_key_dense
from infoset import _LABELS_169
import pandas as pd 
from tqdm import tqdm

# Actions canon
ACTIONS = ["FOLD", "CHECK", "CALL", "RAISE", "ALL-IN"]
ID_TO_PHASE = {0:"PREFLOP",1:"FLOP",2:"TURN",3:"RIVER",4:"SHOWDOWN"}
ROLE_LABELS = ["SB", "BB", "BTN"]
_169_TO_LABEL = {i: _LABELS_169[i] for i in range(len(_LABELS_169))}
def _format_actions(proba_by_action: Dict[str, float]) -> str:
    present = [(a, proba_by_action[a]) for a in ACTIONS if a in proba_by_action]
    present.sort(key=lambda x: x[1], reverse=True)
    parts = [f"{a:<6} {p*100:6.2f}%" for a, p in present]
    return "  " + "  |  ".join(parts)


def _decode_fields(unpacked: Dict[str, int], policy_dist: Dict[str, float] = None) -> Dict[str, str]:
    decoded: Dict[str, str] = {}
    decoded["PHASE"] = ID_TO_PHASE.get(unpacked["PHASE"], str(unpacked["PHASE"]))
    decoded["ROLE"] = ROLE_LABELS[unpacked["ROLE"]] if 0 <= unpacked["ROLE"] < len(ROLE_LABELS) else str(unpacked["ROLE"]) 
    decoded["HAND"] = _169_TO_LABEL.get(unpacked["HAND"], str(unpacked["HAND"]))
    # Keep technical buckets numeric for now (concise)
    decoded["BOARD"] = str(unpacked["BOARD"]) 
    decoded["POT"] = str(unpacked["POT"]) 
    decoded["RATIO"] = str(unpacked["RATIO"]) 
    decoded["SPR"] = str(unpacked["SPR"]) 
    decoded["HEROBOARD"] = str(unpacked["HEROBOARD"]) 
    
    # Add policy values for all actions, defaulting to 0.0 if not present
    if policy_dist is not None:
        for action in ACTIONS:
            decoded[f"PROB_{action}"] = str(policy_dist.get(action, 0.0))
    
    return decoded


def _format_fields(decoded: Dict[str, str]) -> str:
    order = ["PHASE", "ROLE", "HAND", "BOARD", "POT", "RATIO", "SPR", "HEROBOARD"]
    labels = {
        "PHASE": "Phase",
        "ROLE": "Role",
        "HAND": "Hand",
        "BOARD": "Board",
        "POT": "PotQ",
        "RATIO": "RatioQ",
        "SPR": "SprQ",
        "HEROBOARD": "HeroBoard",
    }
    max_label = max(len(v) for v in labels.values())
    lines = [f"{labels[k]:<{max_label}} : {decoded[k]}" for k in order]
    return "\n".join(lines)

def mix_actions_by_phase(policy_json):
    phases = ["PREFLOP","FLOP","TURN","RIVER"]
    mix = {ph:{a:0.0 for a in ACTIONS} for ph in phases}
    count = {ph:0 for ph in phases}

    for k, dist in policy_json.items():
        f = unpack_infoset_key_dense(int(k))
        ph = ID_TO_PHASE.get(f["PHASE"], str(f["PHASE"]))
        if ph not in mix: 
            continue

        # normalise la dist de l’infoset (sur actions présentes)
        s = sum(dist.values())
        if s <= 0: 
            continue
        norm = {a: (dist.get(a,0.0)/s) for a in ACTIONS}

        # moyenne "macro" : somme des vecteurs puis / nb d'infosets
        for a in ACTIONS:
            mix[ph][a] += norm[a]
        count[ph] += 1

    for ph in phases:
        if count[ph] == 0:
            print(f"\n== {ph} ==\n(no data)")
            continue
        print(f"\n== {ph} ==")
        for a in ACTIONS:
            pct = 100.0 * mix[ph][a] / count[ph]
            print(f"{a}: {pct:.2f}%")


def build_dataframe(policy_json: Dict[str, Dict[str, float]]):
    # Collect all data in a list first for much better performance
    data_rows = []
    
    for k in tqdm(policy_json.keys()):
        unpacked_key = unpack_infoset_key_dense(int(k))
        decoded = _decode_fields(unpacked_key, policy_json[k]) # Pass policy_dist here
        data_rows.append(decoded)
    
    # Create DataFrame from all collected data at once
    df = pd.DataFrame(data_rows)
    return df

def extraction_policy_data(src_path="policy/avg_policy.json"):
    with open(src_path, "r") as f:
        policy_json = json.load(f)

    # Overview of first 5 infosets
    infoset_keys = list(policy_json.keys())
    for k in infoset_keys[:5]:
        unpacked_key = unpack_infoset_key_dense(int(k))
        decoded = _decode_fields(unpacked_key, policy_json[k]) # Pass policy_dist here

        print(f"Infoset: {k}")
        print(_format_actions(policy_json[k]))
        print(_format_fields(decoded))
        print("-" * 100)
        print()

    df = build_dataframe(policy_json)
    df.to_csv("policy/avg_policy.csv", index=False)

if __name__ == "__main__":
    extraction_policy_data()