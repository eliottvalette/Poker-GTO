# artificial_policy.py
from __future__ import annotations
import json
from typing import Dict
from infoset import unpack_infoset_key_dense, _LABELS_169

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
    # 1) charge ta range 169 (prob d'open)
    open_range = load_open_range()

    # 2) charge la policy existante comme squelette
    with open("policy/avg_policy.json", "r", encoding="utf-8") as f:
        base_policy: Dict[str, Dict[str, float]] = json.load(f)

    new_policy: Dict[str, Dict[str, float]] = {}

    touched = 0
    for k_str, dist in base_policy.items():
        k = int(k_str)
        fields = unpack_infoset_key_dense(k)
        phase = ID_TO_PHASE.get(fields["PHASE"], str(fields["PHASE"]))
        role  = fields["ROLE"]
        hand_idx = fields["HAND"]
        spr_q = fields["SPR"]

        
        label_169 = _169_LABEL[hand_idx]  # p.ex. "Q3s"
        p_open = float(open_range.get(label_169, 0.0))  # défaut 0

        # split de base : raise / fold
        p_raise = max(0.0, min(1.0, p_open))
        p_fold  = 1.0 - p_raise
        p_call  = 0.0
        p_allin = 0.0

        # option limp/call
        if LIMP_ALLOWED and LIMP_FRACTION > 0.0:
            move = min(p_raise, LIMP_FRACTION)
            p_raise -= move
            p_call  += move

        # option all-in si SPR bas et main premium
        label_no_suf = label_169  # déjà canon du 169
        if spr_q <= SPR_ALLIN_MAX_BUCKET and label_no_suf in ALLIN_HANDS and p_raise > 0.0:
            move = min(p_raise, p_raise * ALLIN_FRACTION_OF_RAISE)
            p_raise -= move
            p_allin += move

        # normalisation défensive
        s = p_fold + p_call + p_raise + p_allin
        if s <= 0:
            # fallback uniforme sur actions légales préflop (pas de CHECK)
            new_policy[k_str] = {"FOLD": 0.5, "RAISE": 0.5}
        else:
            new_policy[k_str] = {
                "FOLD":  p_fold / s,
                "CALL":  p_call / s if p_call > 0 else 0.0,
                "RAISE": p_raise / s if p_raise > 0 else 0.0,
                "ALL-IN":p_allin / s if p_allin > 0 else 0.0
            }
        touched += 1

    out_path = "policy/avg_policy_artificial.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(new_policy, f, indent=2, ensure_ascii=False)

    print(f"[OK] Infosets modifiés PREFLOP: {touched} / {len(base_policy)}")
    print(f"[SAVE] {out_path}")

if __name__ == "__main__":
    main()
