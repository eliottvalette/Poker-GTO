# visualisation_Full.py
# Viz de la policy moyenne sauvée par cfr_solver.py
# - Charge policy/avg_policy.json
# - Décode les infosets (via infoset.unpack) pour agréger par phase/role/hand169
# - Sauvegarde des PNG dans viz/

from __future__ import annotations
import os, json, math
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt

from infoset import unpack_infoset_key_dense

# actions qu'on s'attend à voir
ACTIONS = ["FOLD","CHECK","CALL","RAISE","ALL-IN"]
ROLE_NAMES = ["SB","BB","BTN"]
PHASES = ["PREFLOP","FLOP","TURN","RIVER","SHOWDOWN"]

def load_policy(path: str) -> Dict[int, Dict[str, float]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # cfr_solver.py sauve des clés str(int_dense)
    return {int(k): v for k, v in raw.items()}

def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

def norm_on_legal(dist: Dict[str, float]) -> Dict[str, float]:
    # renormalise et borne aux actions connues
    vec = {a: dist.get(a, 0.0) for a in ACTIONS}
    s = sum(vec.values())
    if s <= 0:
        n = sum(1 for a in ACTIONS if a in dist)
        if n == 0:
            return {a: 0.0 for a in ACTIONS}
        p = 1.0 / n
        return {a: (p if a in dist else 0.0) for a in ACTIONS}
    return {a: vec[a]/s for a in ACTIONS}

def bar_mix_by_phase_role(pol: Dict[int, Dict[str,float]], outdir: str):
    # agrège probas moyennes par (phase, role)
    agg = {ph: {r: Counter() for r in range(3)} for ph in range(5)}
    cnt = {ph: Counter() for ph in range(5)}

    for k, dist in pol.items():
        f = unpack_infoset_key_dense(k)
        phase = int(f["phase"])
        role  = int(f["role"])
        probs = norm_on_legal(dist)
        agg[phase][role].update(probs)
        cnt[phase][role] = cnt[phase].get(role, 0) + 1

    ensure_dir(outdir)
    for ph in range(4):  # pas SHOWDOWN
        labels = ROLE_NAMES
        data = []
        for role in range(3):
            denom = max(1, cnt[ph][role])
            data.append([agg[ph][role][a]/denom for a in ACTIONS])

        data = np.array(data)  # shape: (3, |A|)
        x = np.arange(len(labels))
        width = 0.15

        plt.figure(figsize=(10,4))
        for i,a in enumerate(ACTIONS):
            plt.bar(x + (i-2)*width, data[:,i], width, label=a)
        plt.xticks(x, labels)
        plt.ylim(0,1)
        plt.title(f"Mix d'actions moyens — phase={PHASES[ph]}")
        plt.legend(ncol=5, fontsize=8, frameon=False)
        plt.ylabel("proba moyenne")
        plt.tight_layout()
        fname = os.path.join(outdir, f"bar_mix_{PHASES[ph].lower()}.png")
        plt.savefig(fname, dpi=160)
        plt.close()

def hand169_heatmaps_preflop(pol: Dict[int, Dict[str,float]], outdir: str):
    # 13x13 = ordre row-major identique à l'encodage (idx = i*13 + j)
    # On calcule, pour PREFLOP et par rôle, la proba M-O-Y-E-N-N-E de chaque action pour chaque hand169.
    PHASE_PREFLOP = 0
    grids = {role: {a: np.zeros((13,13), dtype=np.float64) for a in ACTIONS} for role in range(3)}
    counts = {role: np.zeros((13,13), dtype=np.int32) for role in range(3)}

    for k, dist in pol.items():
        f = unpack_infoset_key_dense(k)
        if int(f["phase"]) != PHASE_PREFLOP:
            continue
        role = int(f["role"])
        hidx = int(f["hand"])  # 0..168
        i = hidx // 13
        j = hidx % 13
        probs = norm_on_legal(dist)
        for a in ACTIONS:
            grids[role][a][i,j] += probs[a]
        counts[role][i,j] += 1

    ensure_dir(outdir)
    for role in range(3):
        # moyenne sur les occurrences de l'infoset (si 0, on laisse 0)
        C = counts[role]
        denom = np.where(C==0, 1, C)
        for a in ACTIONS:
            M = grids[role][a] / denom
            plt.figure(figsize=(6,5))
            plt.imshow(M, origin="upper", interpolation="nearest")
            plt.title(f"Préflop {ROLE_NAMES[role]} — P({a}) moyen (13x13)")
            plt.xlabel("r2 (A→2)")
            plt.ylabel("r1 (A→2)")
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.tight_layout()
            fname = os.path.join(outdir, f"preflop_{ROLE_NAMES[role].lower()}_{a.lower()}.png")
            plt.savefig(fname, dpi=160)
            plt.close()

def main():
    path = "policy/avg_policy.json"
    if not os.path.exists(path):
        print(f"[ERR] {path} introuvable. Entraîne d'abord cfr_solver.py")
        return
    pol = load_policy(path)
    print(f"[LOAD] policy: {len(pol)} infosets")
    ensure_dir("viz_full")

    print("[VIZ] mix d'actions par phasexrôle …")
    bar_mix_by_phase_role(pol, "viz_full")

    print("[VIZ] heatmaps 13x13 préflop …")
    hand169_heatmaps_preflop(pol, "viz_full")

    print("[OK] PNG écrits dans ./viz_full")

if __name__ == "__main__":
    main()
