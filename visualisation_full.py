# visualisation_full.py
# Viz de la policy moyenne sauvée par cfr_solver.py
# - Charge policy/avg_policy.json
# - Décode les infosets (via infoset.unpack) pour agréger par phase/role/hand169
# - Sauvegarde des PNG dans viz/

from __future__ import annotations
import os, json
from collections import Counter
from typing import Dict
import numpy as np
import matplotlib.pyplot as plt

from infoset import unpack_infoset_key_dense

# actions et libellés
ACTIONS = ["FOLD","CHECK","CALL","RAISE","ALL-IN"]
ACTIONS_LOWER = ["fold","check","call","raise","all-in"]
ROLE_NAMES = ["SB","BB","BTN"]
PHASES = ["PREFLOP","FLOP","TURN","RIVER","SHOWDOWN"]

COLOR_PALETTE = ['#003049', '#006DAA', '#D62828', '#F77F00', '#FCBF49', '#EAE2B7']

# Palette modifiée pour regrouper les actions
ACTION_COLORS = {
    'fold':   '#006DAA',      # Bleu 
    'check':  '#97e29b',      # Vert
    'call':   '#97e29b',      # Même vert que check
    'raise':  '#D62828',      # Rouge
    'all-in': '#D62828'       # Même rouge que raise
}

# Mapping pour regrouper les actions dans la visualisation
ACTION_GROUPS = {
    'fold': 'fold',
    'check': 'check_call',
    'call': 'check_call', 
    'raise': 'raise_allin',
    'all-in': 'raise_allin'
}

# Couleurs pour les groupes d'actions
GROUP_COLORS = {
    'fold': '#006DAA',        # Bleu
    'check_call': '#97e29b',  # Vert
    'raise_allin': '#D62828'  # Rouge
}

class VisualizerFull:
    def __init__(self, policy_path: str):
        self.policy_path = policy_path
        self.policy = self.load_policy(policy_path)
        self.outdir = "viz_full"
        os.makedirs(self.outdir, exist_ok=True)

    def load_policy(self, path: str) -> Dict[int, Dict[str, float]]:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        # cfr_solver.py sauve des clés str(int_dense)
        return {int(k): v for k, v in raw.items()}

    def normalize_on_legal_actions(self, dist: Dict[str, float]) -> Dict[str, float]:
        """Renormalise les probas aux actions connues (ACTIONS)."""
        vec = {a: dist.get(a, 0.0) for a in ACTIONS}
        s = sum(vec.values())
        if s <= 0:
            n = sum(1 for a in ACTIONS if a in dist)
            if n == 0:
                return {a: 0.0 for a in ACTIONS}
            p = 1.0 / n
            return {a: (p if a in dist else 0.0) for a in ACTIONS}
        return {a: vec[a]/s for a in ACTIONS}

    # === ancienne bar_mix: conservée si utile ===
    def bar_mix_by_phase_role(self, pol: Dict[int, Dict[str,float]], outdir: str):
        # agrège probas moyennes par (phase, role)
        agg = {ph: {r: Counter() for r in range(3)} for ph in range(5)}
        cnt = {ph: Counter() for ph in range(5)}

        for k, dist in pol.items():
            f = unpack_infoset_key_dense(k)
            phase = int(f["phase"])
            role  = int(f["role"])
            probs = self.normalize_on_legal_actions(dist)
            agg[phase][role].update(probs)
            cnt[phase][role] = cnt[phase].get(role, 0) + 1

        agents = ROLE_NAMES  # SB, BB, BTN
        phases_names = ['preflop', 'flop', 'turn', 'river']
        
        # Actions groupées pour la visualisation
        grouped_actions = ['raise_allin', 'check_call', 'fold']
        grouped_labels = ['Raise/All-in', 'Check/Call', 'Fold']

        phase_action_freq = {agent: {ph: {a: 0.0 for a in grouped_actions} for ph in phases_names} for agent in agents}

        for ph_idx, ph_name in enumerate(phases_names):
            for role in range(3):
                denom = max(1, cnt[ph_idx][role])
                totals = {a: agg[ph_idx][role][a] / denom for a in ACTIONS}
                s = sum(totals.values())
                if s > 0:
                    totals = {a: totals[a]/s for a in ACTIONS}
                
                # Regroupement des actions
                phase_action_freq[agents[role]][ph_name]['raise_allin'] = totals.get('RAISE', 0.0) + totals.get('ALL-IN', 0.0)
                phase_action_freq[agents[role]][ph_name]['check_call'] = totals.get('CHECK', 0.0) + totals.get('CALL', 0.0)
                phase_action_freq[agents[role]][ph_name]['fold'] = totals.get('FOLD', 0.0)

        fig, ax2 = plt.subplots(figsize=(10, 5))
        x = np.arange(len(agents))
        width = 0.2

        phase_has_data = {
            ph_name: any(cnt[p_idx][role] > 0 for role in range(3))
            for p_idx, ph_name in enumerate(phases_names)
        }

        for p_idx, phase in enumerate(phases_names):
            bottom = np.zeros(len(agents))
            for action in grouped_actions:
                values = []
                for agent in agents:
                    total = sum(phase_action_freq[agent][phase].values())
                    freq = phase_action_freq[agent][phase][action] / total if total > 0 else 0.0
                    values.append(freq)

                bars = ax2.bar(
                    x + p_idx * width - width * 1.5,
                    values,
                    width,
                    bottom=bottom,
                    label=f'{grouped_labels[grouped_actions.index(action)]} ({phase})' if p_idx == 0 else "",
                    color=GROUP_COLORS[action],
                    alpha=0.8,
                )

                for idx, (val, b) in enumerate(zip(values, bars)):
                    if val > 0.08:
                        ax2.text(
                            b.get_x() + b.get_width()/2.0,
                            bottom[idx] + val/2.0,
                            f'{val*100:.0f}%',
                            ha='center', va='center', fontsize=7, color='black', rotation=90
                        )
                bottom += np.array(values)

            if not phase_has_data[phase]:
                group_center = np.mean(x + p_idx * width - width * 1.5)
                ax2.text(group_center, 0.5, 'no data', ha='center', va='center', fontsize=8, color='#888888', rotation=90)

        ax2.set_title('Fréquence des actions par position et par phase (actions groupées)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(agents)
        ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        ax2.set_ylim(0, 1)
        ax2.set_facecolor('#F6F6F6')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        fname = os.path.join(outdir, 'bar_mix_actions_by_phase_stacked.png')
        plt.savefig(fname, dpi=160)
        plt.close(fig)

    # === NOUVEAU : “mosaic heatmap” préflop par position ===
    def mosaic_heatmap_by_role(self, pol: Dict[int, Dict[str,float]], outdir: str, label_threshold: float = 0.15, phase: int = 0):
        """
        Un PNG par position (SB/BB/BTN) :
        - grille 13x13 (169 mains)
        - chaque case = bandeaux horizontaux proportionnels aux pourcentages d'actions groupées
        """

        # Agrégation: pour chaque role et hidx (0..168) -> probas moyennes d'actions
        role_hand_action_sum = {role: {h: {a:0.0 for a in ACTIONS} for h in range(169)} for role in range(3)}
        role_hand_count      = {role: {h: 0 for h in range(169)} for role in range(3)}

        for k, dist in pol.items():
            f = unpack_infoset_key_dense(k)
            if int(f["phase"]) != phase:
                continue
            role = int(f["role"])
            hidx = int(f["hand"])  # 0..168 (row-major: i=0..12, j=0..12)
            probs = self.normalize_on_legal_actions(dist)
            for a in ACTIONS:
                role_hand_action_sum[role][hidx][a] += probs[a]
            role_hand_count[role][hidx] += 1

        # labels axes
        card_labels = ["A","K","Q","J","T","9","8","7","6","5","4","3","2"]
        
        # Actions groupées pour la visualisation
        grouped_actions = ['raise_allin', 'check_call', 'fold']
        grouped_labels = ['Raise/All-in', 'Check/Call', 'Fold']

        for role in range(3):
            # figure
            fig_w = 12
            fig_h = 10
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            ax.set_xlim(0, 13)
            ax.set_ylim(0, 13)
            ax.invert_yaxis()  # (0,0) en haut-gauche pour ressembler à la heatmap classique
            ax.set_aspect('equal')

            # fond + grille légère
            ax.set_facecolor('#FFFFFF')
            for i in range(14):
                ax.axhline(i, color='#DDDDDD', linewidth=0.6)
                ax.axvline(i, color='#DDDDDD', linewidth=0.6)

            # dessiner chaque case
            for i in range(13):       # ligne
                for j in range(13):   # colonne
                    hidx = i*13 + j
                    count = role_hand_count[role][hidx]
                    if count == 0:
                        # Case sans data: griser léger
                        rect = plt.Rectangle((j, i), 1, 1, facecolor='#F3F3F3', edgecolor='#E0E0E0', linewidth=0.4)
                        ax.add_patch(rect)
                        continue

                    totals = role_hand_action_sum[role][hidx]
                    # moyenne
                    s = sum(totals.values())
                    if s > 0:
                        avg = {a: totals[a] / count for a in ACTIONS}
                        # renormalise au cas où (défensif)
                        z = sum(avg.values())
                        if z > 0:
                            for a in ACTIONS:
                                avg[a] /= z
                    else:
                        avg = {a: 0.0 for a in ACTIONS}

                    # Regroupement des actions pour la visualisation
                    grouped_probs = {
                        'raise_allin': avg.get('RAISE', 0.0) + avg.get('ALL-IN', 0.0),
                        'check_call': avg.get('CHECK', 0.0) + avg.get('CALL', 0.0),
                        'fold': avg.get('FOLD', 0.0)
                    }

                    # bandeaux horizontaux (de haut en bas ou bas en haut : ici haut->bas)
                    y0 = i
                    x0 = j
                    cell_width = 1.0
                    cell_height = 1.0

                    # On empile dans l'ordre des actions groupées
                    cursor_x = x0
                    for action_group in grouped_actions:
                        frac = grouped_probs[action_group]
                        if frac <= 0:
                            continue
                        band_w = cell_width * frac
                        band = plt.Rectangle(
                            (cursor_x, y0), band_w, cell_height,
                            facecolor=GROUP_COLORS[action_group], edgecolor=None
                        )
                        ax.add_patch(band)

                        # label si assez large
                        if frac >= label_threshold:
                            ax.text(
                                cursor_x + band_w/2.0, y0 + cell_height/2.0,
                                f"{int(round(frac*100))}%",
                                ha='center', va='center',
                                fontsize=6, color='#111111'
                            )
                        cursor_x += band_w

                    # contour fin de la cellule
                    border = plt.Rectangle((x0, y0), 1, 1, fill=False, edgecolor='#BBBBBB', linewidth=0.4)
                    ax.add_patch(border)

            # ticks et labels
            ax.set_xticks(np.arange(13)+0.5)
            ax.set_yticks(np.arange(13)+0.5)
            ax.set_xticklabels(card_labels, fontsize=9)
            ax.set_yticklabels(card_labels, fontsize=9)
            ax.set_xlabel("r2 (A→2)")
            ax.set_ylabel("r1 (A→2)")
            ax.set_title(f"{PHASES[phase]} — {ROLE_NAMES[role]} — mix d'actions groupées par main (169)")

            # Légende actions groupées
            legend_handles = [plt.Rectangle((0,0),1,1, color=GROUP_COLORS[a]) for a in grouped_actions]
            ax.legend(legend_handles, grouped_labels, ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.06), frameon=False)

            plt.tight_layout()
            fname = os.path.join(outdir, f"{PHASES[phase]}_mosaic_{ROLE_NAMES[role].lower()}.png")
            plt.savefig(fname, dpi=170, bbox_inches='tight')
            plt.close(fig)

def main():
    visualizer = VisualizerFull("policy/avg_policy.json")
    print(f"[LOAD] policy: {len(visualizer.policy)} infosets")

    # (optionnel) graphe barres par phase/position
    visualizer.bar_mix_by_phase_role(visualizer.policy, "viz_full")

    for phase in range(4):  # 0=PREFLOP, 1=FLOP, 2=TURN, 3=RIVER
        print(f"[MOSAIC] Génération phase {PHASES[phase]}")
        visualizer.mosaic_heatmap_by_role(visualizer.policy, "viz_full", label_threshold=0.15, phase=phase)

    print("[OK] PNG écrits dans ./viz_full")

if __name__ == "__main__":
    main()
