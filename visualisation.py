import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Set, Tuple, Callable, List
from classes import card_rank, card_suit
from utils import load_ranges_json
from classes import ALL_COMBOS

def visualise_ranges(ranges: Dict[str, Set[Tuple[int,int]]], coverage_pct: Callable[[Set[Tuple[int,int]]], float], iter_num: int, evolution_data: Dict[str, List[float]] = None):
    # Créer le dossier viz s'il n'existe pas
        os.makedirs("viz", exist_ok=True)
        
        # Configuration seaborn pour un style plus clair
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        
        # 1. Créer un PNG séparé pour le bar plot des couvertures
        plt.figure(figsize=(12, 8))
        coverages = [coverage_pct(ranges[name]) for name in ranges.keys()]
        names = list(ranges.keys())
        
        bars = plt.bar(names, coverages, color=sns.color_palette("husl", 5))
        plt.title("Couverture des Ranges par Position", fontsize=16, fontweight='bold')
        plt.ylabel("Couverture (%)", fontsize=12)
        plt.ylim(0, max(coverages) * 1.1)
        
        # Ajouter les valeurs sur les barres
        for bar, coverage in zip(bars, coverages):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{coverage:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('viz/barplot_couverture.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Créer des PNGs séparés pour chaque matrice (sans bar plot)
        ranks = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
        
        # Créer la matrice pour chaque range
        for range_name, combo_set in ranges.items():
            # Créer une matrice 13x13
            matrix = np.zeros((13, 13))
            
            # Compter les combos par case (rang1, rang2)
            for combo in combo_set:
                r1, r2 = card_rank(combo[0]), card_rank(combo[1])
                s1, s2 = card_suit(combo[0]), card_suit(combo[1])
                
                # Déterminer le rang haut et bas
                hi, lo = (r1, r2) if r1 >= r2 else (r2, r1)
                
                # Convertir en indices (A->0, K->1, ..., 2->12)
                i = 12 - (hi - 2)  # A en haut
                j = 12 - (lo - 2)  # 2 en bas
                
                if hi == lo:
                    matrix[i, i] += 1/6                 # 0..6 pour les paires
                elif s1 == s2:
                    matrix[i, j] += 1/4                 # 0..4 suited (triangle supérieur)
                else:
                    matrix[j, i] += 1/12                 # 0..12 offsuit (triangle inférieur)
            
            # Limiter le nombre de décimales à 2
            matrix = np.round(matrix, 2)
            
            # Créer le heatmap pour ce range
            plt.figure(figsize=(10, 8))
            sns.heatmap(matrix, annot=True, fmt='g', cmap='RdYlBu_r', 
                       xticklabels=ranks, yticklabels=ranks,
                       cbar_kws={'label': 'Nombre de combos dans cette case'})
            plt.title(f"Matrice {range_name}", fontsize=16, fontweight='bold')
            plt.xlabel("Rang 2", fontsize=12)
            plt.ylabel("Rang 1", fontsize=12)
            
            # Sauvegarder individuellement
            plt.tight_layout()
            if iter_num == 0:
                plt.savefig(f'viz/matrix_{range_name.replace(" ", "_").replace("vs", "vs_")}.png', 
                            dpi=300, bbox_inches='tight')
            else:
                os.makedirs(f'viz/viz_iter_{iter_num}', exist_ok=True)
                plt.savefig(f'viz/viz_iter_{iter_num}/matrix_{range_name.replace(" ", "_").replace("vs", "vs_")}.png', 
                            dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Créer la matrice globale combinée sur 5 ranges
        global_matrix = np.zeros((13, 13))
        for combo_set in ranges.values():
            for combo in combo_set:
                r1, r2 = card_rank(combo[0]), card_rank(combo[1])
                s1, s2 = card_suit(combo[0]), card_suit(combo[1])
                
                # Déterminer le rang haut et bas
                hi, lo = (r1, r2) if r1 >= r2 else (r2, r1)
                
                # Convertir en indices (A->0, K->1, ..., 2->12)
                i = 12 - (hi - 2)  # A en haut
                j = 12 - (lo - 2)  # 2 en bas
                
                if hi == lo:
                    global_matrix[i, i] += 1/6/5                 # Paires
                elif s1 == s2:
                    global_matrix[i, j] += 1/4/5                 # Suited
                else:
                    global_matrix[j, i] += 1/12/5                 # Offsuit
        
        # Limiter le nombre de décimales à 2
        global_matrix = np.round(global_matrix, 2)
        
        # Afficher la matrice globale
        plt.figure(figsize=(10, 8))
        sns.heatmap(global_matrix, annot=True, fmt='g', cmap='RdYlBu_r',
                   xticklabels=ranks, yticklabels=ranks,
                   cbar_kws={'label': 'Nombre de ranges contenant ce combo'})
        plt.title("Matrice globale - Tous les ranges combinés", fontsize=16, fontweight='bold')
        plt.xlabel("Rang 2", fontsize=12)
        plt.ylabel("Rang 1", fontsize=12)
        
        plt.tight_layout()
        if iter_num == 0:
            plt.savefig('viz/matrix_globale.png', dpi=300, bbox_inches='tight')
        else:
            os.makedirs(f'viz/viz_iter_{iter_num}', exist_ok=True)
            plt.savefig(f'viz/viz_iter_{iter_num}/matrix_globale.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Créer le graphique des courbes d'évolution des couvertures
        if evolution_data and len(evolution_data.get('BTN_shove', [])) > 1:
            plt.figure(figsize=(14, 8))
            
            # Convertir les données d'évolution en pourcentages
            iterations = list(range(1, len(evolution_data['BTN_shove']) + 1))
            
            # Mapper les noms des ranges pour l'affichage
            range_names = {
                'BTN_shove': 'BTN shove',
                'SB_call_vs_BTN': 'SB call vs BTN',
                'BB_call_vs_BTN': 'BB call vs BTN',
                'SB_shove': 'SB shove',
                'BB_call_vs_SB': 'BB call vs SB'
            }
            
            colors = sns.color_palette("husl", 5)
            for i, (key, display_name) in enumerate(range_names.items()):
                if key in evolution_data and len(evolution_data[key]) > 1:
                    # Convertir en pourcentages
                    percentages = [100.0 * count / len(ALL_COMBOS) for count in evolution_data[key]]
                    plt.plot(iterations, percentages, marker='o', linewidth=2, markersize=6, 
                           label=display_name, color=colors[i])
            
            plt.title("Évolution des Couvertures par Itération", fontsize=16, fontweight='bold')
            plt.xlabel("Itération", fontsize=12)
            plt.ylabel("Couverture (%)", fontsize=12)
            plt.ylim(0, 100)
            plt.grid(True, alpha=0.3)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xticks(iterations)
            
            # Ajuster les limites Y pour une meilleure visualisation
            all_percentages = []
            for key in evolution_data.keys():
                if key in evolution_data and len(evolution_data[key]) > 1:
                    percentages = [100.0 * count / len(ALL_COMBOS) for count in evolution_data[key]]
                    all_percentages.extend(percentages)
            
            if all_percentages:
                plt.ylim(0, max(all_percentages) * 1.1)
            
            plt.tight_layout()
            if iter_num == 0:
                plt.savefig('viz/courbes_evolution.png', dpi=300, bbox_inches='tight')
            else:
                plt.savefig(f'viz/viz_iter_{iter_num}/courbes_evolution.png', dpi=300, bbox_inches='tight')
            plt.close()
        
if __name__ == "__main__":

    def coverage_pct(combos_set: Set[Tuple[int,int]]) -> float:
        return 100.0 * len(combos_set) / len(ALL_COMBOS)
    
    ranges = load_ranges_json("ranges/ranges.json")
    visualise_ranges(ranges, coverage_pct, iter_num=0)