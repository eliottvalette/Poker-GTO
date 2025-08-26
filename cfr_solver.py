# cfr_solver.py
# ============================================================
# CFR+ externe (external-sampling) 3-handed, full-street, min-raise only.
# S'appuie sur PokerGameExpresso + infoset.build_infoset_key.
# ============================================================

from __future__ import annotations
import random
import copy
import json
import os
import time
from collections import defaultdict
from typing import Dict, List, Tuple

from tqdm import tqdm, trange
from poker_game_expresso import PokerGameExpresso, GameInit
from infoset import build_infoset_key

# =========================
# Configuration et constantes
# =========================
DEBUG_CFR = True
SAVE_EVERY = 100  # Sauvegarde tous les N itérations

# =========================
# Types et utilitaires
# =========================
Action = str
Key = int  # infoset dense key

def _default_action_dict():
    return defaultdict(float)

def _key_to_str(k: int) -> str:
    """Convertit une clé dense en string hexadécimale lisible"""
    return f"{k:016X}"

# =========================
# Classe principale CFR+
# =========================
class CFRPlusSolver:
    def __init__(
        self,
        seed: int = 1,
        stacks: Tuple[int, int, int] = (100, 100, 100),  # SB, BB, BTN
        hands_per_iter: int = 1
    ):
        self.seed = seed
        self.stacks = stacks
        self.hands_per_iter = hands_per_iter

        # Stockage des regrets et stratégies
        self.regret_sum: Dict[Key, Dict[Action, float]] = defaultdict(_default_action_dict)
        self.strategy_sum: Dict[Key, Dict[Action, float]] = defaultdict(_default_action_dict)

        # RNG et statistiques
        self.rng = random.Random(seed)
        self.stats = {
            'total_infosets': 0,
            'total_actions': 0,
            'training_time': 0.0
        }

    # =========================
    # Utilitaires d'environnement
    # =========================
    def _new_game(self) -> PokerGameExpresso:
        """Initialise une nouvelle main aléatoire et poste les blindes"""
        init = GameInit()
        init.stacks_init = list(self.stacks)              # SB, BB, BTN
        init.total_bets_init = [0, 0, 0]
        init.current_bets_init = [0, 0, 0]
        init.active_init = [True, True, True]
        init.has_acted_init = [False, False, False]
        init.main_pot = 0
        init.phase = "PREFLOP"
        init.community_cards = []
        
        # Seed global random pour que le paquet soit reproductible
        random.seed(self.rng.randrange(10**9))
        game = PokerGameExpresso(init)
        game.deal_small_and_big_blind()
        return game

    @staticmethod
    def _legal_actions(game: PokerGameExpresso) -> List[Action]:
        """Récupère les actions légales pour le joueur courant"""
        p = game.players[game.current_role]
        return game.update_available_actions(
            p,
            game.current_maximum_bet,
            game.number_raise_this_game_phase,
            game.main_pot,
            game.current_phase
        )

    @staticmethod
    def _step_clone(game: PokerGameExpresso, action: Action) -> PokerGameExpresso:
        """Clone le jeu, applique l'action et retourne le nouvel état"""
        g2 = copy.deepcopy(game)
        p2 = g2.players[g2.current_role]
        # process_action mutera et avancera la partie, y compris tirage du board via l'env
        g2.process_action(p2, action)
        return g2

    @staticmethod
    def _terminal_cev(game: PokerGameExpresso, hero_role: int) -> float:
        """Calcule la valeur terminale pour le héros (en BB)"""
        # handle_showdown a déjà rempli net_stack_changes
        name = f"Player_{hero_role}"
        return float(game.net_stack_changes.get(name, 0.0))

    # =========================
    # Régret Matching+ et échantillonnage
    # =========================
    def _strategy_from_regret(self, key: Key, legal: List[Action]) -> Dict[Action, float]:
        """Calcule la stratégie actuelle basée sur les regrets CFR+"""
        # CFR+: regrets tronqués à 0
        rplus = {a: max(0.0, self.regret_sum[key].get(a, 0.0)) for a in legal}
        s = sum(rplus.values())
        if s <= 0.0:
            p = 1.0 / len(legal) if legal else 0.0
            return {a: p for a in legal}
        return {a: rplus[a] / s for a in legal}

    def _sample_from(self, dist: Dict[Action, float]) -> Action:
        """Échantillonne une action selon la distribution de probabilités"""
        # dist supposée normalisée
        x = self.rng.random()
        c = 0.0
        last_a = None
        for a, p in dist.items():
            c += p
            last_a = a
            if x <= c:
                return a
        return last_a  # garde-fou (erreurs d'arrondi)

    # =========================
    # Traversal CFR+ (external sampling)
    # =========================
    def _traverse(self, game: PokerGameExpresso, hero_role: int, reach_others: float) -> float:
        """Traverse l'arbre de jeu avec external sampling CFR+"""
        # Terminal ?
        if game.current_phase == "SHOWDOWN":
            return self._terminal_cev(game, hero_role)

        cur = game.current_role
        player = game.players[cur]
        _, key = build_infoset_key(game, player)

        legal = self._legal_actions(game)

        # Cas rare: aucun move légal (skip via progression de phase)
        if not legal:
            # On force l'env à vérifier la phase, puis on continue
            g2 = copy.deepcopy(game)
            g2.check_phase_completion()
            if g2.current_phase == "SHOWDOWN":
                return self._terminal_cev(g2, hero_role)
            return self._traverse(g2, hero_role, reach_others)

        if cur != hero_role:
            # Joueur "autre" → on échantillonne une seule action selon la stratégie courante
            sigma = self._strategy_from_regret(key, legal)
            a = self._sample_from(sigma)
            g2 = self._step_clone(game, a)
            return self._traverse(g2, hero_role, reach_others * sigma[a])

        # Joueur "héros" → on évalue toutes les actions pour le calcul de regret
        sigma = self._strategy_from_regret(key, legal)
        util_a = {}
        for a in legal:
            g2 = self._step_clone(game, a)
            util_a[a] = self._traverse(g2, hero_role, reach_others)

        u_node = sum(sigma[a] * util_a[a] for a in legal)

        # CFR+ update (regrets cumulés, tronqués à 0)
        for a in legal:
            regret = util_a[a] - u_node
            self.regret_sum[key][a] = max(0.0, self.regret_sum[key].get(a, 0.0) + reach_others * regret)

        # Moyenne des stratégies (pondérée par reach des AUTRES joueurs)
        for a in legal:
            self.strategy_sum[key][a] += reach_others * sigma[a]

        return u_node

    # =========================
    # Entraînement principal
    # =========================
    def train(self, iterations: int = 1000) -> None:
        """Entraîne le solveur CFR+ sur le nombre d'itérations spécifié"""
        print(f"\n{'='*80}")
        print(f"DÉMARRAGE ENTRAÎNEMENT CFR+")
        print(f"{'='*80}")
        print(f"Stacks: SB={self.stacks[0]}BB, BB={self.stacks[1]}BB, BTN={self.stacks[2]}BB")
        print(f"Itérations: {iterations}")
        print(f"Mains par itération: {self.hands_per_iter}")
        print(f"Seed: {self.seed}")
        print(f"Sauvegarde tous les: {SAVE_EVERY} itérations")
        print(f"{'='*80}\n")

        start_time = time.time()
        
        # Créer le dossier policy s'il n'existe pas
        os.makedirs('policy', exist_ok=True)

        # Barre de progression principale
        with trange(1, iterations + 1, desc="CFR+ Training", unit="iter") as pbar:
            for it in pbar:
                # Mise à jour de la description de la barre
                pbar.set_description(f"CFR+ Iter {it}/{iterations}")
                
                # petite randomisation de seed main par main
                self.rng.seed(self.seed + 7919 * it)

                # Barre de progression pour les mains
                for hand_idx in trange(self.hands_per_iter, desc=f"  Hands iter {it}", leave=False):
                    # alterner chaque siège comme "héros" (external sampling par joueur)
                    for hero in (0, 1, 2):  # 0=SB,1=BB,2=BTN
                        game = self._new_game()
                        self._traverse(game, hero_role=hero, reach_others=1.0)

                # Sauvegarde périodique
                if SAVE_EVERY > 0 and (it % SAVE_EVERY == 0):
                    save_path = f"policy/avg_policy_iter_{it}.json"
                    self.save_policy_json(save_path)
                    if DEBUG_CFR:
                        print(f"[SAVE] Policy sauvegardée à l'itération {it} -> {save_path}")

                # Mise à jour des statistiques
                self.stats['total_infosets'] = len(self.strategy_sum)
                total_actions = sum(len(actions) for actions in self.strategy_sum.values())
                self.stats['total_actions'] = total_actions
                
                # Mise à jour de la barre avec les stats
                pbar.set_postfix({
                    'Infosets': self.stats['total_infosets'],
                    'Actions': total_actions
                })

        # Temps total d'entraînement
        self.stats['training_time'] = time.time() - start_time
        
        # Sauvegarde finale
        final_path = "policy/avg_policy.json"
        self.save_policy_json(final_path)
        
        # Résumé final
        self._print_training_summary(iterations, final_path)

    def _print_training_summary(self, iterations: int, final_path: str):
        """Affiche un résumé clair de l'entraînement"""
        print(f"\n{'='*80}")
        print(f"ENTRAÎNEMENT CFR+ TERMINÉ")
        print(f"{'='*80}")
        print(f"Itérations complétées: {iterations}")
        print(f"Temps total: {self.stats['training_time']:.2f}s")
        print(f"Infosets uniques: {self.stats['total_infosets']}")
        print(f"Actions totales: {self.stats['total_actions']}")
        print(f"Policy finale: {final_path}")
        print(f"{'='*80}")

    # =========================
    # Politique moyenne
    # =========================
    def extract_average_policy(self) -> Dict[int, Dict[str, float]]:
        """Extrait la politique moyenne depuis les stratégies cumulées"""
        policy = {}
        for key, counts in self.strategy_sum.items():
            s = sum(counts.values())
            if s > 0:
                policy[key] = {a: counts[a] / s for a in counts}
            else:
                # fallback: regret-matching+ uniforme implicite
                # on normalise sur les actions existantes dans le noeud
                acts = list(counts.keys())
                if not acts:
                    continue
                p = 1.0 / len(acts)
                policy[key] = {a: p for a in acts}
        return policy

    # =========================
    # Sauvegarde/Chargement
    # =========================
    def save_policy_json(self, path: str) -> None:
        """Sauvegarde la politique moyenne au format JSON"""
        pol = self.extract_average_policy()
        # clés JSON = str(key dense)
        out = {str(k): v for k, v in pol.items()}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        
        if DEBUG_CFR:
            print(f"[SAVE] Policy sauvegardée: {path} ({len(out)} infosets)")

    @staticmethod
    def load_policy_json(path: str) -> Dict[int, Dict[str, float]]:
        """Charge une politique depuis un fichier JSON"""
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if DEBUG_CFR:
            print(f"[LOAD] Policy chargée: {path} ({len(raw)} infosets)")
        return {int(k): v for k, v in raw.items()}

    # =========================
    # Statistiques et analyse
    # =========================
    def print_policy_stats(self):
        """Affiche des statistiques sur la politique apprise"""
        policy = self.extract_average_policy()
        
        print(f"\n{'='*60}")
        print(f"STATISTIQUES DE LA POLITIQUE")
        print(f"{'='*60}")
        print(f"Nombre d'infosets: {len(policy)}")
        
        # Compter les actions par infoset
        action_counts = defaultdict(int)
        for infoset_policy in policy.values():
            action_counts[len(infoset_policy)] += 1
        
        print(f"Distribution des actions par infoset:")
        for action_count, count in sorted(action_counts.items()):
            print(f"  {action_count} actions: {count} infosets")
        
        # Afficher quelques exemples d'infosets
        print(f"\nExemples d'infosets (premiers 5):")
        for i, (key, actions) in enumerate(list(policy.items())[:5]):
            key_str = _key_to_str(key)
            print(f"  {key_str}: {list(actions.keys())}")
        
        print(f"{'='*60}")


# =========================
# Exécution principale
# =========================
if __name__ == "__main__":
    print("CFR+ Solver - 3-handed NLHE")
    print("=" * 50)
    
    # Configuration
    seed = 7
    stacks = (100, 100, 100)  # SB, BB, BTN
    hands_per_iter = 1
    iterations = 200
    
    print(f"Configuration:")
    print(f"  Seed: {seed}")
    print(f"  Stacks: {stacks}")
    print(f"  Mains par itération: {hands_per_iter}")
    print(f"  Itérations: {iterations}")
    print()
    
    # Créer et entraîner le solveur
    solver = CFRPlusSolver(
        seed=seed, 
        stacks=stacks, 
        hands_per_iter=hands_per_iter
    )
    
    # Entraînement
    solver.train(iterations=iterations)
    
    # Statistiques finales
    solver.print_policy_stats()
    
    print(f"\nEntraînement terminé avec succès!")
    print(f"Policy sauvegardée dans: policy/avg_policy.json")
