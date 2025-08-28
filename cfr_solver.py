# cfr_solver.py
# ============================================================
# CFR+ externe (external-sampling) 3-handed, full-street, min-raise only.
# S'appuie sur PokerGameExpresso + infoset.build_infoset_key_fast.
# - No-Fallback: on raise si aucune action légale (état détaillé).
# - No recursion: parcours itératif (while) + rollouts itératifs.
# - No abbreviations: noms explicites et messages clairs.
# ============================================================

from __future__ import annotations
import random
import json
import os
import time
from collections import defaultdict
from typing import Dict, List, Tuple
import cProfile

from tqdm import trange
from poker_game_expresso import PokerGameExpresso, GameInit
from infoset import build_infoset_key_fast

# =========================
# Configuration et constantes
# =========================
DEBUG_CFR = True
PROFILE = False
SAVE_EVERY = 10_000  # Sauvegarde tous les N itérations

# =========================
# Types et utilitaires
# =========================
Action = str
Key = int  # infoset dense key

def _default_action_dict():
    return defaultdict(float)

def _key_to_hex_string(key_int: int) -> str:
    """Convertit une clé dense en string hexadécimale lisible."""
    return f"{key_int:016X}"

def _format_game_state_for_debug(game: PokerGameExpresso) -> str:
    player = game.players[game.current_role]
    board = " ".join(str(c) for c in game.community_cards)
    return (
        f"phase={game.current_phase} role={game.current_role} "
        f"pot={game.main_pot:.2f} max_bet={game.current_maximum_bet:.2f} "
        f"raises={game.number_raise_this_game_phase} last_raise={game.last_raise_amount:.2f}\n"
        f"player={player.name} stack={player.stack} cur_bet={player.current_player_bet} "
        f"active={player.is_active} folded={player.has_folded} "
        f"all_in={player.is_all_in} acted={player.has_acted}\n"
        f"board=[{board}]"
    )

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
        self.random_generator = self.rng
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
        current_player = game.players[game.current_role]
        return game.update_available_actions(
            current_player,
            game.current_maximum_bet,
            game.number_raise_this_game_phase,
            game.main_pot,
            game.current_phase
        )

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
        """Calcule la stratégie actuelle basée sur les regrets CFR+ (tronqués à 0)."""
        positive_regrets = {action: max(0.0, self.regret_sum[key].get(action, 0.0)) for action in legal}
        total_positive_regret = sum(positive_regrets.values())
        if total_positive_regret <= 0.0:
            uniform_prob = 1.0 / len(legal) if legal else 0.0
            return {action: uniform_prob for action in legal}
        return {action: positive_regrets[action] / total_positive_regret for action in legal}

    def _sample_from(self, distribution: Dict[Action, float]) -> Action:
        """Échantillonne une action selon la distribution de probabilités"""
        # distribution supposée normalisée
        random_sample = self.random_generator.random()
        cumulative_probability = 0.0
        last_action = None
        for action, prob in distribution.items():
            cumulative_probability += prob
            last_action = action
            if random_sample <= cumulative_probability:
                return action
        return last_action  # garde-fou (erreurs d'arrondi)

    # =========================
    # Rollout itératif jusqu'au terminal (sans récursion)
    # =========================
    def _rollout_until_terminal(
        self, game: PokerGameExpresso, hero_role: int, reach_probability_of_others: float
    ) -> Tuple[float, float]:
        """
        Poursuit la partie jusqu'au SHOWDOWN en échantillonnant les actions selon la stratégie courante
        pour tous les joueurs (y compris le héros). Itératif (boucle while).
        Retourne (utility_hero, updated_reach_probability_of_others).
        """
        while game.current_phase != "SHOWDOWN":
            current_role = game.current_role
            current_player = game.players[current_role]

            key = build_infoset_key_fast(game, current_player)
            legal_actions = self._legal_actions(game)
            if not legal_actions:
                state = _format_game_state_for_debug(game)
                raise RuntimeError(
                    "[CFR+] Aucune action légale pour le joueur courant — état incohérent.\n"
                    f"{state}"
                )

            strategy_distribution = self._strategy_from_regret(key, legal_actions)
            chosen_action = self._sample_from(strategy_distribution)
            if current_role != hero_role:
                reach_probability_of_others *= strategy_distribution[chosen_action]

            game.process_action(current_player, chosen_action)

        return self._terminal_cev(game, hero_role), reach_probability_of_others

    # =========================
    # Traversal CFR+ (external sampling)
    # =========================
    def _traverse(self, game: PokerGameExpresso, hero_role: int, reach_probability_of_others: float) -> float:
        """
        External-sampling CFR+ multi-rues :
        - Adversaires : on échantillonne et on multiplie reach_probability_of_others.
        - Héros : on évalue TOUTES les actions (snapshot/restore + rollout terminal),
                on met à jour regrets/stratégie, PUIS on échantillonne une action
                du héros et on CONTINUE (pas de return prématuré).
        Retourne la valeur terminale du héros.
        """
        while game.current_phase != "SHOWDOWN":
            current_role = game.current_role
            current_player = game.players[current_role]

            key = build_infoset_key_fast(game, current_player)
            legal_actions = self._legal_actions(game)
            if not legal_actions:
                state = _format_game_state_for_debug(game)
                raise RuntimeError(
                    "[CFR+] Aucune action légale pour le joueur courant — état incohérent.\n"
                    f"{state}"
                )

            if current_role == hero_role:
                # 1) Stratégie courante (sigma) depuis regrets+
                strategy_distribution = self._strategy_from_regret(key, legal_actions)

                # 2) Utilités d'action via rollout terminal
                action_utilities: Dict[Action, float] = {}
                for action in legal_actions:
                    snap = game.snapshot()
                    game.process_action(current_player, action)
                    utility, _ = self._rollout_until_terminal(
                        game, hero_role, reach_probability_of_others
                    )
                    action_utilities[action] = utility
                    game.restore(snap)

                # 3) Utility du nœud (espérance sous sigma)
                node_utility = sum(
                    strategy_distribution[a] * action_utilities[a] for a in legal_actions
                )

                # 4) Update CFR+ regrets (tronqués à 0) pondérés par reach des AUTRES
                for action in legal_actions:
                    regret = action_utilities[action] - node_utility
                    self.regret_sum[key][action] = max(
                        0.0,
                        self.regret_sum[key].get(action, 0.0) + reach_probability_of_others * regret,
                    )

                # 5) Accumule stratégie moyenne (pondérée par reach des AUTRES)
                for action in legal_actions:
                    self.strategy_sum[key][action] += (
                        reach_probability_of_others * strategy_distribution[action]
                    )

                # 6) Échantillonne une action du héros pour CONTINUER le parcours
                chosen_action = self._sample_from(strategy_distribution)
                game.process_action(current_player, chosen_action)
                # Note : on NE multiplie PAS reach_probability_of_others par sigma_héros

                continue

            # Adversaire : on échantillonne et on met à jour reach des AUTRES
            strategy_distribution = self._strategy_from_regret(key, legal_actions)
            chosen_action = self._sample_from(strategy_distribution)
            reach_probability_of_others *= strategy_distribution[chosen_action]
            game.process_action(current_player, chosen_action)

        return self._terminal_cev(game, hero_role)


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
            for iter_idx in pbar:
                pbar.set_description(f"CFR+ Iter {iter_idx}/{iterations}")
                
                self.random_generator.seed(self.seed + 7919 * iter_idx)

                for _ in trange(self.hands_per_iter, desc=f"  Hands iter {iter_idx}", leave=False):
                    for hero in (0, 1, 2):  # 0=SB,1=BB,2=BTN
                        game = self._new_game()
                        self._traverse(game, hero_role=hero, reach_probability_of_others=1.0)

                if SAVE_EVERY > 0 and (iter_idx % SAVE_EVERY == 0):
                    save_path = f"policy/avg_policy_iter_{iter_idx}.json"
                    self.save_policy_json(save_path)
                    if DEBUG_CFR:
                        print(f"[SAVE] Policy sauvegardée à l'itération {iter_idx} -> {save_path}")

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
            total = sum(counts.values())
            if total > 0:
                policy[key] = {action: counts[action] / total for action in counts}
            else:
                # fallback: regret-matching+ uniforme implicite
                # on normalise sur les actions existantes dans le noeud
                actions_in_node = list(counts.keys())
                if not actions_in_node:
                    continue
                uniform_prob = 1.0 / len(actions_in_node)
                policy[key] = {action: uniform_prob for action in actions_in_node}
        return policy

    # =========================
    # Sauvegarde/Chargement
    # =========================
    def save_policy_json(self, path: str) -> None:
        """Sauvegarde la politique moyenne au format JSON"""
        avg_policy_dict = self.extract_average_policy()
        # clés JSON = str(key dense)
        out = {str(k): v for k, v in avg_policy_dict.items()}
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
            key_str = _key_to_hex_string(key)
            print(f"  {key_str}: {list(actions.keys())}")
        
        print(f"{'='*60}")


# =========================
# Exécution principale
# =========================
if __name__ == "__main__":
    print("CFR+ Solver - 3-handed NLHE")
    print("=" * 50)
    
    # Configuration
    seed = 42
    stacks = (100, 100, 100)  # SB, BB, BTN
    hands_per_iter = 1
    iterations = 40_000
    
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

    # Load policy
    solver.load_policy_json("policy/avg_policy.json")
    
    if PROFILE:
        profiler = cProfile.Profile()
        profiler.enable()

    # Entraînement
    solver.train(iterations=iterations)

    if PROFILE:
        profiler.disable()
        profiler.dump_stats("profiling/cfr_solver_profile.prof")
    
    # Statistiques finales
    solver.print_policy_stats()
    
    print(f"\nEntraînement terminé avec succès!")
    print(f"Policy sauvegardée dans: policy/avg_policy.json")
