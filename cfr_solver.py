# cfr_solver.py
# ============================================================
# CFR+ externe (external-sampling) 3-handed, full-street, min-raise only.
# S'appuie sur PokerGameExpresso + infoset.build_infoset_key_fast.
# ============================================================

from __future__ import annotations
import random
import json
import os
import time
import gzip
from collections import defaultdict
from typing import List, Tuple
import cProfile

from tqdm import trange
from poker_game_expresso import PokerGameExpresso, GameInit
from infoset import build_infoset_key_fast
from stats_policy import extraction_policy_data

DEBUG_CFR = True
PROFILE = False
SAVE_EVERY = 0  # sauvegarde tous les N itérations

# Actions fixes
ACTIONS = ["FOLD", "CHECK", "CALL", "RAISE", "ALL-IN"]
ACTION_INDEX = {action_name: index for index, action_name in enumerate(ACTIONS)}
N_ACTIONS = len(ACTIONS)


def quantize_distribution(probabilities: list[float], keep_top_k: int = 3, eps: float = 1e-6):
    """
    Convertit un vecteur de probabilités (déjà normalisé) en une représentation compacte.
    - On garde uniquement les top-k probabilités les plus élevées.
    - On les convertit en entiers de 0..255.
    - On ajuste pour que la somme des entiers soit exactement 255.
    - On retourne un bitmask qui indique quelles actions sont présentes, 
      et la liste des valeurs entières correspondantes.
    """
    # filtrer les actions non négligeables
    action_probability_pairs = [
        (action_index, probabilities[action_index])
        for action_index in range(N_ACTIONS)
        if probabilities[action_index] > eps
    ]
    action_probability_pairs.sort(key=lambda x: x[1], reverse=True)
    action_probability_pairs = action_probability_pairs[:keep_top_k]

    total_probability = sum(prob for _, prob in action_probability_pairs)
    if total_probability <= 0:
        raise ValueError("Somme des probabilités <= 0")

    normalized_probabilities = [prob / total_probability for _, prob in action_probability_pairs]
    quantized_values = [int(round(prob * 255)) for prob in normalized_probabilities]

    difference = 255 - sum(quantized_values)
    if difference != 0:
        index_of_max = max(range(len(quantized_values)), key=lambda k: quantized_values[k])
        quantized_values[index_of_max] = max(0, min(255, quantized_values[index_of_max] + difference))

    # construire un masque de bits pour savoir quelles actions sont présentes
    bitmask = 0
    for action_index, _ in action_probability_pairs:
        bitmask |= (1 << action_index)

    return bitmask, quantized_values


def format_game_state_for_debug(game: PokerGameExpresso) -> str:
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


class CFRPlusSolver:
    def __init__(self, seed: int = 1, stacks=(100, 100, 100)):
        self.seed = seed
        self.stacks = stacks

        self.regret_sum = defaultdict(lambda: [0.0] * N_ACTIONS)
        self.strategy_sum = defaultdict(lambda: [0.0] * N_ACTIONS)
        self.visit_count = defaultdict(int)

        self.rng = random.Random(seed)

    # -------------------------
    # Environnement de jeu
    # -------------------------
    def new_game(self) -> PokerGameExpresso:
        init = GameInit()
        init.stacks_init = list(self.stacks)
        init.total_bets_init = [0, 0, 0]
        init.current_bets_init = [0, 0, 0]
        init.active_init = [True, True, True]
        init.has_acted_init = [False, False, False]
        init.main_pot = 0
        init.phase = "PREFLOP"
        init.community_cards = []

        random.seed(self.rng.randrange(10**9))
        game = PokerGameExpresso(init)
        game.deal_small_and_big_blind()
        return game

    @staticmethod
    def legal_actions(game: PokerGameExpresso) -> List[str]:
        current_player = game.players[game.current_role]
        return game.update_available_actions(
            current_player,
            game.current_maximum_bet,
            game.number_raise_this_game_phase,
            game.main_pot,
            game.current_phase
        )

    @staticmethod
    def terminal_expected_value(game: PokerGameExpresso, hero_role: int) -> float:
        name = f"Player_{hero_role}"
        return float(game.net_stack_changes.get(name, 0.0))

    # -------------------------
    # Regret Matching+
    # -------------------------
    def strategy_from_regret(self, infoset_key: int, legal_actions: List[str]) -> List[float]:
        regret_vector = self.regret_sum[infoset_key]

        probabilities = [0.0] * N_ACTIONS
        total_positive_regret = 0.0

        for action_name in legal_actions:
            index = ACTION_INDEX[action_name]
            regret_value = regret_vector[index]
            if regret_value > 0:
                probabilities[index] = regret_value
                total_positive_regret += regret_value

        if total_positive_regret <= 0.0:
            uniform_probability = 1.0 / len(legal_actions)
            for action_name in legal_actions:
                probabilities[ACTION_INDEX[action_name]] = uniform_probability
        else:
            for action_name in legal_actions:
                index = ACTION_INDEX[action_name]
                probabilities[index] /= total_positive_regret

        return probabilities

    def sample_from(self, probabilities: List[float]) -> str:
        random_value = self.rng.random()
        cumulative = 0.0

        for index, probability in enumerate(probabilities):
            cumulative += probability
            if random_value <= cumulative:
                return ACTIONS[index]

        return ACTIONS[-1]

    # -------------------------
    # Rollout
    # -------------------------
    def rollout_until_terminal(self, game: PokerGameExpresso, hero_role: int, reach_probability: float) -> Tuple[float, float]:
        while game.current_phase != "SHOWDOWN":
            current_role = game.current_role
            current_player = game.players[current_role]

            infoset_key = build_infoset_key_fast(game, current_player)
            legal_actions = self.legal_actions(game)

            if not legal_actions or len(legal_actions) < 2:
                raise RuntimeError(f"[CFR+] Aucune action légale.\n{format_game_state_for_debug(game)}")

            probabilities = self.strategy_from_regret(infoset_key, legal_actions)
            chosen_action = self.sample_from(probabilities)

            if current_role != hero_role:
                reach_probability *= probabilities[ACTION_INDEX[chosen_action]]

            game.process_action(current_player, chosen_action)

        return self.terminal_expected_value(game, hero_role), reach_probability

    # -------------------------
    # Traverse CFR+
    # -------------------------
    def traverse(self, game: PokerGameExpresso, hero_role: int, reach_probability: float) -> float:
        while game.current_phase != "SHOWDOWN":
            current_role = game.current_role
            current_player = game.players[current_role]

            infoset_key = build_infoset_key_fast(game, current_player)
            legal_actions = self.legal_actions(game)

            if not legal_actions or len(legal_actions) < 2:
                raise RuntimeError(f"[CFR+] Aucune action légale.\n{format_game_state_for_debug(game)}")

            if current_role == hero_role:
                probabilities = self.strategy_from_regret(infoset_key, legal_actions)

                action_utilities = [0.0] * N_ACTIONS
                node_expected_utility = 0.0
            

                snapshot = game.snapshot()
                for action_name in legal_actions:
                    index = ACTION_INDEX[action_name]
                    game.process_action(current_player, action_name)
                    utility, _ = self.rollout_until_terminal(game, hero_role, reach_probability)
                    game.restore(snapshot)

                    action_utilities[index] = utility
                    node_expected_utility += probabilities[index] * utility

                regret_vector = self.regret_sum[infoset_key]
                strategy_vector = self.strategy_sum[infoset_key]

                for action_name in legal_actions:
                    index = ACTION_INDEX[action_name]

                    advantage = action_utilities[index] - node_expected_utility
                    updated_value = regret_vector[index] + reach_probability * advantage
                    regret_vector[index] = updated_value if updated_value > 0.0 else 0.0

                    strategy_vector[index] += reach_probability * probabilities[index]

                self.visit_count[infoset_key] += 1

                chosen_action = self.sample_from(probabilities)
                game.process_action(current_player, chosen_action)
                continue

            # Adversaire
            probabilities = self.strategy_from_regret(infoset_key, legal_actions)
            chosen_action = self.sample_from(probabilities)

            reach_probability *= probabilities[ACTION_INDEX[chosen_action]]
            game.process_action(current_player, chosen_action)

        return self.terminal_expected_value(game, hero_role)

    # -------------------------
    # Entraînement
    # -------------------------
    def train(self, iterations: int = 1000) -> None:
        print(f"\n{'='*80}")
        print(f"DÉMARRAGE ENTRAÎNEMENT CFR+")
        print(f"{'='*80}")
        print(f"Stacks: {self.stacks}")
        print(f"Itérations: {iterations}")
        print(f"Seed: {self.seed}")
        print(f"{'='*80}\n")

        start_time = time.time()
        os.makedirs('policy', exist_ok=True)

        with trange(1, iterations + 1, desc="CFR+ Training", unit="iter") as progress_bar:
            for iteration_index in progress_bar:
                self.rng.seed(self.seed + 7919 * iteration_index)

                for hero_role in (0, 1, 2):
                    game = self.new_game()
                    self.traverse(game, hero_role=hero_role, reach_probability=1.0)

                if SAVE_EVERY > 0 and (iteration_index % SAVE_EVERY == 0):
                    self.save_policy_json(f"policy/avg_policy_iter_{iteration_index}.json.gz")

        self.save_policy_json("policy/avg_policy.json.gz")
        self.save_policy_json("ui/public/avg_policy.json.gz")
        self.print_training_summary(iterations, "policy/avg_policy.json.gz")

        end_time = time.time()
        print(f"Temps total: {end_time - start_time:.2f}s")

    def print_training_summary(self, iterations: int, final_path: str):
        print(f"\n{'='*80}")
        print(f"ENTRAÎNEMENT CFR+ TERMINÉ")
        print(f"{'='*80}")
        print(f"Itérations complétées: {iterations}")
        print(f"Policy finale: {final_path}")
        print(f"{'='*80}")

    # -------------------------
    # Politique moyenne
    # -------------------------
    def extract_average_policy(self):
        extracted_policy = {}
        for infoset_key, strategy_vector in self.strategy_sum.items():
            total = sum(strategy_vector)
            if total <= 0:
                raise ValueError(f"[EXTRACT] Total <= 0: {total}. Infoset key: {infoset_key}")

            probabilities = [strategy_vector[action_index] / total for action_index in range(N_ACTIONS)]
            bitmask, quantized_values = quantize_distribution(probabilities, keep_top_k=3)

            if bitmask != 0:
                extracted_policy[infoset_key] = [bitmask] + quantized_values

        return extracted_policy

    def save_policy_json(self, path: str) -> None:
        compact_policy = self.extract_average_policy()
        serialized = {}

        for infoset_key, encoded_policy in compact_policy.items():
            serialized[str(infoset_key)] = {
                "policy": encoded_policy,
                "visits": min(self.visit_count[infoset_key], 120)
            }

        data = json.dumps(serialized, separators=(",", ":"), ensure_ascii=False)
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write(data)

        if DEBUG_CFR:
            print(f"[SAVE] Policy gzip: {path} ({len(serialized)} infosets)")

    def warm_start_from_policy(self, path: str):
        if not os.path.exists(path):
            print(f"[WARN] Policy not found: {path}")
            return

        print(f"[LOAD] Policy found: {path}")
        with gzip.open(path, "rt", encoding="utf-8") as f:
            raw = json.load(f)

        for key_str, entry in raw.items():
            infoset_key = int(key_str)

            if "policy" not in entry or "visits" not in entry:
                raise ValueError(f"[LOAD] Policy or visits not found: {entry}. Infoset key: {infoset_key}")

            bitmask = entry["policy"][0]
            quantized_values = entry["policy"][1:]
            visit_count_value = entry["visits"]

            total_quantized = sum(quantized_values)
            if total_quantized <= 0 or visit_count_value <= 0:
                raise ValueError(f"[LOAD] Total quantized or visit count value <= 0: {total_quantized} or {visit_count_value}. Infoset key: {infoset_key}") 

            reconstructed_strategy = [0.0] * N_ACTIONS
            index_quantized = 0
            for action_index in range(N_ACTIONS):
                if (bitmask >> action_index) & 1:
                    q = quantized_values[index_quantized]
                    reconstructed_strategy[action_index] = (visit_count_value * q) / total_quantized
                    index_quantized += 1

            self.strategy_sum[infoset_key] = reconstructed_strategy
            self.visit_count[infoset_key] = visit_count_value
        
        if DEBUG_CFR:
            for index, (infoset_key, strategy_vector) in enumerate(self.strategy_sum.items()):
                print(f"[LOAD] Strategy vector: {strategy_vector}")
                print(f"[LOAD] Visit count: {self.visit_count[infoset_key]}")
                print(f"[LOAD] Infoset key: {infoset_key}")
                if index >= 3:
                    break

    @staticmethod
    def load_policy_json(path: str):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            raw = json.load(f)

        if DEBUG_CFR:
            print(f"[LOAD] Policy chargée: {path} ({len(raw)} infosets)")

        return {int(k): v for k, v in raw.items()}


# =========================
# Exécution principale
# =========================
if __name__ == "__main__":
    import gc
    gc.collect()

    print("CFR+ Solver - 3-handed NLHE")
    print("=" * 50)

    seed = int(time.time())
    stacks = (100, 100, 100)
    iterations = 200_000

    print("Configuration:")
    print(f"  Seed: {seed}")
    print(f"  Stacks: {stacks}")
    print(f"  Itérations: {iterations}")
    print()

    solver = CFRPlusSolver(seed=seed, stacks=stacks)
    solver.warm_start_from_policy("policy/avg_policy.json.gz")

    if PROFILE:
        profiler = cProfile.Profile()
        profiler.enable()

    solver.train(iterations=iterations)

    if PROFILE:
        profiler.disable()
        profiler.dump_stats("profiling/cfr_solver_profile.prof")

    extraction_policy_data()

    print(f"\nEntraînement terminé avec succès!")
    print(f"Policy sauvegardée dans: policy/avg_policy.json.gz")
