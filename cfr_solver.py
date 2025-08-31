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
SAVE_EVERY = 0  # Sauvegarde tous les N itérations

# Actions fixes
ACTIONS = ["FOLD", "CHECK", "CALL", "RAISE", "ALL-IN"]
ACTION_IDX = {a: i for i, a in enumerate(ACTIONS)}
N_ACTIONS = len(ACTIONS)

def _quantize_dist(vec: list[float], keep_top_k: int = 3, eps: float = 1e-6):
    # vec = probabilités sur 5 actions (déjà normalisées)
    # garde top-k, quantifie sur 0..255, renvoie (mask, [q...]) où sum(q)=255
    items = [(i, vec[i]) for i in range(5) if vec[i] > eps]
    items.sort(key=lambda x: x[1], reverse=True)
    items = items[:keep_top_k]
    s = sum(p for _, p in items)
    if s <= 0:
        raise ValueError("Sum of probabilities is less than or equal to 0")
    probs = [p/s for _, p in items]
    qu = [int(round(p*255)) for p in probs]
    # ajuste la somme à 255
    diff = 255 - sum(qu)
    if diff != 0:
        # ajuste l'élément le plus grand
        j = max(range(len(qu)), key=lambda k: qu[k])
        qu[j] = max(0, min(255, qu[j] + diff))
    mask = 0
    order = []
    for i,_ in items:
        mask |= (1 << i)
        order.append(i)
    return mask, qu

# Utilitaires
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

# =========================
# Classe principale CFR+
# =========================
class CFRPlusSolver:
    def __init__(self, seed: int = 1, stacks=(100, 100, 100), hands_per_iter: int = 1):
        self.seed = seed
        self.stacks = stacks
        self.hands_per_iter = hands_per_iter

        # Chaque infoset stocke deux vecteurs fixes [FOLD,CHECK,CALL,RAISE,ALL-IN]
        self.regret_sum = defaultdict(lambda: [0.0] * N_ACTIONS)
        self.strategy_sum = defaultdict(lambda: [0.0] * N_ACTIONS)

        self.rng = random.Random(seed)

    # -------------------------
    # Environnement de jeu
    # -------------------------
    def new_game(self) -> PokerGameExpresso:
        init = GameInit()
        init.stacks_init = list(self.stacks)              # SB, BB, BTN
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
    def terminal_cev(game: PokerGameExpresso, hero_role: int) -> float:
        name = f"Player_{hero_role}"
        return float(game.net_stack_changes.get(name, 0.0))

    # -------------------------
    # Regret Matching+
    # -------------------------
    def strategy_from_regret(self, key: int, legal: List[str]) -> List[float]:
        vec = self.regret_sum[key]
        total = 0.0
        probs = [0.0] * N_ACTIONS

        for a in legal:
            i = ACTION_IDX[a]
            r = vec[i]
            if r > 0:
                probs[i] = r
                total += r

        if total <= 0.0:
            u = 1.0 / len(legal)
            for a in legal:
                probs[ACTION_IDX[a]] = u
        else:
            for a in legal:
                i = ACTION_IDX[a]
                probs[i] /= total
        return probs

    def sample_from(self, probs: List[float]) -> str:
        x = self.rng.random()
        c = 0.0
        for i, p in enumerate(probs):
            c += p
            if x <= c:
                return ACTIONS[i]
        return ACTIONS[-1]

    # -------------------------
    # Rollout
    # -------------------------
    def rollout_until_terminal(self, game: PokerGameExpresso, hero_role: int, reach: float) -> Tuple[float, float]:
        while game.current_phase != "SHOWDOWN":
            current_role = game.current_role
            current_player = game.players[current_role]

            key = build_infoset_key_fast(game, current_player)
            legal = self.legal_actions(game)
            if not legal:
                raise RuntimeError(f"[CFR+] Aucune action légale.\n{format_game_state_for_debug(game)}")

            probs = self.strategy_from_regret(key, legal)
            action = self.sample_from(probs)

            if current_role != hero_role:
                reach *= probs[ACTION_IDX[action]]

            game.process_action(current_player, action)
        return self.terminal_cev(game, hero_role), reach

    # -------------------------
    # Traverse CFR+
    # -------------------------
    def traverse(self, game: PokerGameExpresso, hero_role: int, reach: float) -> float:
        while game.current_phase != "SHOWDOWN":
            current_role = game.current_role
            current_player = game.players[current_role]

            key = build_infoset_key_fast(game, current_player)
            legal = self.legal_actions(game)
            if not legal:
                raise RuntimeError(f"[CFR+] Aucune action légale.\n{format_game_state_for_debug(game)}")

            if current_role == hero_role:
                probs = self.strategy_from_regret(key, legal)

                # Utilities par action
                utils = [0.0] * N_ACTIONS
                node_util = 0.0

                for a in legal:
                    i = ACTION_IDX[a]
                    snap = game.snapshot()
                    game.process_action(current_player, a)
                    u, _ = self.rollout_until_terminal(game, hero_role, reach)
                    game.restore(snap)

                    utils[i] = u
                    node_util += probs[i] * u

                # Update regrets
                regret_sum = self.regret_sum[key]
                strategy_sum = self.strategy_sum[key]
                for a in legal:
                    i = ACTION_IDX[a]
                    reg = utils[i] - node_util
                    val = regret_sum[i] + reach * reg
                    regret_sum[i] = val if val > 0.0 else 0.0
                    strategy_sum[i] += reach * probs[i]

                action = self.sample_from(probs)
                game.process_action(current_player, action)
                continue

            # Adversaire
            probs = self.strategy_from_regret(key, legal)
            action = self.sample_from(probs)
            reach *= probs[ACTION_IDX[action]]
            game.process_action(current_player, action)

        return self.terminal_cev(game, hero_role)

    # -------------------------
    # Entraînement
    # -------------------------
    def train(self, iterations: int = 1000) -> None:
        print(f"\n{'='*80}")
        print(f"DÉMARRAGE ENTRAÎNEMENT CFR+")
        print(f"{'='*80}")
        print(f"Stacks: SB={self.stacks[0]}BB, BB={self.stacks[1]}BB, BTN={self.stacks[2]}BB")
        print(f"Itérations: {iterations}")
        print(f"Mains par itération: {self.hands_per_iter}")
        print(f"Seed: {self.seed}")
        print(f"{'='*80}\n")

        start_time = time.time()
        os.makedirs('policy', exist_ok=True)

        with trange(1, iterations + 1, desc="CFR+ Training", unit="iter") as pbar:
            for it in pbar:
                self.rng.seed(self.seed + 7919 * it)

                for _ in range(self.hands_per_iter):
                    for hero in (0, 1, 2):
                        g = self.new_game()
                        self.traverse(g, hero_role=hero, reach=1.0)

                if SAVE_EVERY > 0 and (it % SAVE_EVERY == 0):
                    self.save_policy_json(f"policy/avg_policy_iter_{it}.json")

        self.save_policy_json("policy/avg_policy.json.gz")
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
        out = {}
        for k, vec in self.strategy_sum.items():
            total = sum(vec)
            if total <= 0:
                continue
            probs = [vec[i]/total for i in range(5)]
            mask, qu = _quantize_dist(probs, keep_top_k=3)
            if mask != 0:
                out[k] = [mask] + qu
        return out

    def save_policy_json(self, path: str) -> None:
        compact = self.extract_average_policy()
        out = {str(k): v for k, v in compact.items()}
        data = json.dumps(out, separators=(",",":"), ensure_ascii=False)
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write(data)
        if DEBUG_CFR:
            print(f"[SAVE] Policy gzip: {path} ({len(out)} infosets)")

    def warm_start_from_policy(self, path: str, weight: float = 1000.0):
        if not os.path.exists(path):
            return
        with gzip.open(path, "rt", encoding="utf-8") as f:
            raw = json.load(f)
        for k_str, entry in raw.items():
            k = int(k_str)
            if not isinstance(entry, list) or not entry or not isinstance(entry[0], int):
                continue
            mask = entry[0]
            qs = entry[1:]
            total = sum(qs)
            if total <= 0:
                continue
            vec = [0.0] * 5
            idx_q = 0
            for i in range(5):
                if (mask >> i) & 1:
                    q = qs[idx_q]
                    vec[i] = weight * (q / total)
                    idx_q += 1
            self.strategy_sum[k] = vec

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
    
    # Configuration
    seed = 42
    stacks = (100, 100, 100)  # SB, BB, BTN
    hands_per_iter = 8
    iterations = 10_000
    
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
    
    solver.warm_start_from_policy("policy/avg_policy.json.gz", weight=1000.0)

    if PROFILE:
        profiler = cProfile.Profile()
        profiler.enable()

    # Entraînement
    solver.train(iterations=iterations)

    if PROFILE:
        profiler.disable()
        profiler.dump_stats("profiling/cfr_solver_profile.prof")
    
    # Statistiques finales
    extraction_policy_data()
    
    print(f"\nEntraînement terminé avec succès!")
    print(f"Policy sauvegardée dans: policy/avg_policy.json.gz")
