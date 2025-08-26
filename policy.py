# policy.py
# Politique (chargement / action) basée sur la stratégie moyenne sauvegardée.

from __future__ import annotations
import json
import random
from typing import Dict, List
from infoset import build_infoset_key
from poker_game_expresso import PokerGameExpresso


class AveragePolicy:
    def __init__(self, policy: Dict[int, Dict[str, float]], seed: int = 123):
        self.policy = policy          # key(int) -> {action: prob}
        self.rng = random.Random(seed)

    @staticmethod
    def load(path: str, seed: int = 123) -> "AveragePolicy":
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        pol = {int(k): v for k, v in raw.items()}
        return AveragePolicy(pol, seed=seed)

    @staticmethod
    def legal_actions(game: PokerGameExpresso) -> List[str]:
        p = game.players[game.current_role]
        return game.update_available_actions(
            p,
            game.current_maximum_bet,
            game.number_raise_this_game_phase,
            game.main_pot,
            game.current_phase
        )

    def _sample(self, dist: Dict[str, float]) -> str:
        x = self.rng.random()
        c = 0.0
        last = None
        for a, p in dist.items():
            c += p
            last = a
            if x <= c:
                return a
        return last

    def act(self, game: PokerGameExpresso) -> str:
        # Retourne une action légale échantillonnée selon la politique moyenne.
        player = game.players[game.current_role]
        _, key = build_infoset_key(game, player)
        legal = self.legal_actions(game)
        if not legal:
            raise ValueError(f"[POLICY] legal actions : {legal}")

        dist = self.policy.get(key)
        if not dist:
            # inconnu → uniforme sur les actions légales
            p = 1.0 / len(legal)
            dist = {a: p for a in legal}
        else:
            # restreindre aux actions légales & renormaliser
            dist = {a: dist.get(a, 0.0) for a in legal}
            s = sum(dist.values())
            if s <= 1e-12:
                p = 1.0 / len(legal)
                dist = {a: p for a in legal}
            else:
                dist = {a: v / s for a, v in dist.items()}

        return self._sample(dist)
