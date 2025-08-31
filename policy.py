# policy.py
from __future__ import annotations
import json
import random
import gzip
from typing import Dict, List
from infoset import build_infoset_key_fast
from poker_game_expresso import PokerGameExpresso

ACTIONS = ["FOLD","CHECK","CALL","RAISE","ALL-IN"]

def _decode_compact_entry(entry: list[int]) -> Dict[str, float]:
    mask = entry[0]
    qs = entry[1:]
    total = sum(qs)
    if total <= 0:
        return {}
    dist = {}
    idx_q = 0
    for i, a in enumerate(ACTIONS):
        if (mask >> i) & 1:
            q = qs[idx_q]
            dist[a] = q / total
            idx_q += 1
    return dist

class AveragePolicy:
    def __init__(self, policy: Dict[int, Dict[str, float]], seed: int = 123):
        self.policy = policy
        self.rng = random.Random(seed)

    @staticmethod
    def load(path: str, seed: int = 123) -> "AveragePolicy":
        # GZIP + format compact OBLIGATOIRES
        with gzip.open(path, "rt", encoding="utf-8") as f:
            raw = json.load(f)
        pol: Dict[int, Dict[str, float]] = {}
        for k, v in raw.items():
            key = int(k)
            if not isinstance(v, list) or not v or not isinstance(v[0], int):
                continue
            dist = _decode_compact_entry(v)
            if dist:
                pol[key] = dist
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

    def sample(self, dist: Dict[str, float]) -> str:
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
        player = game.players[game.current_role]
        key = build_infoset_key_fast(game, player)
        legal = self.legal_actions(game)
        if not legal:
            raise ValueError(f"[POLICY] legal actions : {legal}")

        dist = self.policy.get(key)
        if not dist:
            p = 1.0 / len(legal)
            dist = {a: p for a in legal}
        else:
            dist = {a: dist.get(a, 0.0) for a in legal}
            s = sum(dist.values())
            if s <= 1e-12:
                p = 1.0 / len(legal)
                dist = {a: p for a in legal}
            else:
                dist = {a: v / s for a, v in dist.items()}

        return self.sample(dist)