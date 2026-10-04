"""Local Python engine service for Test Live. Bind only to loopback."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import random
import threading
from typing import Any
import uuid

from actions import ACTION_IDS, legal_actions
from cfr_solver import sample_index
from infoset import observe
from scripts.benchmark_policy import BOT_NAMES, scripted_action
from tournament import TournamentState

CONTRACT_VERSION = 2


@dataclass
class TableSession:
    tournament: TournamentState
    hero: int
    opponents: str
    rng: random.Random
    revision: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


class PokerService:
    def __init__(self, average_policy=None, max_sessions: int = 64):
        if average_policy is not None and average_policy.objective != "tournament_winner":
            raise ValueError("Test Live requires a tournament_winner average-policy model")
        self.average_policy = average_policy
        self.max_sessions = max_sessions
        self.sessions: dict[str, TableSession] = {}
        self.lock = threading.Lock()

    def _bots(self, session: TableSession) -> None:
        t = session.tournament
        count = 0
        while not t.hand.terminal and t.current_player != session.hero:
            count += 1
            if count > 100:
                raise ValueError("Bot action budget exceeded: 100 decisions per hand")
            scripted_action(session.opponents, t.hand, session.rng).apply(t.hand)

    def new(self, seed: int, hero: int = 2, opponents: str = "random") -> dict:
        if type(seed) is not int or type(hero) is not int or hero not in (0, 1, 2) or opponents not in BOT_NAMES:
            raise ValueError(f"Invalid table configuration: seed={seed}, hero={hero}, opponents={opponents}")
        t = TournamentState(rng=random.Random(seed))
        t.start_hand()
        session = TableSession(t, hero, opponents, random.Random(seed + 1))
        self._bots(session)
        session_id = str(uuid.uuid4())
        with self.lock:
            if len(self.sessions) >= self.max_sessions:
                raise ValueError(f"Local session limit reached ({self.max_sessions}); close a table or restart the service")
            result = self.view(session_id, session)
            self.sessions[session_id] = session
        return result

    def command(self, session_id: str, revision: int, operation: str, action: str | None = None) -> dict:
        if type(revision) is not int:
            raise ValueError(f"Revision must be an integer, got {revision!r}")
        if session_id not in self.sessions:
            raise KeyError(f"Unknown session {session_id}; create a new tournament")
        session = self.sessions[session_id]
        with session.lock:
            if revision != session.revision:
                raise ValueError(f"Stale revision {revision}; expected {session.revision}")
            candidate = TableSession(session.tournament.clone(), session.hero, session.opponents,
                                     copy.deepcopy(session.rng), session.revision)
            t = candidate.tournament
            if operation == "action":
                if t.current_player != session.hero:
                    raise ValueError(f"Hero {session.hero} is not acting; actor={t.current_player}")
                t.act(action)
            elif operation == "next":
                t.start_hand()
            else:
                raise ValueError(f"Invalid table operation {operation}")
            self._bots(candidate)
            candidate.revision += 1
            result = self.view(session_id, candidate)
            session.tournament, session.rng = candidate.tournament, candidate.rng
            session.revision = candidate.revision
            return result

    def close(self, session_id: str) -> None:
        with self.lock:
            if session_id not in self.sessions:
                raise KeyError(f"Unknown session {session_id}")
            del self.sessions[session_id]

    def view(self, session_id: str, session: TableSession) -> dict[str, Any]:
        t, h = session.tournament, session.tournament.hand
        players = []
        for i in t.original_players:
            p = h.players.get(i)
            stack = t.stacks[i] if p is None else p.stack
            players.append({"player_id": i, "stack_bb": stack,
                            "active": i in t.active, "position": None if p is None else p.position,
                            "folded": False if p is None else p.folded,
                            "bet_bb": 0 if p is None else p.street_bet,
                            "cards": list(p.cards) if p is not None and (i == session.hero or (h.showdown and not p.folded)) else []})
        policy = {"status": "unavailable", "reason": "No compatible tournament average policy loaded", "probabilities": None}
        if self.average_policy is not None and not h.terminal and h.current_player == session.hero:
            try:
                probabilities = self.average_policy.query(observe(t))
                policy = {"status": "experimental", "reason": "Approximation confidence is uncalibrated",
                          "probabilities": dict(zip(ACTION_IDS, probabilities))}
            except KeyError as error:
                policy["reason"] = str(error)
        return {"version": CONTRACT_VERSION, "session_id": session_id, "revision": session.revision,
                "hero": session.hero, "hand_number": t.hand_number, "button": t.button,
                "active_players": list(t.active), "players": players, "board": h.board,
                "street": h.street, "pot_bb": h.pot, "to_call_bb": 0 if h.terminal else h.to_call(),
                "actor": h.current_player, "hand_terminal": h.terminal,
                "tournament_terminal": t.terminal, "winner": t.winner,
                "hand_results_bb": {i: h.utility(i) if i in h.players else 0.0
                                    for i in t.original_players} if h.terminal else {},
                "hero_result_bb": (next(p["stack_bb"] for p in players if p["player_id"] == session.hero)
                                   if h.terminal else t.stacks[session.hero]) - 25,
                "total_chips_bb": t.total_chips, "blinds": asdict(t.blinds),
                "legal_actions": [asdict(a) for a in legal_actions(h)] if h.current_player == session.hero else [],
                "history": [asdict(e) for e in h.history], "policy": policy}


def serve(host: str = "127.0.0.1", port: int = 8765, average_policy=None) -> None:
    if host != "127.0.0.1":
        raise ValueError(f"Local engine service requires loopback binding, got {host}")
    service = PokerService(average_policy)
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            status = 200
            try:
                if self.headers.get("Origin") is not None:
                    raise ValueError("Direct browser requests are excluded; use the same-origin Next.js proxy")
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= 10000:
                    raise ValueError(f"Invalid JSON body length: {length}")
                payload = json.loads(self.rfile.read(length))
                if self.path == "/table/new":
                    result = service.new(payload["seed"], payload.get("hero", 2), payload.get("opponents", "random"))
                elif self.path == "/table/close":
                    service.close(payload["session_id"])
                    result = {"closed": True}
                elif self.path in ("/table/action", "/table/next"):
                    result = service.command(payload["session_id"], payload["revision"], self.path.rsplit("/", 1)[-1], payload.get("action"))
                else:
                    raise ValueError(f"Unknown endpoint {self.path}")
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
                status, result = 400, {"error": str(error)}
            except Exception as error:
                status, result = 500, {"error": f"{type(error).__name__}: {error}"}
            body = json.dumps(result, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
    http_server = ThreadingHTTPServer((host, port), Handler)
    print(f"Poker engine listening on http://{host}:{port}; policy={'unavailable' if average_policy is None else 'experimental average model'}", flush=True)
    try:
        http_server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        http_server.server_close()


if __name__ == "__main__":
    path = os.environ.get("POKER_AVERAGE_POLICY")
    policy = None
    if path is not None:
        from ml.deep_cfr import NeuralAveragePolicy
        policy = NeuralAveragePolicy(path)
    serve(average_policy=policy)
