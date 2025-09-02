// ui/src/components/TestTable.tsx
"use client";
import React, { useEffect, useState, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { PokerGame, buildInfosetKeyFast, type Action } from "@/lib/game";
import { normalize, type Policy } from "@/lib/policy";
import PokerTableFrame from "@/components/PokerTableFrame";

type Seat = 0|1|2;

type ActionHistoryEntry = {
  phase: string;
  player: string;
  action: string;
  timestamp: number;
};

function makeNewGame(): PokerGame {
  const g = new PokerGame({
    stacks: [100,100,100],
    totalBets: [0,0,0],
    currentBets: [0,0,0],
    active: [true,true,true],
    hasActed: [false,false,false],
    mainPot: 0,
    phase: "PREFLOP",
    community: [],
  });
  g.deal_private();
  g.deal_blinds();
  return g;
}

function sampleAction(dist: Record<string, number>, legal: Action[]): Action {
  const mix = normalize(Object.fromEntries(legal.map(a=>[a, dist[a] ?? 0])));
  const xs = legal.map(a => mix[a]);
  const s = xs.reduce((a,b)=>a+b,0);
  let r = Math.random() * (s > 0 ? s : 1);
  for (let i=0;i<legal.length;i++){
    r -= xs[i] || 0;
    if (r<=0) return legal[i];
  }
  return legal[legal.length-1];
}

export default function TestTable({ policy }: { policy: Policy | null }) {
  const [heroSeat, setHeroSeat] = useState<Seat>(2);
  const [game, setGame] = useState<PokerGame>(() => makeNewGame());
  const [actionHistory, setActionHistory] = useState<ActionHistoryEntry[]>([]);
  const [sessionPnL, setSessionPnL] = useState(0);
  const [handId, setHandId] = useState(1);
  const [scoredHandId, setScoredHandId] = useState<number | null>(null);
  const [, force] = useState(0); // re-render

  const addToHistory = useCallback((player: string, action: string) => {
    setActionHistory(prev => [...prev, {
      phase: game.current_phase,
      player,
      action,
      timestamp: Date.now()
    }]);
  }, [game.current_phase]);

  const stepBotsForward = useCallback((g: PokerGame) => {
    if (!policy) return;
    // boucle jusqu'au héros ou showdown
    while (g.current_phase!=="SHOWDOWN") {
      const p = g.players[g.current_role];
      if (p.role===heroSeat) break;
      const legal = g.update_available_actions(p);
      if (legal.length===0) break;
      const key = buildInfosetKeyFast(g, p);
      const entry = policy[key];
      const dist = entry?.dist ?? {};
      const choice = sampleAction(dist, legal);
      g.process_action(p, choice);
      addToHistory(p.name, choice);
      // continue tant que ce n'est pas au héros
    }
  }, [policy, heroSeat, addToHistory]);

  function newHand() {
    const g = makeNewGame();
    setGame(g);
    setActionHistory([]);
    setHandId(h => h + 1);
    force(n=>n+1);
  }

  useEffect(() => {
    if (!policy) return;
    const g = game;
    stepBotsForward(g);
    force(n=>n+1);
  }, [policy, game, stepBotsForward]);

  const hero = game.players[heroSeat];

  function onHeroAction(a: Action) {
    if (game.current_phase==="SHOWDOWN") return;
    const legal = game.update_available_actions(hero);
    if (!legal.includes(a)) return;
    game.process_action(hero, a);
    addToHistory(hero.name, a);
    stepBotsForward(game);
    force(n=>n+1);
  }

  useEffect(() => {
    newHand();
  }, [heroSeat]);

  // créditer le P&L du héros quand un showdown se termine
  useEffect(() => {
    if (game.current_phase === "SHOWDOWN" && scoredHandId !== handId) {
      const heroName = game.players[heroSeat].name;
      const delta = game.net_stack_changes[heroName] ?? 0;
      setSessionPnL(v => v + delta);
      setScoredHandId(handId);
    }
  }, [game.current_phase, handId, heroSeat, scoredHandId, game]); 

  const boardStr = game.community.map(c=>c.toString()).join(" ");

  return (
    <div className="space-y-4">
      <Card>
        <PokerTableFrame
          seats={[
            {
              id: 0,
              label: "SB",
              stack: `${game.players[0].stack.toFixed(1)} BB`,
              smallBlind: true,
              active: !game.players[0].has_folded,
              cards:
                heroSeat === 0
                  ? game.players[0].cards.map(c => c.toString())
                  : game.current_phase === "SHOWDOWN"
                  ? game.players[0].cards.map(c => c.toString())
                  : ["XX", "XX"], // face down
            },
            {
              id: 1,
              label: "BB",
              stack: `${game.players[1].stack.toFixed(1)} BB`,
              bigBlind: true,
              active: !game.players[1].has_folded,
              cards:
                heroSeat === 1
                  ? game.players[1].cards.map(c => c.toString())
                  : game.current_phase === "SHOWDOWN"
                  ? game.players[1].cards.map(c => c.toString())
                  : ["XX", "XX"],
            },
            {
              id: 2,
              label: "BTN",
              stack: `${game.players[2].stack.toFixed(1)} BB`,
              active: !game.players[2].has_folded,
              cards:
                heroSeat === 2
                  ? game.players[2].cards.map(c => c.toString())
                  : game.current_phase === "SHOWDOWN"
                  ? game.players[2].cards.map(c => c.toString())
                  : ["XX", "XX"],
            },
          ]}
          potLabel={`${game.main_pot.toFixed(2)} BB`}
          heroSeat={heroSeat}
          board={boardStr}
        >
          {/* barre d'actions */}
          <div className="grid grid-cols-3 gap-2 justify-items-center">
            <div className="flex gap-2">
              <Button variant={heroSeat===0?"default":"secondary"} onClick={()=>setHeroSeat(0)}>Héro: SB</Button>
              <Button variant={heroSeat===1?"default":"secondary"} onClick={()=>setHeroSeat(1)}>Héro: BB</Button>
              <Button variant={heroSeat===2?"default":"secondary"} onClick={()=>setHeroSeat(2)}>Héro: BTN</Button>
            </div>

            <Button onClick={newHand} className="w-35">Nouvelle main</Button>

            <div>
              {game.current_phase!=="SHOWDOWN" && game.current_role===heroSeat && (
                <div className="flex gap-2">
                  {game.update_available_actions(hero).map(a=>(
                    <Button
                      key={a}
                      onClick={()=>onHeroAction(a)}
                      variant="default"
                    >
                      {a}
                    </Button>
                  ))}
                </div>
              )}
            </div>
          </div>
        </PokerTableFrame>
        <div className="flex flex-row gap-4 border-t border-border">
          <div className=" bg-card/40 p-4 border-r border-border w-4/5">
            <div className="mb-2 text-center font-semibold">Showdown — Résultats</div>

            <div className="grid grid-cols-1 gap-2 sm:grid-cols-3">
              {game.players.map((p) => {
                const delta = game.net_stack_changes[p.name] ?? 0;
                const win = delta > 0;
                const even = delta === 0;

                return (
                  <div key={p.name} className="rounded-lg bg-background/40 p-3 ring-1 ring-border">
                    <div className="mb-1 flex items-center justify-between">
                      <span className="inline-flex items-center gap-2 text-sm font-medium">
                        <span className="grid h-6 w-6 place-items-center rounded-full bg-muted text-[0.7rem]">
                          {p.name}
                        </span>
                        {win && <span className="text-xs">🏆</span>}
                      </span>
                      <span
                        className={
                          win
                            ? "rounded-md bg-emerald-500/15 px-2 py-0.5 text-xs font-semibold text-emerald-400"
                            : even
                            ? "rounded-md bg-muted px-2 py-0.5 text-xs text-muted-foreground"
                            : "rounded-md bg-rose-500/15 px-2 py-0.5 text-xs font-semibold text-rose-400"
                        }
                      >
                        {win ? "+" : ""}{delta.toFixed(2)} BB
                      </span>
                    </div>

                    <div className="flex items-center justify-between text-xs text-muted-foreground">
                      <span>Stack final</span>
                      <span className="font-mono">{game.final_stacks[p.name].toFixed(2)} BB</span>
                    </div>
                  </div>
                );
              })}
            </div>

            <div className="mt-3 text-center text-xs text-muted-foreground">
              Pot distribué :{" "}
              {Object.values(game.net_stack_changes)
                .filter((x) => x > 0)
                .reduce((a, b) => a + b, 0)
                .toFixed(2)}{" "}
              BB
            </div>
            
          </div>
          <div className="flex flex-col justify-center items-center gap-2 w-1/5">
            <h1>Total P&L (Héro)</h1>
            <div className="text-sm text-muted-foreground">
              Héro: {game.players[heroSeat].name}
            </div>
            <div
              className={
                sessionPnL > 0
                  ? "font-semibold text-emerald-400"
                  : sessionPnL < 0
                  ? "font-semibold text-rose-400"
                  : "font-semibold text-muted-foreground"
              }
            >
              {sessionPnL >= 0 ? "+" : ""}
              {sessionPnL.toFixed(2)} BB
            </div>
            <Button variant="secondary" onClick={()=>setSessionPnL(0)}>Reset</Button>
          </div>
        </div>
      </Card>

      {/* Historique des actions */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Historique des actions</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="max-h-64 overflow-y-auto space-y-2">
            {actionHistory.length === 0 ? (
              <div className="text-gray-500 text-center py-4">Aucune action enregistrée</div>
            ) : (
              actionHistory.map((entry, index) => (
                <div key={index} className="flex items-center justify-between p-2 bg-primary/5 rounded-lg">
                  <div className="flex items-center gap-3">
                    <span className="text-xs font-medium text-primary bg-primary/10 px-2 py-1 rounded">
                      {index}
                    </span>
                    <span className="text-xs font-medium text-primary bg-primary/10 px-2 py-1 rounded">
                      {entry.phase}
                    </span>
                    <span className="font-medium text-primary">{entry.player}</span>
                    <span className="text-primary/60">→</span>
                    <span className="font-semibold text-primary">{entry.action}</span>
                  </div>
                  <span className="text-xs text-primary/50">
                    {new Date(entry.timestamp).toLocaleTimeString()}
                  </span>
                </div>
              ))
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
