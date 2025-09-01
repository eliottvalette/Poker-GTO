// ui/src/components/TestTable.tsx
"use client";
import React, { useEffect, useState, useCallback } from "react";
import { Card as UICard, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { PokerGame, buildInfosetKeyFast, type Action } from "@/lib/game";
import { normalize, type Policy } from "@/lib/policy";

type Seat = 0|1|2;

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
  const [, force] = useState(0); // re-render

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
      // continue tant que ce n'est pas au héros
    }
  }, [policy, heroSeat]);

  function newHand() {
    const g = makeNewGame();
    setGame(g);
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
    stepBotsForward(game);
    force(n=>n+1);
  }

  const boardStr = game.community.map(c=>c.toString()).join(" ");
  const seatLabel = (s:Seat)=> s===0?"SB":s===1?"BB":"BTN";

  return (
    <UICard>
      <CardHeader className="pb-2">
        <CardTitle>Test</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex gap-2">
          <Button variant={heroSeat===0?"default":"outline"} onClick={()=>setHeroSeat(0)}>Héro: SB</Button>
          <Button variant={heroSeat===1?"default":"outline"} onClick={()=>setHeroSeat(1)}>Héro: BB</Button>
          <Button variant={heroSeat===2?"default":"outline"} onClick={()=>setHeroSeat(2)}>Héro: BTN</Button>
          <Button onClick={newHand} className="ml-auto">Nouvelle main</Button>
        </div>

        <div className="text-sm">
          <div>Phase: <b>{game.current_phase}</b> | Pot: <b>{game.main_pot.toFixed(2)} BB</b> | Au tour de: <b>Player_{game.current_role} ({seatLabel(game.current_role as Seat)})</b></div>
          <div>Board: [{boardStr||"-"}]</div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          {game.players.map((p)=>(
            <div key={p.name} className={`p-3 rounded border ${p.role===heroSeat?"border-primary":"border-border"} bg-muted/30`}>
              <div className="text-sm font-medium">{p.name} — {seatLabel(p.role as Seat)}</div>
              <div className="text-xs">Stack: {p.stack.toFixed(2)} | Bet: {p.current_player_bet.toFixed(2)} | {p.has_folded?"FOLD":p.is_all_in?"ALL-IN":""}</div>
              <div className="text-sm mt-1">Cartes: {p.cards.length? `${p.cards[0].toString()} ${p.cards[1].toString()}` : "-"}</div>
            </div>
          ))}
        </div>

        {game.current_phase!=="SHOWDOWN" && game.current_role===heroSeat && (
          <div className="flex flex-wrap gap-2">
            {game.update_available_actions(hero).map(a=>(
              <Button key={a} onClick={()=>onHeroAction(a)}>{a}</Button>
            ))}
          </div>
        )}

        {game.current_phase==="SHOWDOWN" && (
          <div className="text-sm">
            <div className="font-medium">Showdown</div>
            <div>Stacks finaux: {game.players.map(p=>`${p.name} ${p.stack.toFixed(2)}BB`).join(" | ")}</div>
          </div>
        )}
      </CardContent>
    </UICard>
  );
}
