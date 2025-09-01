// ui/src/components/TestTable.tsx
"use client";
import React, { useEffect, useState, useCallback } from "react";
import { Card as UICard, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { PokerGame, buildInfosetKeyFast, type Action } from "@/lib/game";
import { normalize, type Policy } from "@/lib/policy";
import PokerTableFrame from "@/components/PokerTableFrame";

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
        <PokerTableFrame
          seats={[
            { id:0, label:"SB", stack:`${game.players[0].stack.toFixed(1)} BB`, smallBlind:true, active:!game.players[0].has_folded },
            { id:1, label:"BB", stack:`${game.players[1].stack.toFixed(1)} BB`, bigBlind:true,  active:!game.players[1].has_folded },
            { id:2, label:"BTN", stack:`${game.players[2].stack.toFixed(1)} BB`, active:!game.players[2].has_folded },
          ]}
          potLabel={`${game.main_pot.toFixed(2)} BB`}
          heroSeat={heroSeat}
          board={boardStr}
        >
          {/* barre d'actions */}
          <div className="flex flex-wrap items-stretch gap-2">
            <div className="flex gap-2">
              <Button variant={heroSeat===0?"default":"secondary"} onClick={()=>setHeroSeat(0)}>Héro: SB</Button>
              <Button variant={heroSeat===1?"default":"secondary"} onClick={()=>setHeroSeat(1)}>Héro: BB</Button>
              <Button variant={heroSeat===2?"default":"secondary"} onClick={()=>setHeroSeat(2)}>Héro: BTN</Button>
              <Button onClick={newHand} className="ml-2">Nouvelle main</Button>
            </div>

            {game.current_phase!=="SHOWDOWN" && game.current_role===heroSeat && (
              <div className="ml-auto flex gap-2">
                {game.update_available_actions(hero).map(a=>(
                  <button
                    key={a}
                    onClick={()=>onHeroAction(a)}
                    className={[
                      // bouton glossy style Winamax
                      "px-4 py-2 rounded-xl font-semibold text-sm tracking-wide",
                      "bg-gradient-to-b from-[#263b50] to-[#172636] text-white",
                      "ring-1 ring-white/15 shadow-[inset_0_1px_0_0_rgba(255,255,255,.08),0_6px_20px_-4px_rgba(0,0,0,.6)]",
                      "hover:from-[#2c4863] hover:to-[#193247] active:scale-[.98] transition",
                    ].join(" ")}
                  >
                    {a}
                  </button>
                ))}
              </div>
            )}
          </div>

          {game.current_phase==="SHOWDOWN" && (
            <div className="mt-3 text-center text-cyan-100">
              <div className="font-semibold">Showdown</div>
              <div className="text-sm opacity-90">
                {game.players.map(p=>`${p.name} ${p.stack.toFixed(2)}BB`).join(" | ")}
              </div>
            </div>
          )}
        </PokerTableFrame>
      </CardContent>
    </UICard>
  );
}
