"use client";
import TableFormat from "./TableFormat";
import React, { useCallback, useEffect, useRef, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { cardLabel, BrowserTable, type TableView } from "@/lib/game";
import { type LoadedAveragePolicy } from "@/lib/onnx-policy";
import { ACTION_IDS } from "@/lib/poker/actions";
import { NUMERIC_NAMES, type Observation } from "@/lib/poker/observation";
import PokerTableFrame from "@/components/PokerTableFrame";
import { actionColor } from "@/lib/poker/presentation";
import styles from "./TestTable.module.css";
import HybridAnalysis, { type DecisionSnapshot } from "./HybridAnalysis";
import { showdownHighlights } from "@/lib/poker/showdown-display";
import RangeInspection from "./RangeInspection";
import { rangeStateKey, type RangeSnapshot } from "@/lib/poker/beliefs";

type Seat = 0 | 1 | 2;

function actionLabel(action: TableView["legal_actions"][number]): string {
  if (action.category === "RAISE" && action.amount_to !== null) {
    return `${action.action_id === "ALL_IN" ? "ALL IN" : "RAISE"} ${action.amount_to?.toFixed(2)} BB`;
  }
  return action.action_id.replaceAll("_", " ");
}

export default function TestTable({ policies }: { policies: Record<number, LoadedAveragePolicy> }) {
  const opponentProfile = "published";
  const [playerCount, setPlayerCount] = useState<2 | 3>(3);
  const seatIds: Seat[] = playerCount === 2 ? [0, 2] : [0, 1, 2];
  const [heroSeat, setHeroSeat] = useState<Seat>(2);
  const [rangeSnapshot, setRangeSnapshot] = useState<RangeSnapshot | null>(null);
  const [analysisSession, setAnalysisSession] = useState(0);
  const [decision, setDecision] = useState<DecisionSnapshot | null>(null);
  const [cancelCalculation, setCancelCalculation] = useState(0);
  const [game, setGame] = useState<TableView | null>(null);
  const [busy, setBusy] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [carriedPnL, setCarriedPnL] = useState(0);
  const locked = useRef(false);
  const started = useRef(false);
  const table = useRef<BrowserTable | null>(null);
  const models = useRef<Record<number, LoadedAveragePolicy>>({});
  models.current = policies;


  function opponentPolicy() {
    return async (observation:Observation):Promise<number[]> => {
      // Route by the manifest's observable player-count field, never hidden holdings.
      const players=Math.round(observation.numeric[NUMERIC_NAMES.indexOf("player_count")]*3);
      const model=models.current[players];
      if(!model) throw new Error(`No published opponent policy for ${players} players`);
      const output=await model.query(observation);
      return ACTION_IDS.map(a=>output[a]);
    };
  }

  const newTournament = useCallback(async (hero: Seat, count: 2 | 3 = 3, preservePnL = false) => {
    if (locked.current) return;
    locked.current = true;
    setBusy(true);
    setError(null);
    try {
      if (!models.current[2] || (count === 3 && !models.current[3])) {
        throw new Error("Trained policy unavailable: export both HU and 3-max with migrate.py");
      }
      const tableResultBefore = table.current?.view().hero_result_bb;
      const candidate = await BrowserTable.createWithPolicy(Date.now(), hero, "published", opponentPolicy(), count);
      const next = candidate.view();
      table.current = candidate;
      started.current = true;
      setGame(next);
      setAnalysisSession(value => value + 1);
      setHeroSeat(hero);
      setPlayerCount(count);
      setCarriedPnL(previous => preservePnL ? previous + (tableResultBefore ?? 0) : 0);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      locked.current = false;
      setBusy(false);
    }
  }, []);

  useEffect(() => {
    if (started.current) return;
    if (!policies[2] || !policies[3]) {
      setBusy(false);
      setError("Trained policy unavailable: export both HU and 3-max with migrate.py");
      return;
    }
    void newTournament(2);
  }, [newTournament, policies]);

  async function command(operation: "action" | "next", action?: string) {
    if (!table.current || locked.current) return;
    locked.current = true;
    setBusy(true);
    setError(null);
    try {
      const candidate = table.current.clone();
      if (operation === "next") await candidate.nextHandWithPolicy(opponentPolicy());
      else {
        if (!action) throw new Error("Action ID is required");
        await candidate.actWithPolicy(action,opponentPolicy());
      }
      const next = candidate.view();
      if (operation === "next") setAnalysisSession(value => value + 1);
      table.current = candidate;
      setGame(next);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      locked.current = false;
      setBusy(false);
    }
  }

  useEffect(() => {
    let cancelled = false;
    async function refreshView() {
      while (locked.current && !cancelled) await new Promise(resolve => setTimeout(resolve, 100));
      if (cancelled || !table.current) return;
      locked.current = true;
      setBusy(true);
      try {
        const next = table.current.view();
        if (!cancelled) setGame(next);
      } catch (cause) {
        if (!cancelled) setError(cause instanceof Error ? cause.message : String(cause));
      } finally {
        locked.current = false;
        if (!cancelled) setBusy(false);
      }
    }
    void refreshView();
    return () => { cancelled = true; };
  }, [policies]);

  const showdown = table.current?.tournament.hand ? showdownHighlights(table.current.tournament.hand) : {};
  const sessionPnL = (game?.hero_result_bb ?? 0) + carriedPnL;
  const actionHistory = game?.history ?? [];
  const currentDecision = !busy && table.current?.tournament.hand && decision?.sessionId === analysisSession
    && decision.stateKey === rangeStateKey(table.current.tournament.hand, heroSeat) ? decision : null;
  const calculation = currentDecision?.status === "current" ? currentDecision.result : undefined;

  const visibleActions: TableView["legal_actions"] = game?.legal_actions ?? [
    { action_id: "FOLD", category: "FOLD", amount_to: null },
    { action_id: "CHECK", category: "CHECK", amount_to: null },
    { action_id: "CALL", category: "CALL", amount_to: null },
    { action_id: "RAISE_2.0X", category: "RAISE", amount_to: null },
    { action_id: "ALL_IN", category: "RAISE", amount_to: null },
  ];

  return (
    <div className="space-y-4">
      <Card>
        <PokerTableFrame
          phase={game?.hand_terminal ? "SHOWDOWN" : game?.street}
          seats={seatIds.map(id => {
            const player = game?.players.find(p => p.player_id === id);
            return {
              id,
              inspection: id !== heroSeat && table.current?.tournament.hand?.players[id] ? <RangeInspection seat={id} hand={table.current.tournament.hand} snapshot={
                !busy && rangeSnapshot?.sessionId === analysisSession && rangeSnapshot.stateKey === rangeStateKey(table.current.tournament.hand,heroSeat) ? rangeSnapshot : null
              }/> : undefined,
              label: player ? player.position ?? (player.active ? `P${id}` : "OUT") : `P${id}`,
              stack: player ? `${player.stack_bb.toFixed(1)} BB` : "N/A BB",
              streetBet: player && !game?.hand_terminal ? player.bet_bb : null,
              smallBlind: player?.position === "SB",
              bigBlind: player?.position === "BB",
              active: player ? player.active && !player.folded : true,
              folded: player?.folded,
              cards: player ? player.cards.length ? player.cards.map(cardLabel) : player.active || player.position !== null ? ["XX", "XX"] : [] : ["XX", "XX"],
              revealedCards: game?.hand_terminal && id !== heroSeat
                ? table.current?.tournament.hand?.players[id]?.cards.map(cardLabel) : undefined,
              showdown: showdown[id] ? { ...showdown[id], best: showdown[id].best.map(cardLabel), decisive: showdown[id].decisive.map(cardLabel) } : undefined,
              netStackChange: game?.hand_terminal ? game.hand_results_bb[id] : undefined,
            };
          })}
          potLabel={game ? `${game.pot_bb.toFixed(2)} BB` : "N/A BB"}
          heroSeat={heroSeat}
          board={game?.board.map(cardLabel).join(" ") ?? ""}
        />
        {game && !game.hand_terminal && game.actor === heroSeat && <div className="px-4 py-2 text-xs text-muted-foreground" role="status">
          {currentDecision?.status === "error" ? currentDecision.error : calculation
            ? `${calculation.iterations ? "Local CFR" : "Monte Carlo · conservative continuations"} · ${calculation.samples} samples`
            : "Calculating action values…"}
          {calculation?.warnings.some(w => /unresolved|insufficient/i.test(w)) && " · Ranking uncertain"}
          {currentDecision?.status === "calculating" && <button className="ml-2 underline" onClick={() => setCancelCalculation(value => value + 1)}>Cancel</button>}
        </div>}
        {error && <div role="alert" className="px-4 py-2 text-sm text-destructive">{error}</div>}
        <div className={styles.toolbar} aria-label="Table controls">
          <TableFormat value={playerCount} disabled={busy} onChange={count => {
            if (playerCount !== count) void newTournament(2, count, true);
          }} />
          <div className="text-sm tabular-nums" aria-label="Total P&L">
            <span className="mr-2 text-muted-foreground">P&amp;L</span>
            <span className={sessionPnL > 0 ? "text-emerald-400" : sessionPnL < 0 ? "text-rose-400" : "text-muted-foreground"}>
              {sessionPnL >= 0 ? "+" : ""}{sessionPnL.toFixed(2)} BB
            </span>
          </div>
          <div className={styles.management}>
            <Button disabled={busy} variant="secondary" onClick={() => void newTournament(heroSeat,playerCount)} className="h-10 w-full px-2 text-sm">New game</Button>
            <Button disabled={busy || !game?.hand_terminal}
              className="h-10 w-full px-2 text-sm" onClick={() => void command("next")}>Next hand</Button>
          </div>
        </div>
        <div className="px-3 pb-4 sm:px-4" aria-label="Poker actions">
          <div className={styles.actions}>
            {visibleActions.map(action => {
              const probability = calculation?.probabilities[action.action_id];
              const ev = calculation?.ev[action.action_id];
              return (
                <Button key={action.action_id} disabled={busy || !game || game.hand_terminal || game.actor !== heroSeat}
                  onClick={() => void command("action", action.action_id)}
                  className={`${styles.actionButton} relative h-10 w-full min-w-0 flex-col gap-0 border border-border bg-white py-1 text-black hover:bg-white/90`}>
                  <span className="relative z-10 whitespace-nowrap">{actionLabel(action)}</span>
                  {probability !== undefined && <span className="relative z-10 text-xs">{(probability * 100).toFixed(0)}%{ev !== undefined && table.current?.tournament.hand ? ` · ${(ev / table.current.tournament.hand.blinds.big).toFixed(2)} BB` : ""}</span>}
                  {probability !== undefined && <span className="absolute inset-y-0 left-0 bg-black/10" style={{ width: `${probability * 100}%`, background: actionColor(action.action_id), opacity: .25 }} />}
                </Button>
              );
            })}
          </div>
        </div>
      </Card>
      {table.current?.tournament.hand?.players[heroSeat] &&
        <HybridAnalysis rangesOnly autoAnalyze onDecisionUpdate={setDecision} cancelToken={cancelCalculation} hand={table.current.tournament.hand} hero={table.current.tournament.hand.players[heroSeat].cards}
          observerSeat={heroSeat} opponentProfile={opponentProfile} onRangeUpdate={setRangeSnapshot}
          baseline={null} sessionId={analysisSession} reconstructHistory transitioning={busy}
          initialProfiles={Object.fromEntries(Object.keys(table.current.tournament.hand.players).map(p => [p, Number(p) === heroSeat ? "conservative" : opponentProfile]))} />
      }
      <Card>
        <CardHeader><CardTitle className="text-lg">Action history</CardTitle></CardHeader>
        <CardContent>
          <div className="max-h-64 overflow-y-auto space-y-2">
            {actionHistory.length === 0 ? (
              <div className="text-gray-500 text-center py-4">N/A</div>
            ) : actionHistory.map((entry, index) => (
              <div key={index} className="flex flex-col p-2 bg-primary/5 rounded-lg">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="text-xs font-medium text-primary bg-primary/10 px-2 py-1 rounded">{index}</span>
                    <span className="text-xs font-medium text-primary bg-primary/10 px-2 py-1 rounded">{entry.street}</span>
                    <span className="font-medium text-primary">P{entry.player_id} · {entry.position}</span>
                    <span className="text-primary/60">→</span>
                    <span className="font-semibold text-primary">{entry.action} {entry.amount_to.toFixed(2)} BB</span>
                  </div>
                  <span className="text-xs text-primary/50">Pot: {entry.pot_after.toFixed(2)} BB</span>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
