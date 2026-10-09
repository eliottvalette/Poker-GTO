"use client";
import { ChevronDown } from "lucide-react";
import React, { useCallback, useEffect, useRef, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { cardLabel, BrowserTable, type TableView } from "@/lib/game";
import { PolicyCoverageError, type LoadedAveragePolicy } from "@/lib/onnx-policy";
import { ACTION_IDS } from "@/lib/poker/actions";
import { PROFILES, behaviorProbabilities } from "@/lib/poker/behavior";
import { observe, NUMERIC_NAMES, type Observation } from "@/lib/poker/observation";
import PokerTableFrame from "@/components/PokerTableFrame";
import { actionColor } from "@/lib/poker/presentation";
import styles from "./TestTable.module.css";
import HybridAnalysis from "./HybridAnalysis";
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
  const [opponentProfile, setOpponentProfile] = useState("published");
  const [playerCount, setPlayerCount] = useState<2 | 3>(3);
  const seatIds: Seat[] = playerCount === 2 ? [0, 2] : [0, 1, 2];
  const [heroSeat, setHeroSeat] = useState<Seat>(2);
  const [rangeSnapshot, setRangeSnapshot] = useState<RangeSnapshot | null>(null);
  const [analysisSession, setAnalysisSession] = useState(0);
  const [game, setGame] = useState<TableView | null>(null);
  const [busy, setBusy] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [pnlBaseline, setPnlBaseline] = useState(0);
  const locked = useRef(false);
  const started = useRef(false);
  const table = useRef<BrowserTable | null>(null);
  const models = useRef<Record<number, LoadedAveragePolicy>>({});
  models.current = policies;

  async function viewWithPolicy(candidate: BrowserTable): Promise<TableView> {
    const view = candidate.view();
    const count = Object.keys(candidate.tournament.hand?.players ?? {}).length;
    const policy = models.current[count];
    if (!policy) view.policy.reason = `Average policy unavailable for ${count} players; publish this track with migrate.py`;
    if (policy && !view.hand_terminal && view.actor === candidate.hero) {
      try {
        view.policy = { status: "experimental", reason: "Loaded average policy; equilibrium quality has not been certified",
          probabilities: await policy.query(observe(candidate.tournament)) };
      } catch (cause) {
        if (!(cause instanceof PolicyCoverageError)) throw cause;
        view.policy.reason = cause.message;
      }
    }
    return view;
  }

  function opponentPolicy(profile:string) {
    return async (observation:Observation):Promise<number[]> => {
      if(profile!=="published") return behaviorProbabilities(observation,profile);
      // Route by the manifest's observable player-count field, never hidden holdings.
      const players=Math.round(observation.numeric[NUMERIC_NAMES.indexOf("player_count")]*3);
      const model=models.current[players];
      if(!model) throw new Error(`No published opponent policy for ${players} players`);
      const output=await model.query(observation);
      return ACTION_IDS.map(a=>output[a]);
    };
  }

  const newTournament = useCallback(async (hero: Seat, profile = "published", count: 2 | 3 = 3) => {
    if (locked.current) return;
    locked.current = true;
    setBusy(true);
    setError(null);
    try {
      if (profile === "published" && (!models.current[2] || (count === 3 && !models.current[3]))) {
        throw new Error("Trained policy unavailable: export both HU and 3-max with migrate.py");
      }
      const candidate = await BrowserTable.createWithPolicy(Date.now(), hero, profile, opponentPolicy(profile), count);
      const next = await viewWithPolicy(candidate);
      table.current = candidate;
      started.current = true;
      setGame(next);
      setAnalysisSession(value => value + 1);
      setHeroSeat(hero);
      setOpponentProfile(profile);
      setPlayerCount(count);
      setPnlBaseline(0);
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
      if (operation === "next") await candidate.nextHandWithPolicy(opponentPolicy(candidate.opponentProfile));
      else {
        if (!action) throw new Error("Action ID is required");
        await candidate.actWithPolicy(action,opponentPolicy(candidate.opponentProfile));
      }
      const next = await viewWithPolicy(candidate);
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
        const next = await viewWithPolicy(table.current);
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

  const sessionPnL = (game?.hero_result_bb ?? 0) - pnlBaseline;
  const actionHistory = game?.history ?? [];
  const policy = game?.policy;

  const visibleActions: TableView["legal_actions"] = game?.legal_actions ?? [
    { action_id: "FOLD", category: "FOLD", amount_to: null },
    { action_id: "CHECK", category: "CHECK", amount_to: null },
    { action_id: "CALL", category: "CALL", amount_to: null },
    { action_id: "RAISE_2.0X", category: "RAISE", amount_to: null },
    { action_id: "ALL_IN", category: "RAISE", amount_to: null },
  ];

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between gap-3">
      <label className="flex items-center gap-2 text-sm">Opponents
        <span className="relative inline-flex items-center">
        <select aria-label="Opponent strategy" disabled={busy} value={opponentProfile} className="appearance-none rounded border bg-card py-2 pl-3 pr-9"
          onChange={event=>void newTournament(heroSeat,event.target.value,playerCount)}>
          <option value="conservative">Computed conservative</option>
          {Object.keys(PROFILES).filter(p=>p!=="conservative").map(p=><option key={p} value={p}>{p.replaceAll("_"," ")}</option>)}
          <option value="published" disabled={!policies[2]||(playerCount===3&&!policies[3])}>Trained policy{!policies[2]||(playerCount===3&&!policies[3])?" — export HU + 3-max first":""}</option>
          <option value="uniform">Random baseline</option>
        </select>
        <ChevronDown aria-hidden="true" className="pointer-events-none absolute right-3 h-4 w-4 text-muted-foreground" />
        </span>
      </label>
      <span className="relative inline-flex shrink-0 items-center">
        <select aria-label="Table format" disabled={busy} value={playerCount} className="appearance-none rounded border bg-card py-2 pl-3 pr-9 text-sm"
          onChange={event => { const count = Number(event.target.value) as 2 | 3; void newTournament(count === 2 && heroSeat === 1 ? 2 : heroSeat, opponentProfile, count); }}>
          <option value={3}>3-Max</option><option value={2}>HU</option>
        </select>
        <ChevronDown aria-hidden="true" className="pointer-events-none absolute right-3 h-4 w-4 text-muted-foreground" />
      </span>
      </div>
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
              label: player ? player.active ? player.position ?? `P${id}` : "OUT" : `P${id}`,
              stack: player ? `${player.stack_bb.toFixed(1)} BB` : "N/A BB",
              streetBet: player && !game?.hand_terminal ? player.bet_bb : null,
              smallBlind: player?.position === "SB",
              bigBlind: player?.position === "BB",
              active: player ? player.active && !player.folded : true,
              cards: player ? player.cards.length ? player.cards.map(cardLabel) : player.active ? ["XX", "XX"] : [] : ["XX", "XX"],
              netStackChange: game?.hand_terminal ? game.hand_results_bb[id] : undefined,
            };
          })}
          potLabel={game ? `${game.pot_bb.toFixed(2)} BB` : "N/A BB"}
          heroSeat={heroSeat}
          board={game?.board.map(cardLabel).join(" ") ?? ""}
        />
        {game && !game.hand_terminal && game.actor === heroSeat && !policy?.probabilities && <div className="px-4 py-2 text-sm text-muted-foreground">{policy?.reason}</div>}
        {error && <div role="alert" className="px-4 py-2 text-sm text-destructive">{error}</div>}
        <div className={styles.toolbar} aria-label="Table controls">
          <div className="flex items-center gap-2">
            <span className="text-sm text-muted-foreground">Hero</span>
            {seatIds.map(seat => (
              <Button key={seat} disabled={busy || !game} aria-pressed={heroSeat === seat}
                variant={heroSeat === seat ? "default" : "secondary"} className="h-10 w-10 p-0"
                onClick={() => void newTournament(seat,opponentProfile,playerCount)}>P{seat}</Button>
            ))}
          </div>
          <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
            <span>Hand {game?.hand_number ?? "N/A"}</span>
            {game?.tournament_terminal && <span className="font-semibold text-primary">🏆 P{game.winner}</span>}
            {game && !game.players.find(p => p.player_id === heroSeat)?.active && <span>Eliminated</span>}

          </div>
          <div className={styles.management}>
            <Button disabled={busy} variant="secondary" onClick={() => void newTournament(heroSeat,opponentProfile,playerCount)} className="h-10 w-full px-2 text-sm">New game</Button>
            <Button disabled={busy || !game?.hand_terminal || game.tournament_terminal}
              className="h-10 w-full px-2 text-sm" onClick={() => void command("next")}>Next hand</Button>
          </div>
        </div>
        <div className="px-3 pb-4 sm:px-4" aria-label="Poker actions">
          <div className={styles.actions}>
            {visibleActions.map(action => {
              const probability = policy?.probabilities?.[action.action_id];
              return (
                <Button key={action.action_id} disabled={busy || !game || game.hand_terminal || game.actor !== heroSeat}
                  onClick={() => void command("action", action.action_id)}
                  className="relative h-10 w-full min-w-0 flex-col gap-0 overflow-hidden border border-border bg-white px-2 py-1 text-xs text-black hover:bg-white/90">
                  <span className="relative z-10">{actionLabel(action)}</span>
                  {probability !== undefined && <span className="relative z-10 text-xs">{(probability * 100).toFixed(0)}%</span>}
                  {probability !== undefined && <span className="absolute inset-y-0 left-0 bg-black/10" style={{ width: `${probability * 100}%`, background: actionColor(action.action_id), opacity: .25 }} />}
                </Button>
              );
            })}
          </div>
        </div>
        <div className="grid border-t border-border lg:grid-cols-[minmax(0,1fr)_200px]">
          <div className="min-w-0 bg-card/40 p-4 lg:border-r lg:border-border">
            <div className="mb-2 text-center font-semibold">Showdown results</div>
            <div className="grid grid-cols-1 gap-2 sm:grid-cols-3">
              {seatIds.map(id => {
                const player = game?.players.find(p => p.player_id === id);
                const delta = game?.hand_terminal ? game.hand_results_bb[id] : null;
                const win = delta !== null && delta > 0;
                const even = delta === null || delta === 0;
                return (
                  <div key={id} className="rounded-lg bg-background/40 p-3 ring-1 ring-border">
                    <div className="mb-1 flex items-center justify-between">
                      <span className="inline-flex items-center gap-2 text-sm font-medium">
                        <span className="grid h-6 w-6 place-items-center rounded-full bg-muted text-[0.7rem]">P{id}</span>
                        {win && <span className="text-xs">🏆</span>}
                        {player && !player.active && <span className="text-xs">Out</span>}
                      </span>
                      <span className={win ? "rounded-md bg-emerald-500/15 px-2 py-0.5 text-xs font-semibold text-emerald-400" : even ? "rounded-md bg-muted px-2 py-0.5 text-xs text-muted-foreground" : "rounded-md bg-rose-500/15 px-2 py-0.5 text-xs font-semibold text-rose-400"}>
                        {delta === null ? "N/A" : `${win ? "+" : ""}${delta.toFixed(2)} BB`}
                      </span>
                    </div>
                    <div className="flex items-center justify-between text-xs text-muted-foreground">
                      <span>Stack</span>
                      <span className="font-mono">{player ? player.stack_bb.toFixed(2) : "N/A"} BB</span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
          <div className="flex flex-col items-center justify-center gap-2 border-t border-border p-4 lg:border-t-0">
            <h2 className="font-medium">Total P&amp;L (Hero)</h2>
            <div className={sessionPnL > 0 ? "font-semibold text-emerald-400" : sessionPnL < 0 ? "font-semibold text-rose-400" : "font-semibold text-muted-foreground"}>
              {game ? `${sessionPnL >= 0 ? "+" : ""}${sessionPnL.toFixed(2)}` : "N/A"} BB
            </div>
            <Button disabled={busy || !game} className="h-10" variant="secondary" onClick={() => game && setPnlBaseline(game.hero_result_bb)}>Reset</Button>
          </div>
        </div>
      </Card>
      {table.current?.tournament.hand?.players[heroSeat] &&
        <HybridAnalysis rangesOnly hand={table.current.tournament.hand} hero={table.current.tournament.hand.players[heroSeat].cards}
          observerSeat={heroSeat} opponentProfile={opponentProfile} onRangeUpdate={setRangeSnapshot}
          baseline={policy?.probabilities ?? null} sessionId={analysisSession} reconstructHistory transitioning={busy}
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
