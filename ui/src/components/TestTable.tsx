"use client";
import React, { useCallback, useEffect, useRef, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { cardLabel, BrowserTable, type TableView } from "@/lib/game";
import { loadAveragePolicy, PolicyCoverageError, type LoadedAveragePolicy } from "@/lib/onnx-policy";
import { observe } from "@/lib/poker/observation";
import PokerTableFrame from "@/components/PokerTableFrame";
import styles from "./TestTable.module.css";

type Seat = 0 | 1 | 2;

function actionLabel(action: TableView["legal_actions"][number]): string {
  if (action.category === "RAISE" && action.amount_to !== null) {
    return `${action.action_id === "ALL_IN" ? "ALL IN" : "RAISE"} ${action.amount_to?.toFixed(2)} BB`;
  }
  return action.action_id.replaceAll("_", " ");
}

export default function TestTable() {
  const [heroSeat, setHeroSeat] = useState<Seat>(2);
  const [game, setGame] = useState<TableView | null>(null);
  const [busy, setBusy] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [pnlBaseline, setPnlBaseline] = useState(0);
  const locked = useRef(false);
  const started = useRef(false);
  const table = useRef<BrowserTable | null>(null);
  const models = useRef<Record<number, LoadedAveragePolicy>>({});
  const modelFiles = useRef<HTMLInputElement | null>(null);

  async function viewWithPolicy(candidate: BrowserTable, selected?: LoadedAveragePolicy): Promise<TableView> {
    const view = candidate.view();
    const count = Object.keys(candidate.tournament.hand?.players ?? {}).length;
    const policy = selected?.manifest.supported_player_counts.includes(count) ? selected : models.current[count];
    if (!policy) view.policy.reason = `Average policy unavailable for ${count} players; load its model and manifest`;
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

  const newTournament = useCallback(async (hero: Seat) => {
    if (locked.current) return;
    locked.current = true;
    setBusy(true);
    setError(null);
    try {
      const candidate = BrowserTable.create(Date.now(), hero);
      const next = await viewWithPolicy(candidate);
      table.current = candidate;
      setGame(next);
      setHeroSeat(hero);
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
    started.current = true;
    void newTournament(2);
  }, [newTournament]);

  async function command(operation: "action" | "next", action?: string) {
    if (!table.current || locked.current) return;
    locked.current = true;
    setBusy(true);
    setError(null);
    try {
      const candidate = table.current.clone();
      if (operation === "next") candidate.nextHand();
      else {
        if (!action) throw new Error("Action ID is required");
        candidate.act(action);
      }
      const next = await viewWithPolicy(candidate);
      table.current = candidate;
      setGame(next);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      locked.current = false;
      setBusy(false);
    }
  }

  async function importPolicy(files: FileList | null) {
    if (!files?.length || locked.current) return;
    locked.current = true;
    setBusy(true);
    setError(null);
    let loaded: LoadedAveragePolicy | null = null;
    try {
      const selected = Array.from(files);
      const weights = selected.filter(file => file.name.endsWith(".onnx"));
      const manifests = selected.filter(file => file.name.endsWith(".json"));
      if (selected.length !== 2 || weights.length !== 1 || manifests.length !== 1) {
        throw new Error("Select exactly one .onnx average policy and its .json manifest together");
      }
      loaded = await loadAveragePolicy(await weights[0].arrayBuffer(), JSON.parse(await manifests[0].text()));
      const next = table.current ? await viewWithPolicy(table.current, loaded) : null;
      const previous = new Set(loaded.manifest.supported_player_counts.map(count => models.current[count]).filter(Boolean));
      for (const count of loaded.manifest.supported_player_counts) models.current[count] = loaded;
      loaded = null;
      if (next) setGame(next);
      for (const policy of previous) {
        if (!Object.values(models.current).includes(policy)) await policy.release();
      }
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      if (loaded) await loaded.release();
      if (modelFiles.current) modelFiles.current.value = "";
      locked.current = false;
      setBusy(false);
    }
  }

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
      <Card>
        <PokerTableFrame
          phase={game?.hand_terminal ? "SHOWDOWN" : game?.street}
          seats={([0, 1, 2] as const).map(id => {
            const player = game?.players.find(p => p.player_id === id);
            return {
              id,
              label: player ? player.active ? player.position ?? `P${id}` : "OUT" : `P${id}`,
              stack: player ? `${player.stack_bb.toFixed(1)} BB` : "— BB",
              streetBet: player && !game?.hand_terminal ? player.bet_bb : null,
              smallBlind: player?.position === "SB",
              bigBlind: player?.position === "BB",
              active: player ? player.active && !player.folded : true,
              cards: player ? player.cards.length ? player.cards.map(cardLabel) : player.active ? ["XX", "XX"] : [] : ["XX", "XX"],
              netStackChange: game?.hand_terminal ? game.hand_results_bb[id] : undefined,
            };
          })}
          potLabel={game ? `${game.pot_bb.toFixed(2)} BB` : "— BB"}
          heroSeat={heroSeat}
          board={game?.board.map(cardLabel).join(" ") ?? ""}
        />
        {error && <div role="alert" className="px-4 py-2 text-sm text-destructive">{error}</div>}
        <div className={styles.toolbar} aria-label="Table controls">
          <div className="flex items-center gap-2">
            <span className="text-sm text-muted-foreground">Hero</span>
            {([0, 1, 2] as const).map(seat => (
              <Button key={seat} disabled={busy || !game} aria-pressed={heroSeat === seat}
                variant={heroSeat === seat ? "default" : "secondary"} className="h-10 w-10 p-0"
                onClick={() => void newTournament(seat)}>P{seat}</Button>
            ))}
          </div>
          <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
            <span>Hand {game?.hand_number ?? "—"}</span>
            {game?.tournament_terminal && <span className="font-semibold text-primary">🏆 P{game.winner}</span>}
            {game && !game.players.find(p => p.player_id === heroSeat)?.active && <span>Eliminated</span>}
            <input ref={modelFiles} type="file" accept=".onnx,.json" multiple hidden
              aria-label="Average policy files" onChange={event => void importPolicy(event.target.files)} />
            <button type="button" disabled={busy} className="rounded border border-border px-2 py-1 disabled:opacity-50"
              title={`${policy?.reason ?? ""}. Load an average policy (.onnx + .json)`}
              onClick={() => modelFiles.current?.click()}>
              {!game ? "Policy —" : policy?.status === "experimental" ? "Experimental policy" : "Policy unavailable"}
            </button>
          </div>
          <div className={styles.management}>
            <Button disabled={busy} variant="secondary" onClick={() => void newTournament(heroSeat)} className="h-10 w-full px-2 text-sm">New game</Button>
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
                  {probability !== undefined && <span className="relative z-10 text-[10px]">{(probability * 100).toFixed(0)}%</span>}
                  {probability !== undefined && <span className="absolute inset-y-0 left-0 bg-black/10" style={{ width: `${probability * 100}%` }} />}
                </Button>
              );
            })}
          </div>
        </div>
        <div className="grid border-t border-border lg:grid-cols-[minmax(0,1fr)_200px]">
          <div className="min-w-0 bg-card/40 p-4 lg:border-r lg:border-border">
            <div className="mb-2 text-center font-semibold">Showdown — Results</div>
            <div className="grid grid-cols-1 gap-2 sm:grid-cols-3">
              {([0, 1, 2] as const).map(id => {
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
                        {delta === null ? "—" : `${win ? "+" : ""}${delta.toFixed(2)} BB`}
                      </span>
                    </div>
                    <div className="flex items-center justify-between text-xs text-muted-foreground">
                      <span>Stack</span>
                      <span className="font-mono">{player ? player.stack_bb.toFixed(2) : "—"} BB</span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
          <div className="flex flex-col items-center justify-center gap-2 border-t border-border p-4 lg:border-t-0">
            <h2 className="font-medium">Total P&amp;L (Hero)</h2>
            <div className={sessionPnL > 0 ? "font-semibold text-emerald-400" : sessionPnL < 0 ? "font-semibold text-rose-400" : "font-semibold text-muted-foreground"}>
              {game ? `${sessionPnL >= 0 ? "+" : ""}${sessionPnL.toFixed(2)}` : "—"} BB
            </div>
            <Button disabled={busy || !game} className="h-10" variant="secondary" onClick={() => game && setPnlBaseline(game.hero_result_bb)}>Reset</Button>
          </div>
        </div>
      </Card>
      <Card>
        <CardHeader><CardTitle className="text-lg">Action history</CardTitle></CardHeader>
        <CardContent>
          <div className="max-h-64 overflow-y-auto space-y-2">
            {actionHistory.length === 0 ? (
              <div className="text-gray-500 text-center py-4">—</div>
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
