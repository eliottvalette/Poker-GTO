"use client";
import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Button } from "./ui/button";
import { type LoadedAveragePolicy } from "@/lib/onnx-policy";
import { analysisRoot, holeObservation, queryRange, withHeroCards, type RangeCell } from "@/lib/poker/policy-analysis";
import { HandState, STREETS, type Position, type Street } from "@/lib/poker/engine";
import { applyAction, legalActions } from "@/lib/poker/actions";
import { cardLabel } from "@/lib/game";

const COLORS = ["#64748b", "#22c55e", "#16a34a", "#fbbf24", "#f59e0b", "#f97316", "#ea580c", "#fb7185", "#f43f5e", "#e11d48", "#be123c", "#9f1239", "#7c3aed"];
import { ACTION_IDS } from "@/lib/poker/actions";
const color = (action: string) => COLORS[ACTION_IDS.indexOf(action as typeof ACTION_IDS[number])];

export default function PolicyAnalysis({ mode, policies }: {
  mode: "overview" | "case"; policies: Record<number, LoadedAveragePolicy>;
}) {
  const [count, setCount] = useState(3);
  const [position, setPosition] = useState<Position>("BTN");
  const [street, setStreet] = useState<Street>("PREFLOP");
  const [stacks, setStacks] = useState([25, 25, 25]);
  const [big, setBig] = useState(1);
  const [seed, setSeed] = useState(42);
  const [hand, setHand] = useState<HandState>(() => analysisRoot(3, "BTN", "PREFLOP", [25, 25, 25], 1, 42));
  const [hero, setHero] = useState<[number, number]>([48, 49]);
  const [boardText, setBoardText] = useState("");
  const [cells, setCells] = useState<RangeCell[] | null>(null);
  const [probabilities, setProbabilities] = useState<Record<string, number> | null>(null);
  const [selected, setSelected] = useState<RangeCell | null>(null);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const policy = policies[Object.keys(hand.players).length];

  function build() {
    try {
      const next = analysisRoot(count, position, street, stacks, big, seed);
      const labels = boardText.trim().split(/\s+/).filter(Boolean);
      if (labels.length) {
        const cards = labels.map(label => {
          const token = /^(10|[2-9TJQKA])([shdc])$/i.exec(label);
          if (!token) throw new Error(`Invalid board card ${label}; use As Kh 10d`);
          const rank = "23456789TJQKA".indexOf(token[1].toUpperCase() === "10" ? "T" : token[1].toUpperCase());
          const suit = "shdc".indexOf(token[2].toLowerCase());
          return rank * 4 + suit;
        });
        if (cards.length !== next.board.length || new Set(cards).size !== cards.length) throw new Error(`Expected ${next.board.length} distinct board cards for ${street}`);
        // Remap the entire deal so board edits preserve card uniqueness for continuations.
        cards.forEach((card, index) => {
          const previous = next.board[index];
          const swap = (value: number) => value === previous ? card : value === card ? previous : value;
          next.board = next.board.map(swap);
          next.deck = next.deck.map(swap);
          for (const player of Object.values(next.players)) player.cards = player.cards.map(swap) as [number, number];
        });
      }
      next.assertInvariants();
      setHand(next); setError(null); setSelected(null);
    } catch (cause) { setError(cause instanceof Error ? cause.message : String(cause)); }
  }

  useEffect(() => {
    let cancelled = false;
    setCells(null); setProbabilities(null); setProgress(0); setSelected(null); setError(null);
    if (!policy || hand.terminal) return;
    const request = Promise.resolve().then(() => mode === "overview"
      ? queryRange(hand, policy, () => cancelled, done => { if (!cancelled) setProgress(done); }).then(result => { if (!cancelled) setCells(result); })
      : policy.query(holeObservation(hand, hero)).then(result => { if (!cancelled) setProbabilities(result); }));
    void request.catch(cause => { if (!cancelled) setError(cause instanceof Error ? cause.message : String(cause)); });
    return () => { cancelled = true; };
  }, [mode, hand, policy, hero]);

  function act(action: string) {
    try {
      const next = mode === "case" ? withHeroCards(hand, hero) : hand.clone();
      applyAction(next, action); setHand(next); setSelected(null);
    } catch (cause) { setError(cause instanceof Error ? cause.message : String(cause)); }
  }
  const actions = legalActions(hand);
  const mix = mode === "overview" ? selected?.probabilities : probabilities;
  return <Card>
    <CardHeader><CardTitle>{mode === "overview" ? "Action range · 169 hands" : "Exact hand policy"}</CardTitle></CardHeader>
    <CardContent className="space-y-4">
      <div className="flex flex-wrap items-end gap-3">
        <label>Players<select className="block rounded border p-2" value={count} onChange={event => {
          const n = Number(event.target.value); setCount(n); if (n === 2 && position === "BTN") setPosition("SB");
        }}><option value={3}>3-max</option><option value={2}>HU</option></select></label>
        <label>Position<select className="block rounded border p-2" value={position} onChange={event => setPosition(event.target.value as Position)}>
          {(count === 3 ? ["BTN", "SB", "BB"] : ["SB", "BB"]).map(p => <option key={p}>{p}</option>)}
        </select></label>
        <label>Street<select className="block rounded border p-2" value={street} onChange={event => setStreet(event.target.value as Street)}>
          {STREETS.map(s => <option key={s}>{s}</option>)}
        </select></label>
        {stacks.slice(0, count).map((stack, i) => <label key={i}>P{i} chips<input className="block w-24 rounded border p-2" type="number" min="0.01" step="0.5" value={stack}
          onChange={event => setStacks(stacks.map((value, index) => index === i ? Number(event.target.value) : value))} /></label>)}
        <label>BB chips<input className="block w-24 rounded border p-2" type="number" min="0.01" step="0.5" value={big} onChange={event => setBig(Number(event.target.value))} /></label>
        <label>Deal seed<input className="block w-24 rounded border p-2" type="number" value={seed} onChange={event => setSeed(Number(event.target.value))} /></label>
        <label>Board (optional)<input className="block w-48 rounded border p-2" placeholder="As Kh 2d" value={boardText} onChange={event => setBoardText(event.target.value)} /></label>
        <Button onClick={build}>Apply state</Button>
      </div>
      <div className="text-sm text-muted-foreground">
        {hand.terminal ? "Hand settled" : `${Object.keys(hand.players).length === 2 ? "HU" : "3-max"} · ${hand.street} · ${hand.actor.position}`}
        {` · Pot ${(hand.pot / hand.blinds.big).toFixed(2)} BB · Board ${hand.board.map(cardLabel).join(" ") || "—"}`}
        {policy && ` · Model iteration ${policy.manifest.iteration}`}
      </div>
      {!policy && <div role="status">Average policy unavailable for {Object.keys(hand.players).length} players; publish this track with migrate.py.</div>}
      {error && <div role="alert" className="text-destructive">{error}</div>}
      {mode === "case" && <div className="flex gap-3">
        {hero.map((card, i) => <label key={i}>Hero card {i + 1}<select className="block rounded border p-2" value={card} onChange={event => setHero(hero.map((value, index) => i === index ? Number(event.target.value) : value) as [number, number])}>
          {Array.from({ length: 52 }, (_, value) => <option key={value} value={value}>{cardLabel(value)}</option>)}
        </select></label>)}
      </div>}
      {mode === "overview" && policy && !hand.terminal && !error && <>
        {!cells && <div role="status">Querying exact combos… {progress}</div>}
        {cells && <>
          <div className="text-sm text-muted-foreground">{cells.reduce((n, cell) => n + cell.combos, 0)} exact combos. Each cell averages its board-compatible combos uniformly. Select a cell for the full action mixture.</div>
          <div className="overflow-auto"><div className="grid min-w-[650px] gap-1" style={{ gridTemplateColumns: "repeat(13, minmax(0, 1fr))" }}>
            {cells.map(cell => <button key={cell.label} onClick={() => setSelected(cell)} className="relative h-14 overflow-hidden rounded border text-xs" title={Object.entries(cell.probabilities).map(([a, p]) => `${a}: ${(p * 100).toFixed(1)}%`).join("\n")}>
              <div className="absolute inset-0 flex">{Object.entries(cell.probabilities).map(([action, p]) => <span key={action} style={{ width: `${p * 100}%`, background: color(action) }} />)}</div>
              <span className="relative rounded bg-black/65 px-1 text-white">{cell.label}</span>
              {!cell.combos && <span className="relative block">—</span>}
            </button>)}
          </div></div>
        </>}
      </>}
      <div className="flex flex-wrap gap-3 text-xs">{actions.map(action => <span key={action.action_id} className="flex items-center gap-1"><span className="h-3 w-3" style={{ background: color(action.action_id) }} />{action.action_id}</span>)}</div>
      {mix && <div className="space-y-2">{selected && <div>{selected.label} · {selected.combos} combos</div>}{actions.map(action => <div key={action.action_id} className="flex items-center gap-3 text-sm">
        <span className="w-44">{action.action_id}{action.amount_to !== null ? ` ${(action.amount_to / hand.blinds.big).toFixed(2)} BB` : ""}</span>
        <div className="h-3 w-64 rounded bg-muted"><div className="h-full rounded" style={{ width: `${(mix[action.action_id] ?? 0) * 100}%`, background: color(action.action_id) }} /></div>
        <span>{((mix[action.action_id] ?? 0) * 100).toFixed(1)}%</span>
      </div>)}</div>}
      <div className="flex flex-wrap gap-2">{actions.map(action => <Button variant="secondary" key={action.action_id} onClick={() => act(action.action_id)}>{action.action_id}{action.amount_to !== null ? ` ${(action.amount_to / hand.blinds.big).toFixed(2)} BB` : ""}</Button>)}</div>
      <details><summary>Public action history ({hand.history.length})</summary><ol className="space-y-1 text-sm">{hand.history.map((event, i) => <li key={i}>{event.street} · {event.position} · {event.action} {(event.amount_to / hand.blinds.big).toFixed(2)} BB · pot {(event.pot_after / hand.blinds.big).toFixed(2)} BB</li>)}</ol></details>
    </CardContent>
  </Card>;
}
