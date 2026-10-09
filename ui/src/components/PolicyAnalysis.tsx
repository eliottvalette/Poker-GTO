"use client";
import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Button } from "./ui/button";
import { type LoadedAveragePolicy } from "@/lib/onnx-policy";
import { analysisRoot, holeObservation, withHeroCards } from "@/lib/poker/policy-analysis";
import { HandState, STREETS, type Position, type Street } from "@/lib/poker/engine";
import { applyAction, legalActions } from "@/lib/poker/actions";
import { NativeSelect } from "./ui/native-select";
import { Input } from "./ui/input";
import TableFormat from "./TableFormat";
import HoleCardsPicker from "./HoleCardsPicker";

import { actionColor as color } from "@/lib/poker/presentation";

export default function PolicyAnalysis({ policies }: {
  policies: Record<number, LoadedAveragePolicy>;
}) {
  const [cardsLocked, setCardsLocked] = useState(false);
  const [count, setCount] = useState(3);
  const [position, setPosition] = useState<Position>("BTN");
  const [street, setStreet] = useState<Street>("PREFLOP");
  const [stacks, setStacks] = useState([25, 25, 25]);
  const [hand, setHand] = useState<HandState>(() => withHeroCards(analysisRoot(3, "BTN", "PREFLOP", [25, 25, 25], 1, 42), [48, 49]));
  const [hero, setHero] = useState<[number, number]>([48, 49]);
  const [boardText, setBoardText] = useState("");
  const [probabilities, setProbabilities] = useState<Record<string, number> | null>(null);
  const [error, setError] = useState<string | null>(null);
  const policy = policies[Object.keys(hand.players).length];

  function build() {
    try {
      const next = analysisRoot(count, position, street, stacks, 1, 42);
      const labels = (street === "PREFLOP" ? "" : boardText).trim().split(/\s+/).filter(Boolean);
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
      setHand(withHeroCards(next, hero)); setCardsLocked(false); setError(null);
    } catch (cause) { setError(cause instanceof Error ? cause.message : String(cause)); }
  }

  useEffect(() => {
    let cancelled = false;
    setProbabilities(null); setError(null);
    if (!policy || hand.terminal) return;
    const request = Promise.resolve().then(() => policy.query(holeObservation(hand, hero)))
      .then(result => { if (!cancelled) setProbabilities(result); });
    void request.catch(cause => { if (!cancelled) setError(cause instanceof Error ? cause.message : String(cause)); });
    return () => { cancelled = true; };
  }, [hand, policy, hero]);

  function act(action: string) {
    try {
      const next = withHeroCards(hand, hero);
      applyAction(next, action); setHand(next); setCardsLocked(true); if (!next.terminal) setHero([...next.actor.cards]);
    } catch (cause) { setError(cause instanceof Error ? cause.message : String(cause)); }
  }
  const actions = legalActions(hand);
  return <Card>
    <CardHeader><CardTitle>Specific spot</CardTitle></CardHeader>
    <CardContent className="space-y-6">
      <div className="space-y-4 border-b pb-5">
        <div className="flex flex-wrap items-end gap-4">
          <div className="space-y-1.5"><span className="text-xs text-muted-foreground">Players</span>
            <TableFormat value={count} onChange={n => { setCount(n); if (n === 2 && position === "BTN") setPosition("SB"); }} />
          </div>
          <label className="flex flex-col gap-1.5 text-xs text-muted-foreground">Position
            <NativeSelect aria-label="Position" value={position} onChange={event => setPosition(event.target.value as Position)}>
              {(count === 3 ? ["BTN", "SB", "BB"] : ["SB", "BB"]).map(p => <option key={p}>{p}</option>)}
            </NativeSelect>
          </label>
          <label className="flex flex-col gap-1.5 text-xs text-muted-foreground">Street
            <NativeSelect aria-label="Street" value={street} onChange={event => setStreet(event.target.value as Street)}>
              {STREETS.map(s => <option key={s}>{s}</option>)}
            </NativeSelect>
          </label>
        </div>
        <div className="flex flex-wrap items-end gap-3">
          {stacks.slice(0, count).map((stack, i) => <label className="space-y-1.5 text-xs text-muted-foreground" key={i}>P{i} stack (BB)
            <Input className="h-10 w-28 text-foreground" type="number" min="0.01" step="0.5" value={stack}
              onChange={event => setStacks(stacks.map((value, index) => index === i ? Number(event.target.value) : value))} />
          </label>)}
          {street !== "PREFLOP" && <label className="space-y-1.5 text-xs text-muted-foreground">Board (optional)
            <Input className="h-10 w-44 text-foreground" placeholder="As Kh 2d" value={boardText} onChange={event => setBoardText(event.target.value)} />
          </label>}
          <Button className="h-10" onClick={build}>Apply state</Button>
        </div>
      </div>
      {!policy && <div role="status">Average policy unavailable for {Object.keys(hand.players).length} players; publish this track with migrate.py.</div>}
      {error && <div role="alert" className="text-destructive">{error}</div>}
      <div className="grid items-start gap-6 lg:grid-cols-[minmax(0,26rem)_minmax(0,1fr)]">
        <HoleCardsPicker value={hero} board={hand.board} disabled={cardsLocked || hand.terminal} onChange={cards => {
          try {
            setHand(withHeroCards(hand, cards)); setHero(cards); setError(null);
          } catch (cause) { setError(cause instanceof Error ? cause.message : String(cause)); }
        }} />
        <section className="min-w-0 space-y-4" aria-label="Action policy">
          <h3 className="text-sm font-medium">{hand.terminal ? "Hand settled" : "Actions"}</h3>
          {!probabilities && policy && !hand.terminal && <p role="status" className="text-sm text-muted-foreground">Loading policy…</p>}
          <div className="space-y-2">{actions.map(action => <Button variant="ghost" key={action.action_id}
            onClick={() => act(action.action_id)} className="h-auto w-full justify-between gap-3 border px-3 py-3 text-xs"
            style={{ backgroundColor: `color-mix(in srgb, ${color(action.action_id)} 12%, transparent)`, borderColor: `color-mix(in srgb, ${color(action.action_id)} 25%, transparent)` }}>
            <span className="text-left">{action.action_id}{action.amount_to !== null ? ` · ${(action.amount_to / hand.blinds.big).toFixed(2)} BB` : ""}</span>
            {probabilities && <span className="flex shrink-0 items-center gap-3">
              <span className="hidden h-1.5 w-20 overflow-hidden rounded bg-muted sm:block"><span className="block h-full rounded" style={{ width: `${(probabilities[action.action_id] ?? 0) * 100}%`, background: color(action.action_id) }} /></span>
              <span className="w-12 text-right tabular-nums">{((probabilities[action.action_id] ?? 0) * 100).toFixed(1)}%</span>
            </span>}
          </Button>)}</div>
        </section>
      </div>
      <details className="border-t pt-4"><summary className="cursor-pointer text-sm text-muted-foreground">Public action history ({hand.history.length})</summary><ol className="mt-3 space-y-1 text-sm">{hand.history.map((event, i) => <li key={i}>{event.street} · {event.position} · {event.action} {(event.amount_to / hand.blinds.big).toFixed(2)} BB · pot {(event.pot_after / hand.blinds.big).toFixed(2)} BB</li>)}</ol></details>
    </CardContent>
  </Card>;
}
