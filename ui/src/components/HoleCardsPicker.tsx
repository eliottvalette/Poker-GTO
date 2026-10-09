"use client";
import { useState } from "react";
import { cardLabel } from "@/lib/game";
import HandMatrix, { HAND_CLASSES } from "./HandMatrix";

function combinations(row: number, column: number): [number, number][] {
  const first = (12 - row) * 4, second = (12 - column) * 4;
  const pairs: [number, number][] = [];
  for (let a = 0; a < 4; a++) for (let b = 0; b < 4; b++) {
    if (row === column ? a < b : row < column ? a === b : a !== b) pairs.push([first + a, second + b]);
  }
  return pairs;
}

export default function HoleCardsPicker({ value, board, disabled, onChange }: {
  value: [number, number]; board: number[]; disabled: boolean; onChange: (cards: [number, number]) => void;
}) {
  const [opened, setOpened] = useState<string | null>(null);
  const ranks = value.map(card => 12 - Math.floor(card / 4));
  const high = Math.min(...ranks), low = Math.max(...ranks);
  const selectedIndex = value[0] % 4 === value[1] % 4 ? high * 13 + low : low * 13 + high;
  const selected = HAND_CLASSES[selectedIndex].label;
  const active = HAND_CLASSES.find(cell => cell.label === (opened ?? selected))!;
  const available = (row: number, column: number) => combinations(row, column).filter(cards => cards.every(card => !board.includes(card)));

  return <section className="min-w-0 space-y-3" aria-label="Acting player's hand">
    <div className="flex items-center justify-between gap-3">
      <h3 className="text-sm font-medium">Acting player’s hand</h3>
      <span className="font-mono text-sm">{value.map(cardLabel).join(" ")}</span>
    </div>
    <HandMatrix>{({ label, row, column }) => <button key={label} type="button" aria-pressed={selected === label}
      disabled={disabled || available(row, column).length === 0} onClick={() => setOpened(label)}
      className={`min-w-0 rounded-sm border py-2 text-center text-[10px] font-semibold transition-colors hover:bg-accent focus-visible:outline-2 disabled:opacity-35 ${selected === label ? "border-blue-400 bg-blue-500/25" : active.label === label ? "border-blue-400/50 bg-blue-500/10" : "border-border bg-muted/40"}`}>
      {label}
    </button>}</HandMatrix>
    <div className="flex flex-wrap gap-1.5" role="group" aria-label={`${active.label} exact combinations`}>
      {available(active.row, active.column).map(cards => <button key={cards.join("-")} type="button" disabled={disabled}
        aria-pressed={cards.every(card => value.includes(card))} onClick={() => onChange(cards)}
        className="rounded border px-2 py-1 text-xs aria-pressed:border-blue-400 aria-pressed:bg-blue-500/20 hover:bg-accent disabled:opacity-35">
        {cards.map(cardLabel).join(" ")}
      </button>)}
    </div>
    <p className="text-xs text-muted-foreground">{disabled ? "Apply state to choose another hand." : "Select a hand class, then its exact suits."}</p>
  </section>;
}
