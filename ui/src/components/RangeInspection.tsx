"use client";
import { useState } from "react";
import { handClassMasses, type RangeSnapshot } from "@/lib/poker/beliefs";
import { rangeColor } from "@/lib/poker/range-display";
import { formatRangeCopy } from "@/lib/poker/range-copy";
import type { HandState } from "@/lib/poker/engine";

import HandMatrix from "./HandMatrix";
export default function RangeInspection({ seat, snapshot, hand }: { seat: number; snapshot: RangeSnapshot | null; hand: Pick<HandState, "board" | "history"> }) {
  const [copyState, setCopyState] = useState<"idle" | "copying" | "copied">("idle");
  const [copyError, setCopyError] = useState<string | null>(null);
  if (!snapshot || snapshot.status === "updating") return <p role="status">Updating estimated range…</p>;
  if (snapshot.status === "error") return <p role="alert">Range unavailable: {snapshot.error}</p>;
  const rows = snapshot.marginals?.[seat];
  if (!rows) return <p role="alert">No current range for this seat.</p>;
  const display = snapshot.display;
  if (!display) return <p role="status">Updating range display…</p>;
  const classes = handClassMasses(rows);
  const maximumMass = Math.max(0, ...Object.values(classes));
  const composition = display.composition[seat];
  const colors = ["#a78bfa", "#22d3ee", "#fbbf24", "#64748b"];
  return <section aria-label={`Player ${seat} estimated range`} className="space-y-2">
    <div className="flex items-center justify-between gap-2">
      <h3 className="font-semibold">P{seat}</h3>
      <span className="text-sm tabular-nums">Equity {display.equities?.[seat] ? `≈ ${(display.equities[seat].probability*100).toFixed(0)}%` : "…"}</span>
      <button type="button" className="rounded border px-2 py-1 text-xs" disabled={copyState === "copying"}
        onClick={async () => {
          setCopyState("copying"); setCopyError(null);
          try {
            if (!navigator.clipboard) throw new Error("Clipboard unavailable in this browser context");
            await navigator.clipboard.writeText(formatRangeCopy(seat, hand, rows, snapshot.profiles[seat]));
            setCopyState("copied");
          } catch (error) {
            setCopyState("idle"); setCopyError(error instanceof Error ? error.message : String(error));
          }
        }}>{copyState === "copied" ? "Copied" : copyState === "copying" ? "Copying…" : "Copy"}</button>
    </div>
    {copyError && <p role="alert" className="text-xs text-destructive">Copy failed: {copyError}</p>}
    <div className="space-y-1 text-[10px]" aria-label="Range probability color scale">
      <div className="flex justify-between"><span className="text-zinc-400">0%</span><span>Hand probability</span><span>{(maximumMass*100).toFixed(1)}% · max</span></div>
      <div className="h-2 rounded" style={{background:`linear-gradient(to right, rgb(39,39,42), ${rangeColor(maximumMass/2,maximumMass)}, ${rangeColor(maximumMass,maximumMass)})`}} />
    </div>
    <HandMatrix>{({ label }) => {
        const mass = classes[label] ?? 0;
        const feasible=display.feasibleCounts[label]??0;
        const background = !feasible ? "repeating-linear-gradient(135deg,#111827 0px,#111827 3px,#475569 3px,#475569 4px)"
          : rangeColor(mass,maximumMass);
        const explanation = !feasible ? "Blocked" : `${(mass*100).toFixed(3)}% hand probability · ${feasible} unblocked combos`;
        return <div key={label} title={explanation} aria-label={`${label}: ${(mass*100).toFixed(3)}%`}
          className="min-w-0 rounded-sm py-1 text-center text-[10px] leading-tight border border-border"
          style={{background, color:Number((mass*100).toFixed(1)) === 0 ? "#a1a1aa" : "#fff", textShadow:"0 1px 2px #000"}}>
          <span className="block font-semibold">{label}</span><span className="block">{feasible ? `${(mass*100).toFixed(1)}%` : "×"}</span>
        </div>;
      }}</HandMatrix>
    {composition && <div className="space-y-1" aria-label="Range composition" title="Best five-card hand including the board. Draws are unpaired flush/straight draws before the river.">
      <div className="flex h-2 overflow-hidden rounded">{Object.entries(composition).map(([label,mass],i)=><span key={label} title={`${label}: ${(mass*100).toFixed(1)}%`} style={{width:`${mass*100}%`,backgroundColor:colors[i]}} />)}</div>
      <div className="flex flex-wrap gap-x-3 text-[10px]">{Object.entries(composition).map(([label,mass],i)=><span key={label} style={{color:colors[i]}}>{label} {(mass*100).toFixed(0)}%</span>)}</div>
    </div>}
  </section>;
}
