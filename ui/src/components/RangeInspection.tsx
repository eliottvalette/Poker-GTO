"use client";
import { useState } from "react";
import { handClassMasses, type RangeSnapshot } from "@/lib/poker/beliefs";
import { rangeColor } from "@/lib/poker/range-display";
import { cardLabel } from "@/lib/game";

const RANKS = "AKQJT98765432";
export default function RangeInspection({ seat, snapshot }: { seat: number; snapshot: RangeSnapshot | null }) {
  const [mode, setMode] = useState<"range" | "reactions">("range");
  const [selected, setSelected] = useState<string | null>(null);
  if (!snapshot || snapshot.status === "updating") return <p role="status">Updating estimated range…</p>;
  if (snapshot.status === "error") return <p role="alert">Range unavailable: {snapshot.error}</p>;
  const rows = snapshot.marginals?.[seat];
  if (!rows) return <p role="alert">No current range for this seat.</p>;
  const display = snapshot.display;
  if (!display) return <p role="status">Updating range display…</p>;
  const classes = handClassMasses(rows), reactions = display.reactions[seat];
  const activeMode = mode === "reactions" && reactions ? "reactions" : "range";
  const composition = display.composition[seat];
  const colors = ["#a78bfa", "#22d3ee", "#fbbf24", "#64748b"];
  const combos = selected ? rows.filter(row => Object.hasOwn(handClassMasses([row]), selected)) : rows;
  return <section aria-label={`Player ${seat} estimated range`} className="space-y-2">
    <h3 className="font-semibold">P{seat} · Estimated range</h3>
    <div className="flex items-center gap-2 text-xs">
      <button className="rounded border px-2 py-1" aria-pressed={activeMode==="range"} onClick={()=>setMode("range")}>Range</button>
      <button className="rounded border px-2 py-1 disabled:opacity-40" aria-pressed={activeMode==="reactions"} disabled={!reactions} title={reactions?"Assumed action frequencies":"Available when this player acts"} onClick={()=>setMode("reactions")}>Reactions</button>
    </div>
    {activeMode === "range" ? <div className="flex justify-between text-[10px]" aria-label="Range color legend"><span className="text-orange-400">≤¼× Less likely</span><span className="text-slate-400">1× Uniform</span><span className="text-cyan-400">More likely ≥4×</span></div>
      : <div className="flex gap-3 text-[10px]"><span className="text-blue-400">Fold</span><span className="text-emerald-400">Check / call</span><span className="text-red-400">Bet / raise</span><span className="text-muted-foreground">{snapshot.profiles[seat]}</span></div>}
    <div className="grid gap-px" style={{gridTemplateColumns:"repeat(13,minmax(0,1fr))"}} aria-label="169 hand classes">
      {[...RANKS].flatMap((a,i) => [...RANKS].map((b,j) => {
        const label = i === j ? a+b : i < j ? a+b+"s" : b+a+"o", mass = classes[label] ?? 0;
        const feasible=display.feasibleCounts[label]??0, ratio=feasible?mass/display.uniformMass[label]:0, mix=reactions?.[label];
        const background = !feasible ? "repeating-linear-gradient(135deg,#111827 0px,#111827 3px,#475569 3px,#475569 4px)"
          : activeMode === "reactions" && mix ? `linear-gradient(to right,#1d4ed8 0%,#1d4ed8 ${mix.fold*100}%,#047857 ${mix.fold*100}%,#047857 ${(mix.fold+mix.passive)*100}%,#b91c1c ${(mix.fold+mix.passive)*100}%,#b91c1c 100%)` : rangeColor(ratio);
        const explanation = !feasible ? "Blocked" : activeMode === "reactions" && mix ? `Fold ${(mix.fold*100).toFixed(1)}%, check/call ${(mix.passive*100).toFixed(1)}%, bet/raise ${(mix.aggressive*100).toFixed(1)}%` : `${ratio.toFixed(2)}× uniform · ${feasible} unblocked combos`;
        return <button key={label} title={explanation} type="button" aria-label={`${label}: ${(mass*100).toFixed(3)}%`} aria-pressed={selected===label}
          className="min-w-0 rounded-sm py-1 text-[10px] leading-tight focus-visible:outline-2 focus-visible:outline-primary border border-border"
          style={{background, color:"#fff", textShadow:"0 1px 2px #000"}}
          onClick={()=>setSelected(selected===label ? null : label)}>
          <span className="block font-semibold">{label}</span><span className="block">{feasible ? `${(mass*100).toFixed(1)}%` : "×"}</span>
        </button>;
      }))}
    </div>
    {composition && <div className="space-y-1" aria-label="Range composition" title="Best five-card hand including the board. Draws are unpaired flush/straight draws before the river.">
      <div className="flex h-2 overflow-hidden rounded">{Object.entries(composition).map(([label,mass],i)=><span key={label} title={`${label}: ${(mass*100).toFixed(1)}%`} style={{width:`${mass*100}%`,backgroundColor:colors[i]}} />)}</div>
      <div className="flex flex-wrap gap-x-3 text-[10px]">{Object.entries(composition).map(([label,mass],i)=><span key={label} style={{color:colors[i]}}>{label} {(mass*100).toFixed(0)}%</span>)}</div>
    </div>}
    <details open={selected !== null}>
      <summary className="cursor-pointer text-sm">{selected ?? "All exact combinations"} · {combos.length} combos</summary>
      {selected && <button className="text-xs underline" onClick={()=>setSelected(null)}>Show all combinations</button>}
      <div className="max-h-40 overflow-auto"><table className="w-full text-xs"><thead><tr><th className="text-left">Combination</th><th className="text-right">Posterior</th></tr></thead>
        <tbody>{[...combos].sort((a,b)=>b.probability-a.probability).map(row=><tr key={row.cards.join()}><td>{row.cards.map(cardLabel).join(" ")}</td><td className="text-right">{(row.probability*100).toFixed(4)}%</td></tr>)}</tbody>
      </table>{!combos.length && <p>No feasible combination in this class.</p>}</div>
    </details>
  </section>;
}
