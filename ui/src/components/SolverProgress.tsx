"use client";
import {useEffect, useState} from "react";
import {Button} from "@/components/ui/button";
import {fetchSolverHistory, type SolverHistory} from "@/lib/solver-history";
const streets = ["PREFLOP", "FLOP", "TURN", "RIVER"];
const colors = ["#60a5fa", "#34d399", "#fbbf24", "#c084fc"];
export default function SolverProgress() {
  const [track, setTrack] = useState<"hu" | "3max">("hu");
  const [days, setDays] = useState(1);
  const [revision, setRevision] = useState(0);
  const [history, setHistory] = useState<SolverHistory | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    let alive = true;
    let controller = new AbortController();
    async function load() {
      controller.abort(); controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 15000);
      try { const h = await fetchSolverHistory(track, controller.signal); if (alive) {setHistory(h); setError("");} }
      catch (e) {if (alive) setError(e instanceof Error ? e.message : String(e));}
      finally {clearTimeout(timeout);}
    }
    setHistory(null); void load();
    const timer = setInterval(() => {if (!document.hidden) void load();}, 60000);
    return () => {alive = false; controller.abort(); clearInterval(timer);};
  }, [track, revision]);
  const latest = history?.runs.at(-1);
  const runs = history?.runs.filter(r => Date.parse(r.at) >= Date.now() - days * 86400000) ?? [];
  const maximum = Math.max(.01, ...runs.flatMap(r => r.streets.map(s => s.response_shortfall_bb)));
  const start = Date.parse(runs[0]?.at ?? ""), end = Date.parse(runs.at(-1)?.at ?? "");
  const x = (date: string) => 55 + (end === start ? .5 : (Date.parse(date)-start)/(end-start))*790;
  const y = (v: number) => 210 - v/maximum*180;
  return <section className="mx-auto max-w-6xl space-y-5 py-3">
    <div className="flex flex-wrap items-center justify-between gap-3"><h2 className="text-xl font-semibold">Training</h2><div className="flex gap-2">
      {(["hu", "3max"] as const).map(t => <Button key={t} size="sm" variant={track===t?"default":"secondary"} onClick={() => setTrack(t)}>{t==="hu"?"HU":"3-Max"}</Button>)}
      {[1,7,30].map(d => <Button key={d} size="sm" variant={days===d?"secondary":"ghost"} onClick={() => setDays(d)}>{d===1?"24h":`${d}d`}</Button>)}
      <Button size="sm" variant="outline" onClick={() => setRevision(v=>v+1)}>Refresh</Button>
    </div></div>
    {error && <p role="alert" className="text-sm text-destructive">{error}</p>}
    {!latest && !error && <p className="text-sm text-muted-foreground">Loading solver progress…</p>}
    {latest && <>
      <div className="flex flex-wrap justify-between gap-2 border-b pb-3 text-sm"><span>Computed CFR knowledge · {latest.tables} subgames · {latest.nodes.toLocaleString()} nodes</span><span className="text-muted-foreground">{new Date(latest.at).toLocaleString()}</span></div>
      <p className="text-sm text-muted-foreground">Fixed reference spots · action-value shortfall in BB · lower is better. This measures local solving, not full-game winnings.</p>
      <div className="flex gap-5 text-xs">{streets.map((s,i)=><span key={s} style={{color: colors[i]}}>{s}</span>)}</div>
      {runs.length > 0 && <svg viewBox="0 0 900 250" className="w-full" role="img" aria-label="Solver reference shortfall history">
        {[0,maximum/2,maximum].map(v => <g key={v}><line x1="55" x2="845" y1={y(v)} y2={y(v)} stroke="currentColor" opacity=".12"/><text x="45" y={y(v)+4} textAnchor="end" fill="currentColor" fontSize="11">{v.toFixed(3)}</text></g>)}
        {streets.map((s,i)=><g key={s}><polyline points={runs.map(r=>`${x(r.at)},${y(r.streets.find(v=>v.street===s)!.response_shortfall_bb)}`).join(" ")} fill="none" stroke={colors[i]} strokeWidth="2"/>{runs.map(r=><circle key={r.at} cx={x(r.at)} cy={y(r.streets.find(v=>v.street===s)!.response_shortfall_bb)} r="3" fill={colors[i]}/>)}</g>)}
        <text x="55" y="240" fill="currentColor" fontSize="11">{new Date(start).toLocaleString()}</text><text x="845" y="240" textAnchor="end" fill="currentColor" fontSize="11">{new Date(end).toLocaleString()}</text>
      </svg>}
      <div className="overflow-x-auto"><table className="w-full text-sm"><thead><tr className="border-b text-left"><th className="py-2">Street</th><th>Reference CFR iterations</th><th>Shortfall · BB</th><th>River response gain · BB</th></tr></thead><tbody>{latest.streets.map(s=><tr key={s.street} className="border-b"><td className="py-3">{s.street}</td><td>{s.iterations}</td><td>{s.response_shortfall_bb.toFixed(4)}</td><td>{s.river_response_gain_bb?.toFixed(4) ?? "—"}</td></tr>)}</tbody></table></div>
      <p className="text-xs text-muted-foreground">{latest.resolved_labels}/{latest.labels} response labels meet the precision target · {latest.failed_roots} roots reached their compute limit · neural assistance experimental</p>
    </>}
  </section>;
}
