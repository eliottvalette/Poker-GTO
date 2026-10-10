"use client";

import { useEffect, useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { evaluationLabel, fetchTrainingHistory, historyWindow, type EvaluationRun, type Estimate, type Period, type TrainingHistory } from "@/lib/training-history";

function number(value: number | null, digits = 1) { return value === null ? "—" : value.toFixed(digits); }
function interval(value: Estimate) { return value.interval ? `${number(value.interval[0])} to ${number(value.interval[1])}` : "Not enough independent groups"; }

function HistoryChart({ runs, metric, selected, select }: {
  runs: EvaluationRun[]; metric: "gain" | "difference"; selected: string; select: (id: string) => void;
}) {
  const points = runs.filter(r => r.status !== "failed" && r[metric].mean !== null);
  if (!points.length) return <div className="flex h-52 items-center justify-center text-sm text-muted-foreground">No completed deal groups in this period.</div>;
  const values = points.flatMap(r => [r[metric].mean!, ...(r[metric].interval ?? [])]);
  const low = Math.min(0, ...values), high = Math.max(0, ...values);
  const pad = Math.max(1, (high - low) * 0.1);
  const start = Date.parse(runs[0].evaluated_at), end = Date.parse(runs[runs.length - 1].evaluated_at);
  const x = (r: EvaluationRun) => 72 + 788 * (end === start ? 0.5 : (Date.parse(r.evaluated_at) - start) / (end - start));
  const y = (v: number) => 210 - 180 * (v - low + pad) / (high - low + 2 * pad);
  return <svg viewBox="0 0 900 250" role="group" aria-label="Policy performance history in big blinds per 100 hands" className="w-full min-h-48">
    {[low, (low + high) / 2, high].filter((v, i, a) => a.indexOf(v) === i).map(v => <g key={v}>
      <line x1="72" x2="860" y1={y(v)} y2={y(v)} stroke="currentColor" opacity="0.12" />
      <text x="60" y={y(v) + 4} textAnchor="end" fill="currentColor" fontSize="11" opacity="0.65">{number(v)}</text>
    </g>)}
    {points.map(r => <g key={r.release_id}>
      {r[metric].interval && <line x1={x(r)} x2={x(r)} y1={y(r[metric].interval![0])} y2={y(r[metric].interval![1])} stroke="#60a5fa" strokeWidth="3" opacity="0.3" />}
      <circle cx={x(r)} cy={y(r[metric].mean!)} r={r.release_id === selected ? 5 : 3} fill={r.status === "budget_limited" ? "#fbbf24" : "#60a5fa"} />
      <circle cx={x(r)} cy={y(r[metric].mean!)} r="10" fill="transparent" role="button" tabIndex={0}
        aria-label={`Select iteration ${r.iteration}: ${number(r[metric].mean)} BB per 100 hands`}
        onClick={() => select(r.release_id)} onKeyDown={e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); select(r.release_id); } }} className="cursor-pointer" />
    </g>)}
    <text x="72" y="240" fill="currentColor" fontSize="11" opacity="0.65">{new Date(start).toLocaleString()}</text>
    <text x="860" y="240" textAnchor="end" fill="currentColor" fontSize="11" opacity="0.65">{new Date(end).toLocaleString()}</text>
  </svg>;
}

export default function TrainingProgress({ published = {} }: { published?: Partial<Record<"hu" | "3max", number>> }) {
  const [track, setTrack] = useState<"hu" | "3max">("hu");
  const [period, setPeriod] = useState<Period>("24h");
  const [metric, setMetric] = useState<"gain" | "difference">("difference");
  const [history, setHistory] = useState<TrainingHistory | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [revision, setRevision] = useState(0);
  const [selected, setSelected] = useState("");
  useEffect(() => {
    let alive = true;
    let request: AbortController | null = null;
    setHistory(null); setSelected(""); setLoading(true); setError(null);
    async function load() {
      request?.abort();
      request = new AbortController();
      const timeout = setTimeout(() => request?.abort(), 15000);
      try {
        const next = await fetchTrainingHistory(track, request.signal);
        if (alive) { setHistory(next); setError(null); }
      } catch (e) {
        if (alive) setError(e instanceof Error ? e.message : String(e));
      } finally { clearTimeout(timeout); if (alive) setLoading(false); }
    }
    void load();
    const timer = setInterval(() => { if (!document.hidden) void load(); }, 300000);
    return () => { alive = false; clearInterval(timer); request?.abort(); };
  }, [track, revision]);
  const runs = useMemo(() => historyWindow(history?.runs ?? [], period), [history, period]);
  const latest = runs.at(-1);
  const comparable = runs.filter(r => r.suite === latest?.suite && r.reference.model_sha256 === latest?.reference.model_sha256);
  const current = comparable.find(r => r.release_id === selected) ?? comparable.at(-1);

  return <section className="mx-auto max-w-6xl space-y-5 py-3">
    <div className="flex flex-wrap items-center justify-between gap-3">
      <h2 className="text-xl font-semibold">Training</h2>
      <div className="flex flex-wrap gap-2">
        {(["hu", "3max"] as const).map(t => <Button key={t} variant={track === t ? "default" : "secondary"} onClick={() => setTrack(t)}>{t === "hu" ? "HU" : "3-Max"}</Button>)}
        {(["24h", "7d", "30d"] as const).map(p => <Button key={p} variant={period === p ? "default" : "outline"} onClick={() => setPeriod(p)}>{p}</Button>)}
        <Button variant="outline" onClick={() => setRevision(v => v + 1)}>Refresh</Button>
      </div>
    </div>
    {loading && <p role="status" className="text-sm text-muted-foreground">Loading evaluations…</p>}
    {error && <p role="alert" className="text-sm text-destructive">{error}{history ? " Showing the last loaded results." : ""}</p>}
    {published[track] !== undefined && latest && published[track]! > latest.iteration && <p role="status" className="text-sm text-muted-foreground">Published policy #{published[track]} · latest evaluation #{latest.iteration}</p>}
    {latest && Date.now() - Date.parse(latest.evaluated_at) > 3 * 3600000 && <p role="status" className="text-sm text-amber-500">No new evaluation for over 3 hours. Check the evaluation service if training is still running.</p>}
    {!loading && !error && !latest && <div className="rounded-xl border p-8 text-sm text-muted-foreground">No evaluations in this period. Results appear after the VPS evaluates a published model.</div>}
    {current && <>
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        {[["Policy", `Iteration ${current.iteration}`, evaluationLabel(current)],
          ["Gain vs initial reference", `${number(current.difference.mean)} BB/100`, `Approx. 95%: ${interval(current.difference)}`],
          ["Independent groups", String(current.completed_groups), `${current.hands} hands · ${current.discarded_hands} discarded`],
          ["Evaluation CPU", `${number(current.cpu_seconds)} s`, new Date(current.evaluated_at).toLocaleString()]].map(([label, value, note]) =>
          <div key={label} className="rounded-xl border p-4"><p className="text-xs text-muted-foreground">{label}</p><p className="mt-2 text-lg font-medium tabular-nums">{value}</p><p className="mt-1 text-xs text-muted-foreground">{note}</p></div>)}
      </div>
      <div className="rounded-xl border p-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex gap-2"><Button variant={metric === "difference" ? "secondary" : "ghost"} onClick={() => setMetric("difference")}>vs reference</Button><Button variant={metric === "gain" ? "secondary" : "ghost"} onClick={() => setMetric("gain")}>vs opponent pool</Button></div>
          <span className="text-xs text-muted-foreground">Policy only · BB/100 · reference #{current.reference.iteration}</span>
        </div>
        <HistoryChart runs={comparable} metric={metric} selected={current.release_id} select={setSelected} />
        <p className="text-xs text-muted-foreground">Bars show approximate 95% intervals. Amber points hit the compute budget. Select a point to inspect it.</p>
        {comparable.length < runs.length && <p className="mt-2 text-xs text-muted-foreground">Older evaluation suites are excluded from this comparison.</p>}
      </div>
      <div className="overflow-x-auto rounded-xl border">
        <table className="w-full text-left text-sm"><caption className="p-4 text-left font-medium">Opponent breakdown · iteration {current.iteration}</caption>
          <thead className="border-y text-xs text-muted-foreground"><tr>{["Opponent", "Stacks", "Gain BB/100", "vs reference", "Approx. 95% difference", "Groups"].map(h => <th className="px-4 py-2 font-normal" key={h}>{h}</th>)}</tr></thead>
          <tbody>{current.segments.map(s => <tr key={`${s.profile}/${s.region}`} className="border-b last:border-0"><td className="px-4 py-2 capitalize">{s.profile.replaceAll("_", " ")}</td><td className="px-4 py-2">{s.region}</td><td className="px-4 py-2 tabular-nums">{number(s.gain.mean)}</td><td className="px-4 py-2 tabular-nums">{number(s.difference.mean)}</td><td className="px-4 py-2 tabular-nums">{interval(s.difference)}</td><td className="px-4 py-2">{s.difference.n}</td></tr>)}</tbody>
        </table>
      </div>
      <details className="rounded-xl border p-4 text-sm"><summary className="cursor-pointer">Evaluation details</summary>
        <div className="mt-3 space-y-2 text-muted-foreground">
          <p>{current.suite} · Model {current.model_sha256.slice(0, 12)}</p>
          <p>Conditional river response gain: {current.river ? `${number(current.river.best_response_gain_bb, 4)} BB (${current.river.nodes} nodes)` : "Not evaluated"}. This is not full-game exploitability.</p>
          {current.warnings.map(w => <p key={w}>{w}</p>)}
          <p>Initial reference is a fixed comparison, not a certified strong policy. No automatic quality promotion. Hourly history: 30 days. Local detailed traces: 48 hours.</p>
        </div>
      </details>
      <details className="rounded-xl border p-4 text-sm"><summary className="cursor-pointer">All evaluations ({runs.length})</summary><div className="mt-3 flex flex-wrap gap-2">{runs.map(r => <Button variant="outline" key={r.release_id} disabled={!comparable.includes(r)} onClick={() => setSelected(r.release_id)}>#{r.iteration} · {evaluationLabel(r)}</Button>)}</div></details>
    </>}
  </section>;
}
