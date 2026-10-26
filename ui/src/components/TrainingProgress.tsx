"use client";

import { useEffect, useMemo, useState } from "react";
import SelfPlayStatus from "./SelfPlayStatus";
import { Button } from "@/components/ui/button";
import { fetchTrainingHistory, historyWindow, trainingSeries, type EvaluationRun, type Estimate, type Period, type TrainingHistory } from "@/lib/training-history";

function number(value: number | null, digits = 1) { return value === null ? "—" : value.toFixed(digits); }
function interval(value: Estimate) { return value.interval ? `${number(value.interval[0])} to ${number(value.interval[1])}` : "Not enough independent groups"; }
const seriesColors = ["#60a5fa", "#34d399", "#c084fc", "#fb923c", "#f472b6"];

function HistoryChart({ runs, metric, selected, select }: {
  runs: EvaluationRun[]; metric: "gain" | "difference"; selected: string; select: (id: string) => void;
}) {
  const points = runs.filter(r => r.status !== "failed" && r[metric].mean !== null);
  const series = trainingSeries(runs);
  const colors = new Map(series.flatMap((group, i) => group.runs.map(run => [run.release_id, seriesColors[i % seriesColors.length]] as const)));
  if (!points.length) return <div className="flex h-52 items-center justify-center text-sm text-muted-foreground">No completed deal groups in this period.</div>;
  const values = points.flatMap(r => [r[metric].mean!, ...(r[metric].interval ?? [])]);
  const low = Math.min(0, ...values), high = Math.max(0, ...values);
  const pad = Math.max(1, (high - low) * 0.1);
  const start = Date.parse(runs[0].evaluated_at), end = Date.parse(runs[runs.length - 1].evaluated_at);
  const x = (r: EvaluationRun) => 72 + 788 * (end === start ? 0.5 : (Date.parse(r.evaluated_at) - start) / (end - start));
  const y = (v: number) => 210 - 180 * (v - low + pad) / (high - low + 2 * pad);
  return <><svg viewBox="0 0 900 250" role="group" aria-label="Policy performance history in big blinds per 100 hands" className="w-full min-h-48">
    {[low, (low + high) / 2, high].filter((v, i, a) => a.indexOf(v) === i).map(v => <g key={v}>
      <line x1="72" x2="860" y1={y(v)} y2={y(v)} stroke="currentColor" opacity="0.12" />
      <text x="60" y={y(v) + 4} textAnchor="end" fill="currentColor" fontSize="11" opacity="0.65">{number(v)}</text>
    </g>)}
    {series.flatMap(series => series.runs.reduce<EvaluationRun[][]>((groups, run) => {
      if (run.status === "failed" || run[metric].mean === null) groups.push([]);
      else groups[groups.length - 1].push(run);
      return groups;
    }, [[]]).filter(group => group.length > 1)).map(group => <g key={group[0].release_id}>
      {group.every(r => r[metric].interval) && <polygon
        points={[...group.map(r => `${x(r)},${y(r[metric].interval![1])}`), ...group.toReversed().map(r => `${x(r)},${y(r[metric].interval![0])}`)].join(" ")}
        fill={colors.get(group[0].release_id)} opacity="0.12" />}
      <polyline points={group.map(r => `${x(r)},${y(r[metric].mean!)}`).join(" ")}
        fill="none" stroke={colors.get(group[0].release_id)} strokeWidth="2" strokeLinejoin="round" />
    </g>)}
    {points.map(r => <g key={r.release_id}>
      {r[metric].interval && <line x1={x(r)} x2={x(r)} y1={y(r[metric].interval![0])} y2={y(r[metric].interval![1])} stroke={colors.get(r.release_id)} strokeWidth="3" opacity="0.3" />}
      <circle cx={x(r)} cy={y(r[metric].mean!)} r={r.release_id === selected ? 5 : 3} fill={colors.get(r.release_id)} />
      <circle cx={x(r)} cy={y(r[metric].mean!)} r="10" fill="transparent" role="button" tabIndex={0}
        aria-label={`Select iteration ${r.iteration}: ${number(r[metric].mean)} BB per 100 hands`}
        onClick={() => select(r.release_id)} onKeyDown={e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); select(r.release_id); } }} className="cursor-pointer" />
    </g>)}
    <text x="72" y="240" fill="currentColor" fontSize="11" opacity="0.65">{new Date(start).toLocaleString()}</text>
    <text x="860" y="240" textAnchor="end" fill="currentColor" fontSize="11" opacity="0.65">{new Date(end).toLocaleString()}</text>
  </svg>
    <ul aria-label="Evaluation series" className="flex flex-wrap gap-x-5 gap-y-2 text-xs text-muted-foreground">
      {series.map((group, i) => <li key={group.id} className="flex items-center gap-2">
        <span className="h-2 w-2 rounded-full" style={{ backgroundColor: seriesColors[i % seriesColors.length] }} />
        {group.label} · {new Date(group.runs[0].evaluated_at).toLocaleString()} · baseline #{group.runs[0].reference.iteration} · {group.runs.length} evaluations
      </li>)}
    </ul>
  </>;
}

export function LegacyTrainingProgress({ published = {} }: { published?: Partial<Record<"hu" | "3max", number>> }) {
  const [track, setTrack] = useState<"hu" | "3max">("hu");
  const [period, setPeriod] = useState<Period>("24h");
  const [metric, setMetric] = useState<"gain" | "difference">("difference");
  const [history, setHistory] = useState<TrainingHistory | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [revision, setRevision] = useState(0);
  const [selected, setSelected] = useState("");
  const [now, setNow] = useState(Date.now);
  useEffect(() => { setHistory(null); setSelected(""); }, [track]);
  useEffect(() => {
    let alive = true;
    let request: AbortController | null = null;
    setLoading(true); setError(null);
    async function load() {
      request?.abort();
      const controller = new AbortController();
      request = controller;
      const timeout = setTimeout(() => controller.abort(), 15000);
      try {
        const next = await fetchTrainingHistory(track, controller.signal);
        if (alive && request === controller) { setHistory(next); setError(null); }
      } catch (e) {
        if (alive && request === controller) setError(e instanceof Error ? e.message : String(e));
      } finally { clearTimeout(timeout); if (alive && request === controller) { setLoading(false); setNow(Date.now()); } }
    }
    void load();
    const refreshVisible = () => { if (!document.hidden) void load(); };
    const timer = setInterval(refreshVisible, 60000);
    document.addEventListener("visibilitychange", refreshVisible);
    return () => { alive = false; clearInterval(timer); document.removeEventListener("visibilitychange", refreshVisible); request?.abort(); };
  }, [track, revision]);
  const runs = useMemo(() => historyWindow(history?.runs ?? [], period, now), [history, period, now]);
  const latest = useMemo(() => historyWindow(history?.runs ?? [], "30d", now).at(-1), [history, now]);
  const current = runs.find(r => r.release_id === selected) ?? runs.at(-1);


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
    {latest && now - Date.parse(latest.evaluated_at) > 3 * 3600000 && <p role="status" className="text-sm text-amber-500">No new evaluation for over 3 hours. Evaluations are recorded when a new published model is evaluated; training may still be progressing.</p>}
    <p className="text-xs text-muted-foreground">{runs.length} evaluations across all experiments · checks for updates every minute and when you return to this tab. Each point represents an evaluated publication.</p>
    {!loading && !error && !runs.length && <div className="rounded-xl border p-8 text-sm text-muted-foreground">No evaluations in this period. Results appear after the VPS evaluates a published model.</div>}
    {current && <>
      <div className="flex flex-wrap items-center justify-between gap-3 border-b pb-3">
        <div className="flex gap-1">
          <Button size="sm" variant={metric === "difference" ? "secondary" : "ghost"} onClick={() => setMetric("difference")}>vs reference</Button>
          <Button size="sm" variant={metric === "gain" ? "secondary" : "ghost"} onClick={() => setMetric("gain")}>vs opponent pool</Button>
        </div>
        <span className="text-xs text-muted-foreground">NN policy · BB/100{metric === "difference" ? ` · baseline #${current.reference.iteration}` : " · synthetic opponents"}</span>
      </div>
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div className="flex flex-wrap items-baseline gap-3">
          <strong className="text-2xl font-medium tabular-nums">{current.is_reference && metric === "difference" ? "Baseline" : `${number(current[metric].mean)} BB/100`}</strong>
          {!(current.is_reference && metric === "difference") && <span className="text-xs text-muted-foreground">95% interval {interval(current[metric])}</span>}
        </div>
        <span className="text-xs text-muted-foreground">Iteration {current.iteration} · {new Date(current.evaluated_at).toLocaleString()}</span>
      </div>
      {current.status === "failed" && <p role="alert" className="text-sm text-destructive">Evaluation failed. {current.warnings.join(" ")}</p>}
      {current.status === "budget_limited" && <p role="status" className="text-sm text-amber-500">Partial evaluation: compute budget reached.</p>}
      <HistoryChart runs={runs} metric={metric} selected={current.release_id} select={setSelected} />

      <div className="overflow-x-auto">
        <table className="w-full text-left text-sm">
          <caption className="pb-3 text-left text-xs text-muted-foreground">{metric === "difference" ? `Change vs #${current.reference.iteration}` : "Win rate"} by opponent and stack · BB/100 · 95% intervals</caption>
          <thead className="border-b text-xs text-muted-foreground"><tr><th className="py-2 font-normal">Opponent</th>{[...new Set(current.segments.map(s => s.region))].map(region => <th key={region} className="px-4 py-2 text-right font-normal capitalize">{region}</th>)}</tr></thead>
          <tbody>{[...new Set(current.segments.map(s => s.profile))].map(profile => <tr key={profile} className="border-b last:border-0">
            <td className="py-3 capitalize">{profile.replaceAll("_", " ")}</td>
            {[...new Set(current.segments.map(s => s.region))].map(region => {
              const estimate = current.segments.find(s => s.profile === profile && s.region === region)?.[metric];
              return <td key={region} className="px-4 py-3 text-right tabular-nums">{estimate ? <><span>{number(estimate.mean)}</span><span className="block text-xs text-muted-foreground">{interval(estimate)}</span></> : "—"}</td>;
            })}
          </tr>)}</tbody>
        </table>
      </div>
    </>}
  </section>;
}

export default function TrainingProgress() {
  return <><SelfPlayStatus /><LegacyTrainingProgress /></>;
}
