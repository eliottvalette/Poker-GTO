"use client";
import { useEffect, useMemo, useState } from "react";
import { type LoadedAveragePolicy } from "@/lib/onnx-policy";
import { analysisRoot, queryRange, type RangeCell } from "@/lib/poker/policy-analysis";
import { legalActions } from "@/lib/poker/actions";
import { type Position } from "@/lib/poker/engine";
import { Button } from "./ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import styles from "./PolicyOverview.module.css";

const RANKS = ["A", "K", "Q", "J", "10", "9", "8", "7", "6", "5", "4", "3", "2"];
function groups(cell: RangeCell) {
  const fold = cell.probabilities.FOLD ?? 0;
  const passive = (cell.probabilities.CALL ?? 0) + (cell.probabilities.CHECK ?? 0);
  const raise = Object.entries(cell.probabilities).reduce((sum, [action, p]) =>
    ["FOLD", "CALL", "CHECK"].includes(action) ? sum : sum + p, 0);
  return { fold, passive, raise };
}
function label(action: string, target: number | null, big: number) {
  if (action === "ALL_IN") return `All in · ${(target! / big).toFixed(1)} BB`;
  if (target !== null && action.startsWith("RAISE")) return `Raise · ${(target / big).toFixed(1)} BB`;
  return action === "FOLD" ? "Fold" : action === "CALL" ? "Call" : "Check";
}

export default function PolicyOverview({ policies }: { policies: Record<number, LoadedAveragePolicy> }) {
  const [count, setCount] = useState(3);
  const [position, setPosition] = useState<Position>("BTN");
  const [cells, setCells] = useState<RangeCell[] | null>(null);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const hand = useMemo(() => analysisRoot(count, position, "PREFLOP",
    count === 3 ? [25, 25, 25] : [37.5, 37.5], 1, 42), [count, position]);
  const policy = policies[count];
  const actions = legalActions(hand);
  const selected = cells?.[selectedIndex];
  const spread = cells ? Math.max(...actions.map(action => {
    const values = cells.map(cell => cell.probabilities[action.action_id] ?? 0);
    return (Math.max(...values) - Math.min(...values)) * 100;
  })) : null;
  useEffect(() => {
    let cancelled = false;
    setCells(null); setProgress(0); setError(null);
    if (!policy) return;
    void queryRange(hand, policy, () => cancelled, done => { if (!cancelled) setProgress(done); })
      .then(result => { if (!cancelled) setCells(result); })
      .catch(cause => { if (!cancelled) setError(cause instanceof Error ? cause.message : String(cause)); });
    return () => { cancelled = true; };
  }, [hand, policy]);
  function selectTrack(next: number) {
    setCount(next); setPosition(next === 3 ? "BTN" : "SB"); setSelectedIndex(0);
  }
  return <Card className={styles.overview}>
    <CardHeader className={styles.header}>
      <div><CardTitle>Preflop overview</CardTitle><p className={styles.subtitle}>Balanced stacks · {count === 3 ? "25 BB each" : "37.5 BB each"}{policy ? ` · Iteration ${policy.manifest.iteration}` : ""}</p></div>
      <div className={styles.controls}>
        <div role="group" aria-label="Player count">{[3, 2].map(n => <Button key={n} size="sm" variant={count === n ? "default" : "secondary"} aria-pressed={count === n} onClick={() => selectTrack(n)}>{n === 3 ? "3-max" : "HU"}</Button>)}</div>
        <div role="group" aria-label="Position">{(count === 3 ? ["BTN", "SB", "BB"] : ["SB", "BB"]).map(p => <Button key={p} size="sm" variant={position === p ? "default" : "secondary"} aria-pressed={position === p} onClick={() => setPosition(p as Position)}>{p}</Button>)}</div>
      </div>
    </CardHeader>
    <CardContent className="space-y-4">
      <div className={styles.context}><span>{position} · {hand.history.filter(event => event.action !== "BLIND").length === 0 ? "Unopened pot" : "After previous positions call"} · Pot {(hand.pot / hand.blinds.big).toFixed(1)} BB</span><span>Postflop, custom stacks and betting lines: Cas précis.</span></div>
      {!policy && <div role="status">No published {count === 3 ? "3-max" : "HU"} policy. Export this track with migrate.py.</div>}
      {error && <div role="alert" className="text-destructive">{error}</div>}
      {policy && !error && <div className={styles.layout}>
        <div>
          <div className={styles.legend}><span><i className={styles.fold} />Fold</span><span><i className={styles.passive} />Call / check</span><span><i className={styles.raise} />Raise / all in</span></div>
          {!cells && <div role="status" className={styles.loading}>Computing range · {progress} / 1326 combos</div>}
          {cells && <div className={styles.matrix} aria-label="Preflop action range">
            <span />{RANKS.map(rank => <span key={rank} className={styles.rank}>{rank}</span>)}
            {RANKS.map((rank, row) => <div key={rank} className={styles.row}>
              <span className={styles.rank}>{rank}</span>
              {cells.slice(row * 13, row * 13 + 13).map((cell, col) => {
                const mix = groups(cell), index = row * 13 + col;
                return <button key={cell.label} className={`${styles.cell} ${index === selectedIndex ? styles.selected : ""}`} aria-pressed={index === selectedIndex}
                  aria-label={`${cell.label}: fold ${(mix.fold * 100).toFixed(1)}%, call/check ${(mix.passive * 100).toFixed(1)}%, raise/all in ${(mix.raise * 100).toFixed(1)}%`}
                  onClick={() => setSelectedIndex(index)} title={`${cell.label} · ${cell.combos} exact combos`}>
                  <span className={styles.mix}><i className={styles.fold} style={{ width: `${mix.fold * 100}%` }} /><i className={styles.passive} style={{ width: `${mix.passive * 100}%` }} /><i className={styles.raise} style={{ width: `${mix.raise * 100}%` }} /></span>
                  <span className={styles.hand}>{cell.label}</span>
                </button>;
              })}
            </div>)}
          </div>}
          {spread !== null && <p className={styles.caption}>1326 exact combos, averaged within each class. Largest action difference between classes: {spread.toFixed(1)} percentage points.</p>}
        </div>
        <aside className={styles.detail}>
          <h2>{selected?.label ?? "Select a hand"}</h2>
          <p className={styles.subtitle}>{selected ? `${selected.combos} exact combos · ${position}` : "Action frequencies appear here."}</p>
          {selected && actions.map(action => {
            const probability = selected.probabilities[action.action_id] ?? 0;
            const category = action.category === "FOLD" ? styles.fold : action.category === "RAISE" ? styles.raise : styles.passive;
            return <div key={action.action_id} className={styles.action}><div><span>{label(action.action_id, action.amount_to, hand.blinds.big)}</span><strong>{(probability * 100).toFixed(1)}%</strong></div><div className={styles.track}><i className={category} style={{ width: `${probability * 100}%` }} /></div></div>;
          })}
          <p className={styles.caption}>Each percentage comes from the loaded average policy for this hand and public state.</p>
        </aside>
      </div>}
    </CardContent>
  </Card>;
}
