"use client";
import { useEffect, useRef, useState } from "react";
import { Button } from "./ui/button";
import { PROFILES } from "@/lib/poker/behavior";
import { handClassMasses, recoverHandStart, rangeStateKey, type RangeSnapshot, type BeliefTransition } from "@/lib/poker/beliefs";
import { HandState } from "@/lib/poker/engine";
import { withHeroCards } from "@/lib/poker/policy-analysis";
import { HYBRID_BUDGETS, transportHand, uniformRanges, type HybridAnalysis as Result, type Ranges } from "@/lib/poker/hybrid";

export default function HybridAnalysis({ hand, hero, baseline, sessionId, reconstructHistory = false, initialProfiles = {}, transitioning = false, observerSeat, onRangeUpdate, opponentProfile, rangesOnly = false }: {
  hand: HandState; hero: [number, number]; baseline: Record<string, number> | null; sessionId: number;
  rangesOnly?: boolean; reconstructHistory?: boolean; initialProfiles?: Record<number,string>; transitioning?: boolean; observerSeat?: number; opponentProfile?: string; onRangeUpdate?: (snapshot: RangeSnapshot) => void;
}) {
  const [budget, setBudget] = useState("FAST");
  const [mode, setMode] = useState("reference");
  const [profiles, setProfiles] = useState<Record<number, string>>({});
  const [interpolate, setInterpolate] = useState(false);
  const [posterior, setPosterior] = useState<Ranges | null>(null);
  const trace = useRef<BeliefTransition[]>([]);
  const initial = useRef(hand);
  const previous = useRef(hand);
  const priorSession = useRef<number | null>(null);
  const [searchMode, setSearchMode] = useState<"behavior" | "public_cfr">("behavior");
  const [rangeText, setRangeText] = useState("");
  const [result, setResult] = useState<Result | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [latency, setLatency] = useState<number | null>(null);
  const worker = useRef<Worker | null>(null);
  const requestSequence = useRef(0);
  const heroKey = hero.join(",");
  const publicKey = JSON.stringify([hand.hand_number, hand.button, hand.board, hand.history, hand.current_player,
    hand.blinds, Object.values(hand.players).map(p => [p.player_id, p.position, p.stack, p.street_bet, p.contribution, p.folded])]);
  useEffect(() => () => { requestSequence.current++; worker.current?.terminate(); worker.current = null; }, []);
  const selectedProfiles = Object.fromEntries(Object.keys(hand.players).map(p => [p, opponentProfile && Number(p)!==observerSeat ? opponentProfile : profiles[Number(p)] ?? initialProfiles[Number(p)] ?? "conservative"]));
  useEffect(() => {
    if (transitioning) {
      requestSequence.current++; worker.current?.terminate(); worker.current = null;
      setBusy(false); setResult(null); setPosterior(null); return;
    }
    if (priorSession.current !== sessionId) {
      worker.current?.terminate(); worker.current = null;
      trace.current = []; initial.current = reconstructHistory ? recoverHandStart(hand) : hand;
      previous.current = initial.current; priorSession.current = sessionId;
    }
    if (previous.current !== hand) {
      let before = previous.current;
      if (JSON.stringify(hand.history.slice(0, before.history.length)) !== JSON.stringify(before.history)) {
        setError("Unexpected hand history replacement; use Apply state to start an explicit new session");
        return;
      }
      for (const event of hand.history.slice(before.history.length)) {
        if (event.action === "BLIND") { setError("Unexpected blind event inside a session"); return; }
        const after = before.clone(); after.act(event.action, event.action === "RAISE" ? event.amount_to : null);
        trace.current.push({ before: transportHand(before), after: transportHand(after),
          action: { action_id: "OBSERVED", category: event.action, amount_to: event.action === "RAISE" ? event.amount_to : null } });
        before = after;
      }
      if (before.board.join() !== hand.board.join() || before.pot !== hand.pot) { setError("Canonical history replay does not match this hand"); return; }
      previous.current = hand;
    }
    calculate(true);
    // Public beliefs are replayed from the declared prior when likelihood assumptions change.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [publicKey, sessionId, rangeText, profiles, interpolate, budget, mode, searchMode, heroKey, transitioning, observerSeat, opponentProfile]);

  function publishRanges(status: RangeSnapshot["status"], marginals?: Ranges, error?: string, display?: RangeSnapshot["display"]) {
    if (observerSeat !== undefined) onRangeUpdate?.({ sessionId, stateKey: rangeStateKey(hand, observerSeat), status, profiles: selectedProfiles, marginals, error, display });
  }

  function calculate(updateOnly = false) {
    if (updateOnly) publishRanges("updating");
    try {
      const state = updateOnly || hand.terminal ? hand : withHeroCards(hand, hero);
      const ranges: Ranges = rangeText.trim() ? JSON.parse(rangeText) : uniformRanges(initial.current);
      if (busy) { worker.current?.terminate(); worker.current = null; }
      const next = worker.current ?? new Worker(new URL("../lib/poker/hybrid.worker.ts", import.meta.url), { type: "module" });
      worker.current = next;
      const requestId = ++requestSequence.current;
      setBusy(true); setError(null); setResult(null);
      next.onmessage = (event: MessageEvent<{ requestId?: number; result?: Result; ranges?: Ranges; marginals?: Ranges; display?: RangeSnapshot["display"]; error?: string; workerMilliseconds?: number }>) => {
        if (event.data.requestId !== requestSequence.current) return;
        setBusy(false);
        if (event.data.error) { setError(event.data.error); setPosterior(null); publishRanges("error", undefined, event.data.error); }
        else {
          publishRanges("current", event.data.marginals, undefined, event.data.display);
          if (event.data.ranges) setPosterior(event.data.ranges);
          if (event.data.result) { setResult(event.data.result); setLatency(event.data.workerMilliseconds ?? null); }
        }
      };
      next.onerror = event => { if (requestId !== requestSequence.current) return; setError(event.message); publishRanges("error", undefined, event.message); setBusy(false); setPosterior(null); next.terminate(); worker.current = null; };
      next.postMessage({ requestId, observerSeat, state: transportHand(state), ranges, budget: HYBRID_BUDGETS[budget],
        searchMode: mode === "exploitative" ? "response" : searchMode,
        profiles: mode === "reference" && !updateOnly ? Object.fromEntries(Object.keys(hand.players).map(p => [p, "conservative"])) : selectedProfiles,
        likelihoodProfiles: selectedProfiles, transitions: trace.current, interpolate, updateOnly });
    } catch (cause) { const message = cause instanceof Error ? cause.message : String(cause); setError(message); publishRanges("error", undefined, message); setBusy(false); }
  }
  if (rangesOnly) return null;
  return <section className="space-y-3 border-t pt-4" aria-label="Hybrid analysis">
    <h3 className="font-semibold">Analysis</h3>
    <label>Calculation <select value={budget} onChange={event => setBudget(event.target.value)} className="rounded border p-2">
      {Object.keys(HYBRID_BUDGETS).map(name => <option key={name} value={name}>{{FAST:"Quick",NORMAL:"Standard",DEEP:"Detailed"}[name]}</option>)}
    </select></label>
    <p data-testid="belief-session" data-actions={trace.current.length} data-status={posterior ? "current" : "unavailable"} className="text-xs text-muted-foreground">{posterior ? "Ranges ready" : "Ranges updating"}</p>
    <details className="rounded border p-3"><summary className="cursor-pointer text-sm text-muted-foreground">Advanced settings</summary>
      <div className="mt-3 space-y-3">
    <label>Strategy <select aria-label="Decision mode" value={mode} onChange={event => { setMode(event.target.value); setResult(null); }}>
      <option value="reference">Reference</option><option value="exploitative">Adapt to profiles</option></select></label>
    <div className="flex flex-wrap gap-3">{Object.keys(hand.players).map(seat => <label key={seat}>P{seat} profile
      <select aria-label={`P${seat} behavior profile`} disabled={!!opponentProfile && Number(seat)!==observerSeat} value={selectedProfiles[Number(seat)]} onChange={event => setProfiles({ ...profiles, [seat]: event.target.value })}>
        {[...Object.keys(PROFILES), "uniform", ...(opponentProfile === "published" ? ["published"] : [])].map(name => <option key={name}>{name}</option>)}</select></label>)}</div>
    <label className="block text-sm"><input type="checkbox" checked={interpolate} onChange={event => setInterpolate(event.target.checked)} /> Estimate reactions to custom bet sizes</label>
    {posterior && <details><summary>Range data</summary>{Object.entries(posterior).map(([seat, rows]) => <details key={seat}><summary>P{seat}: {rows.length} exact combinations</summary><details><summary>169 hand classes</summary><pre className="max-h-48 overflow-auto text-xs">{JSON.stringify(handClassMasses(rows), null, 2)}</pre></details><pre className="max-h-48 overflow-auto text-xs">{JSON.stringify(rows, null, 2)}</pre></details>)}</details>}
    <label className="ml-3">Search <select value={searchMode} onChange={event => setSearchMode(event.target.value as "behavior" | "public_cfr")} className="rounded border p-2">
      <option value="behavior">Simulated responses</option><option value="public_cfr">CFR (small ranges)</option>
    </select></label>
    <p className="text-xs">Up to {HYBRID_BUDGETS[budget].maxNodes.toLocaleString()} nodes · {HYBRID_BUDGETS[budget].iterations} CFR iterations · {HYBRID_BUDGETS[budget].samples} simulations · depth {HYBRID_BUDGETS[budget].maxDepth}</p>
    <details><summary>Custom starting ranges</summary>
      <p className="text-sm">Leave empty for uniform ranges excluding the board. These are priors at the start of this session; observed actions and revealed boards update them automatically. Include Hero’s public range as well as each opponent. Cards use IDs 0–51; the same card cannot appear twice in a dealt world.</p>
      <label className="block">Range JSON<textarea aria-label="Public range JSON" className="block w-full rounded border p-2 font-mono text-xs" rows={5}
        value={rangeText} onChange={event => setRangeText(event.target.value)} placeholder={'{"0":[{"cards":[48,49],"probability":1}],"1":[{"cards":[44,45],"probability":1}]}'} /></label>
    </details>
      </div>
    </details>
    <div className="flex gap-2"><Button onClick={() => calculate(false)} disabled={busy || transitioning || hand.terminal}>Analyze locally</Button>
      {busy && <Button variant="secondary" onClick={() => { requestSequence.current++; worker.current?.terminate(); worker.current = null; setBusy(false); setError("Analysis cancelled; no partial result published"); }}>Cancel calculation</Button>}</div>
    {busy && <p role="status">Calculating…</p>}
    {error && <p role="alert" className="text-destructive">{error}</p>}
    {result && <div className="space-y-3" data-testid="hybrid-result">
      <details><summary className="cursor-pointer text-xs text-muted-foreground">Calculation details</summary><p className="text-sm">{result.method} · {result.nodes.toLocaleString()} nodes · {result.samples} samples · {result.iterations} iterations · {latency?.toFixed(1)} ms</p></details>
      <table className="w-full text-left text-sm"><thead><tr><th>Legal action / raise to</th><th>Optional NN baseline</th><th>Computed behavior baseline</th><th>Local strategy</th><th>EV (current BB)</th><th>Sampling SE (BB)</th></tr></thead>
        <tbody>{result.legalActions.map(action => <tr key={action.action_id}>
          <td>{action.action_id}{action.amount_to !== null ? ` · ${(action.amount_to / hand.blinds.big).toFixed(2)}` : ""}</td>
          <td>{baseline ? `${(100 * (baseline[action.action_id] ?? 0)).toFixed(1)}%` : "unavailable"}</td>
          <td>{result.baselineProbabilities ? `${(100 * result.baselineProbabilities[action.action_id]).toFixed(1)}%` : "unavailable"}</td>
          <td>{(100 * result.probabilities[action.action_id]).toFixed(1)}%</td>
          <td>{(result.ev[action.action_id] / hand.blinds.big).toFixed(4)}</td>
          <td>{result.standardErrors[action.action_id] === null ? "not estimated" : (result.standardErrors[action.action_id]! / hand.blinds.big).toFixed(4)}</td>
        </tr>)}</tbody></table>
      {result.warnings.filter(w => /unresolved|insufficient|exhaust|cancel/i.test(w)).map(warning => <p key={warning} className="text-xs">{warning}</p>)}
      <details><summary className="cursor-pointer text-xs text-muted-foreground">Assumptions</summary>{result.warnings.map(warning => <p key={warning} className="text-xs">{warning}</p>)}</details>
      <details><summary>Range distributions and Hero public range</summary>
        <p className="text-xs">Hero public support: {result.ownPublicRange.length} combinations. Opponent factors are jointly conditioned during calculation.</p>
        {Object.entries(result.ranges).map(([seat, rows]) => <details key={seat}><summary>Player {seat}: {rows.length} combinations</summary>
          <pre className="max-h-48 overflow-auto text-xs">{JSON.stringify(rows, null, 2)}</pre></details>)}
        <details><summary>Hero public distribution</summary><pre className="max-h-48 overflow-auto text-xs">{JSON.stringify(result.ownPublicRange, null, 2)}</pre></details>
      </details>
    </div>}
  </section>;
}
