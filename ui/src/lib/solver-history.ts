import { policyStorageBase } from "./policy-catalog";
export type SolverRow = { at: string; visits: number; tables: number; labels: number; resolved_labels: number; failed_roots: number; nodes: number; cpu_seconds: number; student_accepted: false; streets: {street: string; response_shortfall_bb: number; river_response_gain_bb: number | null; iterations: number}[] };
export type SolverHistory = {engine: "solver_knowledge_v1"; track: "hu" | "3max"; updated_at: string; runs: SolverRow[]};
export function parseSolverHistory(value: unknown, track: "hu" | "3max"): SolverHistory {
  if (!value || typeof value !== "object") throw new Error("Invalid solver history");
  const h = value as SolverHistory;
  if (h.engine !== "solver_knowledge_v1" || h.track !== track || !Array.isArray(h.runs) || h.runs.length > 720 || !Number.isFinite(Date.parse(h.updated_at))) throw new Error("Incompatible solver history");
  for (const r of h.runs) {
    if (!r || !Number.isFinite(Date.parse(r.at)) || r.student_accepted !== false ||
      ![r.visits, r.tables, r.labels, r.resolved_labels, r.failed_roots, r.nodes].every(n => Number.isSafeInteger(n) && n >= 0) ||
      !Number.isFinite(r.cpu_seconds) || r.cpu_seconds < 0 || r.resolved_labels > r.labels ||
      !Array.isArray(r.streets) || r.streets.length !== 4 || new Set(r.streets.map(s => s.street)).size !== 4 ||
      !r.streets.every(s => ["PREFLOP", "FLOP", "TURN", "RIVER"].includes(s.street) && Number.isFinite(s.response_shortfall_bb) && s.response_shortfall_bb >= -1e-9 && Number.isSafeInteger(s.iterations) && s.iterations > 0 && (s.river_response_gain_bb === null || Number.isFinite(s.river_response_gain_bb)))) throw new Error("Invalid solver measurement");
  }
  return h;
}
export async function fetchSolverHistory(track: "hu" | "3max", signal: AbortSignal): Promise<SolverHistory> {
  const base = policyStorageBase();
  if (!base) throw new Error("Solver progress requires the configured Supabase source.");
  const response = await fetch(`${base}/solver/${track}/history.json`, {cache: "no-store", signal});
  if (response.status === 404) throw new Error("Waiting for the first solver evaluation. Legacy NN results are not shown here.");
  if (!response.ok) throw new Error(`Solver history unavailable: HTTP ${response.status}`);
  const bytes = await response.arrayBuffer();
  if (bytes.byteLength > 2 * 1024 * 1024) throw new Error("Solver history exceeds size limit");
  return parseSolverHistory(JSON.parse(new TextDecoder().decode(bytes)), track);
}
