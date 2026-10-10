import { policyStorageBase } from "./policy-catalog";

export type Estimate = { n: number; mean: number | null; m2: number; se: number | null; interval: [number, number] | null };
export type EvaluationRun = {
  suite: string; release_id: string; model_sha256: string; iteration: number;
  evaluated_at: string; published_at: string; status: "complete" | "budget_limited" | "failed";
  is_reference: boolean; reference: { iteration: number; model_sha256: string; suite: string };
  completed_groups: number; hands: number; attempted_hands: number; discarded_hands: number;
  cpu_seconds: number; wall_seconds: number; gain: Estimate; difference: Estimate;
  segments: { profile: string; region: string; gain: Estimate; difference: Estimate }[];
  river: { best_response_gain_bb: number; nodes: number } | null;
  warnings: string[]; units: "BB/100";
};
export type TrainingHistory = { version: 1; track: "hu" | "3max"; runs: EvaluationRun[]; updated_at: string };
export type Period = "24h" | "7d" | "30d";

function object(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
function finite(value: unknown): value is number { return typeof value === "number" && Number.isFinite(value); }
function estimate(value: unknown): value is Estimate {
  if (!object(value) || !Number.isInteger(value.n) || Number(value.n) < 0 || !finite(value.m2) || value.m2 < 0) return false;
  if (value.n === 0) return value.mean === null && value.se === null && value.interval === null;
  if (!finite(value.mean)) return false;
  if (value.n === 1) return value.se === null && value.interval === null;
  return finite(value.se) && value.se >= 0 && Array.isArray(value.interval) && value.interval.length === 2
    && value.interval.every(finite) && value.interval[0] <= value.mean && value.mean <= value.interval[1];
}
const hash = (value: unknown) => typeof value === "string" && /^[a-f0-9]{64}$/.test(value);
const date = (value: unknown) => typeof value === "string" && Number.isFinite(Date.parse(value));

export function parseTrainingHistory(value: unknown, track: "hu" | "3max"): TrainingHistory {
  if (!object(value) || value.version !== 1 || value.track !== track || !Array.isArray(value.runs)
      || value.runs.length > 720 || !date(value.updated_at)) throw new Error("Invalid training history");
  const ids = new Set();
  for (const r of value.runs) {
    if (!object(r) || !hash(r.release_id) || ids.has(r.release_id) || !hash(r.model_sha256)
        || typeof r.suite !== "string" || !Number.isInteger(r.iteration) || Number(r.iteration) < 1
        || !date(r.evaluated_at) || !date(r.published_at) || !["complete", "budget_limited", "failed"].includes(String(r.status))
        || typeof r.is_reference !== "boolean" || !object(r.reference) || !hash(r.reference.model_sha256)
        || !Number.isInteger(r.reference.iteration) || r.reference.suite !== r.suite
        || !estimate(r.gain) || !estimate(r.difference) || r.units !== "BB/100"
        || ![r.completed_groups, r.hands, r.attempted_hands, r.discarded_hands].every(v => Number.isInteger(v) && Number(v) >= 0)
        || ![r.cpu_seconds, r.wall_seconds].every(v => finite(v) && v >= 0)
        || !Array.isArray(r.warnings) || !r.warnings.every(v => typeof v === "string")
        || !Array.isArray(r.segments) || r.segments.length > 24
        || !r.segments.every(s => object(s) && typeof s.profile === "string" && typeof s.region === "string" && estimate(s.gain) && estimate(s.difference))
        || !(r.river === null || (object(r.river) && finite(r.river.best_response_gain_bb) && finite(r.river.nodes)))) {
      throw new Error("Invalid training evaluation record");
    }
    ids.add(r.release_id);
  }
  return value as TrainingHistory;
}

export function historyWindow(runs: EvaluationRun[], period: Period, now = Date.now()): EvaluationRun[] {
  const span = ({ "24h": 1, "7d": 7, "30d": 30 }[period]) * 86400000;
  return runs.filter(r => Date.parse(r.evaluated_at) >= now - span && Date.parse(r.evaluated_at) <= now)
    .sort((a, b) => Date.parse(a.evaluated_at) - Date.parse(b.evaluated_at));
}

export function evaluationLabel(run: EvaluationRun): string {
  if (run.status === "failed") return "Evaluation failed";
  if (run.status === "budget_limited") return "Budget limited";
  if (run.is_reference) return "Initial reference";
  return "Measured · not certified";
}

const cached = new Map<string, { sha256: string; history: TrainingHistory }>();

export async function fetchTrainingHistory(track: "hu" | "3max", signal?: AbortSignal): Promise<TrainingHistory | null> {
  const base = policyStorageBase();
  if (!base) throw new Error("Training history requires the Supabase policy source.");
  const path = `${base}/evaluation/${track}`;
  const response = await fetch(`${path}/index.json`, { cache: "no-store", signal });
  if (response.status === 404) return null;
  if (!response.ok) throw new Error(`Training history unavailable: HTTP ${response.status}`);
  const index = await response.json();
  if (!object(index) || index.version !== 1 || index.track !== track || !hash(index.sha256)) throw new Error("Invalid training history index");
  const previous = cached.get(path);
  if (previous && previous.sha256 === index.sha256) return previous.history;
  const archive = await fetch(`${path}/history.json.gz`, { cache: "no-store", signal });
  if (!archive.ok) throw new Error(`Training history unavailable: HTTP ${archive.status}`);
  const bytes = await archive.arrayBuffer();
  if (bytes.byteLength > 8 * 1024 * 1024) throw new Error("Training history exceeds size limit");
  const digest = Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)), b => b.toString(16).padStart(2, "0")).join("");
  if (digest !== index.sha256) throw new Error("Training history changed while loading. Refresh to retry.");
  const reader = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip")).getReader();
  const parts: Uint8Array<ArrayBuffer>[] = [];
  let size = 0;
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    size += value.byteLength;
    if (size > 8 * 1024 * 1024) { await reader.cancel(); throw new Error("Decompressed history exceeds size limit"); }
    parts.push(new Uint8Array(value));
  }
  const history = parseTrainingHistory(JSON.parse(await new Blob(parts).text()), track);
  cached.set(path, { sha256: digest, history });
  return history;
}
