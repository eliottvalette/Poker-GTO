import test from "node:test";
import assert from "node:assert/strict";
import { gzipSync } from "node:zlib";
import { createHash } from "node:crypto";
import { evaluationLabel, fetchTrainingHistory, historyWindow, parseTrainingHistory } from "../ui/src/lib/training-history";

const estimate = { n: 2, mean: 0, m2: 0, se: 0, interval: [0, 0] };
const run = {
  suite: "hourly-paired-policy-v1", release_id: "a".repeat(64), model_sha256: "b".repeat(64), iteration: 10,
  evaluated_at: "2026-10-10T10:00:00Z", published_at: "2026-10-10T09:59:00Z", status: "complete",
  is_reference: false, reference: { iteration: 1, model_sha256: "c".repeat(64), suite: "hourly-paired-policy-v1" },
  completed_groups: 2, hands: 96, attempted_hands: 96, discarded_hands: 0, cpu_seconds: 1, wall_seconds: 2,
  gain: estimate, difference: estimate, segments: [], river: null, warnings: [], units: "BB/100",
};
const history = { version: 1, track: "hu", updated_at: "2026-10-10T10:00:00Z", runs: [run] };

test("training history validates probabilities-independent statistics and track provenance", () => {
  const parsed = parseTrainingHistory(history, "hu");
  assert.equal(parsed.runs[0].difference.mean, 0);
  assert.throws(() => parseTrainingHistory(history, "3max"), /Invalid/);
  for (const corrupt of [{ iteration: -1 }, { difference: { ...estimate, mean: NaN } }, { evaluated_at: "not a date" }, { river: {} }]) {
    assert.throws(() => parseTrainingHistory({ ...history, runs: [{ ...run, ...corrupt }] }, "hu"), /Invalid/);
  }
  assert.throws(() => parseTrainingHistory({ ...history, runs: [run, run] }, "hu"), /Invalid/);
  assert.notEqual(evaluationLabel(parsed.runs[0]), "Improved");
  assert.equal(evaluationLabel({ ...parsed.runs[0], status: "budget_limited" }), "Budget limited");
});

test("history filters by elapsed time rather than iteration, preserving missing results", () => {
  const parsed = parseTrainingHistory(history, "hu");
  assert.equal(historyWindow(parsed.runs, "24h", Date.parse("2026-10-11T11:00:00Z")).length, 0);
  assert.equal(historyWindow(parsed.runs, "7d", Date.parse("2026-10-11T11:00:00Z")).length, 1);
  assert.equal(historyWindow(parsed.runs, "30d", Date.parse("2026-10-01T11:00:00Z")).length, 0);
});

test("history fetch distinguishes not-yet-evaluated from HTTP failures", async () => {
  const fetchBefore = globalThis.fetch;
  const envBefore = { ...process.env };
  process.env.NEXT_PUBLIC_POLICY_SOURCE = "supabase";
  process.env.NEXT_PUBLIC_SUPABASE_URL = "https://example.supabase.co";
  process.env.NEXT_PUBLIC_SUPABASE_POLICY_BUCKET = "poker-policies";
  try {
    globalThis.fetch = async () => new Response("", { status: 404 });
    assert.equal(await fetchTrainingHistory("hu"), null);
    globalThis.fetch = async () => new Response("", { status: 503 });
    await assert.rejects(fetchTrainingHistory("hu"), /503/);
    const compressed = gzipSync(JSON.stringify(history));
    const index = { version: 1, track: "hu", sha256: createHash("sha256").update(compressed).digest("hex") };
    let downloads = 0;
    globalThis.fetch = async url => {
      if (String(url).endsWith("index.json")) return Response.json(index);
      downloads++; return new Response(new Uint8Array(compressed));
    };
    assert.equal((await fetchTrainingHistory("hu"))?.runs.length, 1);
    assert.equal((await fetchTrainingHistory("hu"))?.runs.length, 1);
    assert.equal(downloads, 1, "Unchanged history must not be downloaded again");
    globalThis.fetch = async url => String(url).endsWith("index.json")
      ? Response.json({ ...index, sha256: "d".repeat(64) }) : new Response(new Uint8Array(compressed));
    await assert.rejects(fetchTrainingHistory("hu"), /changed while loading/);
  } finally { globalThis.fetch = fetchBefore; process.env = envBefore; }
});
