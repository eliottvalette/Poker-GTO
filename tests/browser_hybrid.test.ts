import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { analyzeHybrid, restoreHand, uniformRanges, HYBRID_BUDGETS, type HandTransport, type HybridBudget, type Ranges } from "../ui/src/lib/poker/hybrid";
import { HandState, SeededRNG } from "../ui/src/lib/poker/engine";

const fixtures = JSON.parse(readFileSync("tests/fixtures/hybrid_parity.json", "utf8")) as {
  count: number; street: string; searchMode: "behavior" | "public_cfr"; state: HandTransport; ranges: Ranges; budget: HybridBudget;
  expected: { probabilities: Record<string, number>; ev: Record<string, number>; nodes: number; iterations: number };
}[];
for (const fixture of fixtures) test(`Python/browser genuine CFR parity for ${fixture.count} players/${fixture.street}`, () => {
  const result = analyzeHybrid(restoreHand(fixture.state), fixture.ranges, fixture.budget, fixture.searchMode);
  for (const action of Object.keys(fixture.expected.ev)) {
    assert.ok(Math.abs(result.ev[action] - fixture.expected.ev[action]) < 1e-10, `EV ${action}`);
    assert.ok(Math.abs(result.probabilities[action] - fixture.expected.probabilities[action]) < 1e-10, `strategy ${action}`);
  }
  assert.equal(result.nodes, fixture.expected.nodes);
  assert.equal(result.iterations, fixture.expected.iterations);
});
for (const count of [2, 3]) test(`Live ${count}-player preflop calculation is deterministic and ignores future deck`, () => {
  const hand = HandState.start(Object.fromEntries(Array.from({length: count}, (_, i) => [i, 1.5])), 0, new SeededRNG(8));
  const ranges = uniformRanges(hand);
  const a = analyzeHybrid(hand, ranges, HYBRID_BUDGETS.FAST);
  hand.deck.reverse();
  const b = analyzeHybrid(hand, ranges, HYBRID_BUDGETS.FAST);
  assert.deepEqual(a, b);
  assert.equal(a.samples, 16);
  assert.ok(a.nodes > 16);
  assert.ok(Math.abs(Object.values(a.probabilities).reduce((x, y) => x + y, 0) - 1) < 1e-12);
  assert.throws(() => analyzeHybrid(hand, ranges, { ...HYBRID_BUDGETS.FAST, maxNodes: 1 }), /budget exhausted/);
});
