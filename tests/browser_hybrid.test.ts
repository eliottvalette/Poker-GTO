import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { analyzeHybrid, restoreHand, uniformRanges, HYBRID_BUDGETS, LIVE_DECISION_BUDGET, type HandTransport, type HybridBudget, type Ranges } from "../ui/src/lib/poker/hybrid";
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

for (const count of [2, 3]) test(`Model-free live decisions cover all streets for ${count} players`, () => {
  const hand = HandState.start(Object.fromEntries(Array.from({length: count}, (_, i) => [i, 25])), 0, new SeededRNG(81));
  const profiles = Object.fromEntries(Object.keys(hand.players).map(p => [p, "conservative"]));
  for (const street of ["PREFLOP", "FLOP", "TURN", "RIVER"]) {
    while (hand.street !== street) hand.act(hand.toCall() ? "CALL" : "CHECK");
    const result = analyzeHybrid(hand, uniformRanges(hand), LIVE_DECISION_BUDGET, "behavior", profiles);
    assert.equal(result.samples, 128);
    assert.equal(result.iterations, 0);
    assert.ok(result.nodes <= LIVE_DECISION_BUDGET.maxNodes);
    assert.equal(Object.keys(result.ev).length, result.legalActions.length);
    for (const value of Object.values(result.ev)) assert.ok(Number.isFinite(value));
    const winner = Math.max(...Object.values(result.ev));
    for (const [action, probability] of Object.entries(result.probabilities)) {
      if (probability > 0) assert.ok(Math.abs(result.ev[action] - winner) < 1e-10);
    }
    const other = hand.clone();
    for (const seat of Object.keys(other.players).map(Number)) if (seat !== other.current_player) other.players[seat].cards = [50, 51];
    other.deck.reverse();
    assert.deepEqual(analyzeHybrid(other, uniformRanges(other), LIVE_DECISION_BUDGET, "behavior", profiles), result);
  }
});
