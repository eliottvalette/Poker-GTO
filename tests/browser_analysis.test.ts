import assert from "node:assert/strict";
import test from "node:test";
import { analysisRoot, holeObservation, queryRange, withHeroCards } from "../ui/src/lib/poker/policy-analysis";
import { type LoadedAveragePolicy } from "../ui/src/lib/onnx-policy";

for (const count of [2, 3]) for (const street of ["PREFLOP", "FLOP", "TURN", "RIVER"] as const) {
  for (const position of count === 2 ? ["SB", "BB"] as const : ["BTN", "SB", "BB"] as const) {
    test(`legal analysis root ${count}/${street}/${position}`, () => {
      const hand = analysisRoot(count, position, street, [25, 25, 25], 1, 42);
      hand.assertInvariants();
      assert.equal(hand.street, street); assert.equal(hand.actor.position, position);
      assert.ok(hand.history.length >= 2); assert.equal(Object.keys(hand.players).length, count);
    });
  }
}
const uniform = { query: async () => ({ FOLD: 0.25, CALL: 0.75 }) } as unknown as LoadedAveragePolicy;
test("all 1326 preflop combos aggregate into 6/4/12-combo classes", async () => {
  const hand = analysisRoot(3, "BTN", "PREFLOP", [25, 25, 25], 1, 42);
  const cells = await queryRange(hand, uniform, () => false, () => {});
  assert.ok(cells); assert.equal(cells.reduce((n, cell) => n + cell.combos, 0), 1326);
  assert.equal(cells.find(cell => cell.label === "AA")?.combos, 6);
  assert.equal(cells.find(cell => cell.label === "AKs")?.combos, 4);
  assert.equal(cells.find(cell => cell.label === "AKo")?.combos, 12);
  assert.ok(cells.every(cell => cell.probabilities.CALL === 0.75));
});
test("range queries exclude board, retain public history and ignore simulated hidden cards", async () => {
  const hand = analysisRoot(2, "BB", "RIVER", [25, 25], 1, 42);
  const available = Array.from({length: 52}, (_, i) => i).filter(card => !hand.board.includes(card));
  const hole: [number, number] = [available[0], available[1]];
  const before = holeObservation(hand, hole);
  const changed = hand.clone();
  for (const player of Object.values(changed.players)) if (player.player_id !== changed.current_player) player.cards = [10, 11];
  changed.deck.reverse();
  assert.deepEqual(holeObservation(changed, hole), before);
  assert.throws(() => holeObservation(hand, [hand.board[0], available[0]]), /public board/);
  const cells = await queryRange(hand, uniform, () => false, () => {});
  assert.ok(cells); assert.equal(cells.reduce((n, cell) => n + cell.combos, 0), 1081);
});
test("cancelled range returns no partial result", async () => {
  assert.equal(await queryRange(analysisRoot(2, "SB", "PREFLOP", [25, 25], 1, 42), uniform, () => true, () => {}), null);
});

test("exact-card continuations preserve chosen hero cards and the full deck permutation", async () => {
  const hand = analysisRoot(3, "BTN", "PREFLOP", [25, 25, 25], 1, 42);
  const changed = withHeroCards(hand, [48, 49]);
  changed.assertInvariants();
  assert.deepEqual(changed.actor.cards, [48, 49]);
  assert.equal(new Set([...changed.deck, ...Object.values(changed.players).flatMap(p => p.cards)]).size, 52);
  assert.notDeepEqual(hand.actor.cards, changed.actor.cards);
});
