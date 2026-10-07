import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { neuralObservation, NEURAL_NUMERIC_NAMES } from "../ui/src/lib/poker/neural";
import { NUMERIC_NAMES, type Observation } from "../ui/src/lib/poker/observation";

const fixtures = JSON.parse(readFileSync("tests/fixtures/neural_parity.json", "utf8")) as {
  name: string; observation: Observation; neural: ReturnType<typeof neuralObservation>
}[];
function almostEqual(actual: number[], expected: number[]) {
  assert.equal(actual.length, expected.length);
  actual.forEach((value, i) => assert.ok(Math.abs(value - expected[i]) < 1e-12, `${i}: ${value} != ${expected[i]}`));
}
for (const fixture of fixtures) {
  test(`Python/browser neural feature parity: ${fixture.name}`, () => {
    const actual = neuralObservation(fixture.observation);
    assert.equal(actual.numeric.length, NEURAL_NUMERIC_NAMES.length);
    assert.deepEqual(actual.cards, fixture.neural.cards);
    almostEqual(actual.numeric, fixture.neural.numeric);
    actual.history.forEach((row, i) => almostEqual(row, fixture.neural.history[i]));
  });
}
test("global suit permutation preserves browser neural cards/history/features", () => {
  const fixture = fixtures[3];
  const changed = structuredClone(fixture.observation);
  const permutation = [2, 3, 0, 1];
  const convert = (card: number) => card === 52 ? 52 : Math.floor(card / 4) * 4 + permutation[card % 4];
  changed.cards = changed.cards.map(convert);
  changed.history = changed.history.map(row => {
    if (row[11] === 1) row[10] = convert(Math.round(row[10] * 51)) / 51;
    return row;
  });
  assert.deepEqual(neuralObservation(changed), neuralObservation(fixture.observation));
});

test("arbitrary opponent indexing and stack-token order preserve neural inputs", () => {
  for (const fixture of fixtures) {
    const changed = structuredClone(fixture.observation);
    const count = Math.round(changed.numeric[NUMERIC_NAMES.indexOf("player_count")] * 3);
    changed.hero += 10;
    if (count === 3) {
      for (const feature of ["stack", "street_bet", "contribution", "folded", "effective", "initial"]) {
        const first = NUMERIC_NAMES.indexOf(`${feature}_1`);
        const second = NUMERIC_NAMES.indexOf(`${feature}_2`);
        [changed.numeric[first], changed.numeric[second]] = [changed.numeric[second], changed.numeric[first]];
      }
      const button = NUMERIC_NAMES.indexOf("button");
      const remap = (value: number) => value === 0.5 ? 1 : value === 1 ? 0.5 : value;
      changed.numeric[button] = remap(changed.numeric[button]);
      for (const row of changed.history) row[2] = remap(row[2]);
    }
    changed.history.splice(0, count, ...changed.history.slice(1, count), changed.history[0]);
    assert.deepEqual(neuralObservation(changed), neuralObservation(fixture.observation), fixture.name);
  }
});
