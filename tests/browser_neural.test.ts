import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { neuralObservation, numericNames } from "../ui/src/lib/poker/neural";
import { NUMERIC_NAMES, type Observation } from "../ui/src/lib/poker/observation";

const fixtures = JSON.parse(readFileSync("tests/fixtures/neural_parity.json", "utf8")) as {
  name: string; observation: Observation; neural: ReturnType<typeof neuralObservation>; neural_v3: ReturnType<typeof neuralObservation>; neural_v4: ReturnType<typeof neuralObservation>
}[];
function almostEqual(actual: number[], expected: number[]) {
  assert.equal(actual.length, expected.length);
  actual.forEach((value, i) => assert.ok(Math.abs(value - expected[i]) < 1e-12, `${i}: ${value} != ${expected[i]}`));
}
for (const fixture of fixtures) {
  test(`Python/browser neural feature parity: ${fixture.name}`, () => {
    const actual = neuralObservation(fixture.observation, 2);
    assert.equal(actual.numeric.length, numericNames(2).length);
    assert.deepEqual(actual.cards, fixture.neural.cards);
    almostEqual(actual.numeric, fixture.neural.numeric);
    actual.history.forEach((row, i) => almostEqual(row, fixture.neural.history[i]));
  });
  test(`Python/browser feature-v4 parity and card invariance: ${fixture.name}`, () => {
    const actual = neuralObservation(fixture.observation, 4);
    assert.equal(actual.numeric.length, numericNames(4).length);
    assert.deepEqual(actual.cards, fixture.neural_v4.cards);
    almostEqual(actual.numeric, fixture.neural_v4.numeric);
    actual.history.forEach((row, i) => almostEqual(row, fixture.neural_v4.history[i]));
    for (const permutation of [[2, 3, 0, 1], [3, 1, 0, 2]]) {
      const changed = structuredClone(fixture.observation);
      const count = Math.round(changed.numeric[NUMERIC_NAMES.indexOf("player_count")] * 3);
      const convert = (card: number) => card === 52 ? 52 : Math.floor(card / 4) * 4 + permutation[card % 4];
      changed.cards = changed.cards.map(convert);
      changed.history.forEach(row => { if (row[11] === 1) row[10] = convert(Math.round(row[10] * 51)) / 51; });
      [changed.cards[0], changed.cards[1]] = [changed.cards[1], changed.cards[0]];
      [changed.history[count], changed.history[count + 1]] = [changed.history[count + 1], changed.history[count]];
      assert.deepEqual(neuralObservation(changed, 4), actual);
    }
  });
  test(`Python/browser feature-v3 parity and private-card invariance: ${fixture.name}`, () => {
    const actual = neuralObservation(fixture.observation, 3);
    assert.deepEqual(actual.cards, fixture.neural_v3.cards);
    almostEqual(actual.numeric, fixture.neural_v3.numeric);
    actual.history.forEach((row, i) => almostEqual(row, fixture.neural_v3.history[i]));
    const changed = structuredClone(fixture.observation);
    const count = Math.round(changed.numeric[NUMERIC_NAMES.indexOf("player_count")] * 3);
    [changed.cards[0], changed.cards[1]] = [changed.cards[1], changed.cards[0]];
    [changed.history[count], changed.history[count + 1]] = [changed.history[count + 1], changed.history[count]];
    assert.deepEqual(neuralObservation(changed, 3), actual);
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


test("private descriptors cover pairs, suited cards, broadway and ace-low connection", () => {
  const cases: [number[], number[]][] = [
    [[48, 49], [1, 1, 0, 1, 0, 0, 1]],
    [[48, 44], [1, 11 / 12, 1 / 12, 0, 1, 1, 1]],
    [[20, 1], [5 / 12, 0, 5 / 12, 0, 0, 0, 0]],
    [[48, 0], [1, 0, 1, 0, 1, 1, .5]],
    [[48, 5], [1, 1 / 12, 11 / 12, 0, 0, 0, .5]],
  ];
  for (const [cards, expected] of cases) {
    const obs = structuredClone(fixtures[0].observation);
    const count = Math.round(obs.numeric[NUMERIC_NAMES.indexOf("player_count")] * 3);
    obs.cards.splice(0, 2, ...cards);
    cards.forEach((card, index) => { obs.history[count + index][10] = card / 51; });
    almostEqual(neuralObservation(obs, 4).numeric.slice(-7), expected);
  }
});
