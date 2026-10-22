import test from "node:test";
import assert from "node:assert/strict";
import { rangeQuality } from "../ui/src/lib/poker/range-quality";

test("near-identical all-in ranges are withheld and differentiated output remains experimental", () => {
  const cells = Array.from({ length: 169 }, (_, i) => ({ label: String(i), combos: 4, probabilities: { CALL: .16, ALL_IN: .84 } }));
  assert.equal(rangeQuality(cells).usable, false);
  cells[0].probabilities = { CALL: .2, ALL_IN: .8 };
  assert.equal(rangeQuality(cells).usable, false);
  cells[0].probabilities = { CALL: .8, ALL_IN: .2 };
  assert.equal(rangeQuality(cells).usable, true);
  assert.throws(() => rangeQuality([]), /Complete/);
});
