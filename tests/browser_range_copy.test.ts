import assert from "node:assert/strict";
import test from "node:test";
import { formatRangeCopy } from "../ui/src/lib/poker/range-copy";
import type { ActionEvent } from "../ui/src/lib/poker/engine";

test("Range copy preserves public action amounts and all class probabilities without reading private cards", () => {
  const history: ActionEvent[] = [{street:"PREFLOP",player_id:1,position:"SB",action:"RAISE",
    amount_to:7.25,amount_added:6.75,pot_before:1.5,pot_after:8.25,highest_before:1}];
  const hand = {board:[0,5,10],history,get players(): never {throw new Error("Private state read");}};
  const text = formatRangeCopy(1,hand,[{cards:[48,49],probability:.75},{cards:[44,45],probability:.25}],"published");
  assert.match(text,/PREFLOP P1\(SB\) RAISE to=7.25/);
  assert.match(text,/AA:75%/);
  assert.match(text,/KK:25%/);
  assert.match(text,/AKs:0%/);
  assert.equal(text.match(/:[\d.]+%/g)?.length,169);
  assert.equal(text.split("\n").length,4);
  assert.match(text,/profile=published/);
  assert.match(formatRangeCopy(0,{board:[],history:[]},[],"uniform"),/Board: —\nHistory .*: —/);
});
