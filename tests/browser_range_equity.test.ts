import assert from "node:assert/strict";
import test from "node:test";
import { HandState, SeededRNG } from "../ui/src/lib/poker/engine";
import { rangeEquities } from "../ui/src/lib/poker/range-equity";
import { uniformRanges } from "../ui/src/lib/poker/hybrid";

test("Known river winner and ties use exact ranking, without reading hidden cards",()=>{
  const hand=HandState.start({0:25,1:25},0,new SeededRNG(1));
  hand.players[0].cards=[48,49]; hand.board=[0,5,10,15,20];
  const ranges={0:[{cards:[48,49] as [number,number],probability:1}],1:[{cards:[44,45] as [number,number],probability:1}]};
  assert.equal(rangeEquities(hand,ranges,0,32)[0].probability,1);
  hand.players[1].cards=[48,49]; hand.deck=[];
  assert.equal(rangeEquities(hand,ranges,0,32)[0].probability,1);
  hand.board=[32,36,40,44,48];hand.players[0].cards=[0,1];
  const ties=rangeEquities(hand,{0:[{cards:[0,1],probability:1}],1:[{cards:[4,5],probability:1}]},0,32);
  assert.equal(ties[0].probability,.5);assert.equal(ties[1].probability,.5);
});

test("Three-player equity uses compatible joint probabilities and retains folded blockers",()=>{
  const hand=HandState.start({0:25,1:25,2:25},0,new SeededRNG(2));
  hand.players[0].cards=[0,1];hand.board=[4,13,22,31,36];
  const ranges={0:[{cards:[0,1] as [number,number],probability:1}],
    1:[{cards:[48,49] as [number,number],probability:.5},{cards:[44,45] as [number,number],probability:.5}],
    2:[{cards:[48,50] as [number,number],probability:.5},{cards:[40,41] as [number,number],probability:.5}]};
  const result=rangeEquities(hand,ranges,0,2048);
  assert.ok(Math.abs(result[1].probability-2/3)<4*result[1].standardError);
  assert.equal(result[0].probability,0);
  assert.ok(Math.abs(Object.values(result).reduce((s,r)=>s+r.probability,0)-1)<1e-12);
  assert.deepEqual(result,rangeEquities(hand,ranges,0,2048));
  hand.players[2].folded=true;
  const folded=rangeEquities(hand,ranges,0,32);
  assert.equal(folded[2].probability,0);assert.equal(folded[1].probability,1);
});

test("Broad preflop equity is bounded and conserves win/tie mass",()=>{
  const hand=HandState.start({0:25,1:25,2:25},0,new SeededRNG(3));
  const result=rangeEquities(hand,uniformRanges(hand),0);
  assert.ok(Math.abs(Object.values(result).reduce((s,r)=>s+r.probability,0)-1)<1e-12);
  assert.ok(Object.values(result).every(r=>r.samples===1024&&Number.isFinite(r.standardError)));
});
