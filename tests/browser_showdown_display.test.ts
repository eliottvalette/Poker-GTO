import assert from "node:assert/strict";
import test from "node:test";
import { bestFive, decisiveCards, rank7 } from "../ui/src/lib/poker/evaluator";
import { showdownHighlights } from "../ui/src/lib/poker/showdown-display";
import { HandState, SeededRNG } from "../ui/src/lib/poker/engine";
const card=(rank:number,suit=0)=>(rank-2)*4+suit;

test("Pair and decisive kicker are emphasized; exactly five cards are selected",()=>{
  const board=[card(13),card(13,1),card(7,2),card(5,3),card(2)];
  const winner=bestFive([...board,card(14),card(12)]), loser=bestFive([...board,card(11),card(10)]);
  assert.equal(winner.category,1);
  assert.equal(winner.score,rank7([...board,card(14),card(12)]));
  assert.equal(winner.cards.length,5);
  assert.deepEqual(new Set(decisiveCards(winner,[loser])),new Set([card(13),card(13,1),card(14)]));
  assert.ok(!winner.cards.includes(card(5,3))&&!winner.cards.includes(card(2)));
});

test("Second kicker is decisive when the first is shared",()=>{
  const board=[card(13),card(13,1),card(14),card(5,3),card(2)];
  const winner=bestFive([...board,card(12,2),card(7,1)]), loser=bestFive([...board,card(11,2),card(10,3)]);
  assert.deepEqual(new Set(decisiveCards(winner,[loser])),new Set([card(13),card(13,1),card(12,2)]));
});

test("Board-only ties prefer the board; wheel uses ace as low",()=>{
  const board=[10,11,12,13,14].map(rank=>card(rank));
  const royal=bestFive([...board,card(2,1),card(3,2)]);
  assert.deepEqual(royal.cards,board);assert.equal(royal.category,8);
  const wheel=bestFive([card(14),card(2,1),card(3,2),card(4,3),card(5),card(9),card(11)]);
  assert.equal(wheel.category,4);assert.equal(wheel.ranks[0],5);
  assert.equal(decisiveCards(wheel,[]).length,5);
});

test("Contested side-pot winners are marked; folds and uncalled returns are excluded",()=>{
  const hand=HandState.start({0:10,1:20,2:30},0,new SeededRNG(1));
  hand.terminal=true;hand.showdown=true;
  hand.board=[card(2),card(4,1),card(7,2),card(9,3),card(11)];
  hand.players[0].cards=[card(14),card(14,1)];
  hand.players[1].cards=[card(13),card(13,1)];
  hand.players[2].cards=[card(12),card(12,1)];
  for(const id of [0,1,2]) hand.players[id].contribution=(id+1)*10;
  assert.deepEqual(Object.keys(showdownHighlights(hand)),['0','1']);
  hand.players[0].folded=true;
  assert.deepEqual(Object.keys(showdownHighlights(hand)),['1']);
  hand.showdown=false;
  assert.deepEqual(showdownHighlights(hand),{});
  hand.terminal=false;hand.showdown=true;
  assert.deepEqual(showdownHighlights(hand),{});
});
