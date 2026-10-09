import { rangeDisplay, rangeColor } from "../ui/src/lib/poker/range-display";
import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { observerMarginals, handClassMasses, recoverHandStart, replayBeliefs, updateRanges, type BeliefTransition } from "../ui/src/lib/poker/beliefs";
import { behaviorProbabilities } from "../ui/src/lib/poker/behavior";
import { observe } from "../ui/src/lib/poker/observation";
import { restoreHand, analyzeHybrid, type Ranges, HYBRID_BUDGETS } from "../ui/src/lib/poker/hybrid";

for (const count of [2,3]) test(`Complete ${count}-player public posterior Python/browser parity`, () => {
  const fixture = JSON.parse(readFileSync(`tests/fixtures/phase2-hand-${count}.json`,"utf8")) as {
    ranges: Ranges; profiles: Record<number,string>;
    steps: (BeliefTransition & { posterior: Ranges; likelihoods: number[] })[];
  };
  let ranges = fixture.ranges;
  for (const row of fixture.steps) {
    const before = restoreHand(row.before), after = restoreHand(row.after);
    const probabilities = behaviorProbabilities(observe(before),fixture.profiles[before.current_player!]);
    probabilities.forEach((p,i) => assert.ok(Math.abs(p-row.likelihoods[i]) < 1e-12));
    ranges = updateRanges(before,after,ranges,row.action,fixture.profiles[before.current_player!],true);
    for (const [seat,expected] of Object.entries(row.posterior)) {
      assert.equal(ranges[Number(seat)].length,expected.length);
      for (const hand of expected) {
        const actual=ranges[Number(seat)].find(h => h.cards.join()===hand.cards.join());
        assert.ok(actual);
        assert.ok(Math.abs(actual.probability-hand.probability)<1e-12);
      }
    }
  }
  assert.ok(restoreHand(fixture.steps.at(-1)!.after).terminal);
  const recovered = recoverHandStart(restoreHand(fixture.steps.at(-1)!.after));
  assert.deepEqual(recovered.board, []);
  assert.deepEqual(recovered.players, restoreHand(fixture.steps[0].before).players);
  assert.deepEqual(recovered.deck, restoreHand(fixture.steps[0].before).deck);
  assert.ok(Math.abs(Object.values(handClassMasses(ranges[0])).reduce((a,b)=>a+b,0)-1)<1e-12);
  assert.deepEqual(replayBeliefs(fixture.ranges,fixture.steps,fixture.profiles,true),ranges);
  const first=fixture.steps[0], state=restoreHand(first.before);
  assert.throws(() => updateRanges(state,restoreHand(first.after),fixture.ranges,first.action,'loose_passive'),/interpolation/);
  assert.throws(() => behaviorProbabilities(observe(state),'unsupported'),/Unsupported/);
  const result=analyzeHybrid(state,fixture.ranges,HYBRID_BUDGETS.FAST,'response',fixture.profiles);
  assert.equal(result.samples,16);
  assert.match(result.warnings.join(),/loose_passive/);
  const insufficient = analyzeHybrid(state,fixture.ranges,{...HYBRID_BUDGETS.FAST,samples:1},'response',fixture.profiles);
  assert.match(insufficient.warnings.join(),/Insufficient samples/);
  assert.ok(Object.values(insufficient.standardErrors).every(se => se === null));
});

test("Observer marginals integrate joint blockers and ignore hidden simulator cards", async () => {
  for (const count of [2,3]) {
    const fixture = JSON.parse(readFileSync(`tests/fixtures/phase2-hand-${count}.json`,"utf8"));
    const state = restoreHand(fixture.steps[0].before), observer = 0;
    const known = state.players[observer].cards;
    const available = Array.from({length:52},(_,i)=>i).filter(c=>!known.includes(c));
    const ranges: Ranges = {[observer]:[{cards:known,probability:1}]};
    const opponents = Object.keys(state.players).map(Number).filter(p=>p!==observer);
    ranges[opponents[0]]=[{cards:[available[0],available[1]],probability:.6},{cards:[available[2],available[3]],probability:.4}];
    if(count===3) ranges[opponents[1]]=[{cards:[available[0],available[4]],probability:.8},{cards:[available[5],available[6]],probability:.2}];
    const result=observerMarginals(state,ranges,observer);
    // Independent exhaustive product over the two non-observer holdings.
    const totals=new Map<string,number>(); let z=0;
    for(const a of ranges[opponents[0]]) for(const b of count===3 ? ranges[opponents[1]] : [{cards:[] as number[],probability:1}]) {
      if(a.cards.some(c=>b.cards.includes(c))) continue;
      const w=a.probability*b.probability; z+=w;
      totals.set(a.cards.join(),(totals.get(a.cards.join())??0)+w);
    }
    result[opponents[0]].forEach(r=>assert.ok(Math.abs(r.probability-totals.get(r.cards.join())!/z)<1e-12));
    const altered=state.clone();
    opponents.forEach((p,i)=>{altered.players[p].cards=[available[10+i*2],available[11+i*2]];});
    altered.deck.reverse();
    assert.deepEqual(observerMarginals(altered,ranges,observer),result);
    for(const seat of opponents) {
      assert.ok(Math.abs(result[seat].reduce((s,r)=>s+r.probability,0)-1)<1e-12);
      assert.ok(result[seat].every(r=>!r.cards.some(c=>known.includes(c))));
    }
  }
});

test("Dense three-player marginals preserve uniform combo mass after Hero blockers", () => {
  const fixture=JSON.parse(readFileSync('tests/fixtures/phase2-hand-3.json','utf8'));
  const state=restoreHand(fixture.steps[0].before), hero=0;
  const rows=[];
  for(let a=0;a<52;a++) for(let b=a+1;b<52;b++) rows.push({cards:[a,b] as [number,number],probability:1/1326});
  const result=observerMarginals(state,{0:rows,1:rows,2:rows},hero);
  for(const seat of [1,2]) {
    assert.equal(result[seat].length,1225);
    assert.ok(result[seat].every(r=>Math.abs(r.probability-1/1225)<1e-12));
  }
});


test("Range display separates combo multiplicity, blocked classes and modeled reactions", () => {
  const fixture=JSON.parse(readFileSync('tests/fixtures/phase2-hand-3.json','utf8'));
  const state=restoreHand(fixture.steps[0].before), observer=state.current_player===0?1:0;
  const full=[];
  for(let a=0;a<52;a++) for(let b=a+1;b<52;b++) full.push({cards:[a,b] as [number,number],probability:1/1326});
  const marginals=observerMarginals(state,{0:full,1:full,2:full},observer);
  const profiles={0:"uniform",1:"uniform",2:"uniform"};
  const display=rangeDisplay(state,marginals,observer,profiles);
  const actor=state.current_player!;
  const masses=handClassMasses(marginals[actor]);
  for(const [label,mass] of Object.entries(masses)) assert.ok(Math.abs(mass/display.uniformMass[label]-1)<1e-10);
  assert.equal(rangeColor(0, .03), rangeColor(.0001, .03));
  assert.notEqual(rangeColor(0, .03), rangeColor(.03, .03));
  for(const mix of Object.values(display.reactions[actor])) assert.ok(Math.abs(mix.fold+mix.passive+mix.aggressive-1)<1e-12);
  assert.equal(Object.keys(display.reactions).length,1);
  const alternate=state.clone();
  Object.keys(state.players).map(Number).filter(p=>p!==observer).forEach(p=>{alternate.players[p].cards=[0,1];});
  alternate.deck.reverse();
  assert.deepEqual(rangeDisplay(alternate,marginals,observer,profiles),display);
  const terminal=restoreHand(fixture.steps.at(-1).after);
  const endRanges=observerMarginals(terminal,fixture.steps.at(-1).posterior,observer);
  const end=rangeDisplay(terminal,endRanges,observer,profiles);
  assert.deepEqual(end.reactions,{});
  for(const composition of Object.values(end.composition)) {
    assert.ok(Math.abs(Object.values(composition).reduce((a,b)=>a+b,0)-1)<1e-12);
    assert.equal(composition["Draw (unpaired)"],0);
  }
});

test("externally predicted candidate likelihoods update the same Bayesian factors", () => {
  const fixture=JSON.parse(readFileSync('tests/fixtures/phase2-hand-2.json','utf8'));
  const first=fixture.steps[0], before=restoreHand(first.before), after=restoreHand(first.after);
  const rows=fixture.ranges[before.current_player!];
  const predictions:Record<string,number[]>={};
  for(const row of rows) {
    const view=before.clone(); view.actor.cards=row.cards;
    predictions[row.cards.join()]=behaviorProbabilities(observe(view),'loose_passive');
  }
  assert.deepEqual(updateRanges(before,after,fixture.ranges,first.action,'published',true,predictions),
    updateRanges(before,after,fixture.ranges,first.action,'loose_passive',true));
  assert.throws(()=>updateRanges(before,after,fixture.ranges,first.action,'published',true,{}),/Missing candidate/);
});
