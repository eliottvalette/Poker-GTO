import assert from "node:assert/strict";
import test from "node:test";
import { BrowserTable, cardLabel } from "../ui/src/lib/game";
import { observe, NUMERIC_NAMES } from "../ui/src/lib/poker/observation";
import { SeededRNG, TournamentState, BlindSchedule, BlindStage } from "../ui/src/lib/poker/engine";

function passiveTable(stacks: Record<number, number> = { 0: 25, 1: 25, 2: 25 }): BrowserTable {
  const tournament = new TournamentState(stacks, 0, new SeededRNG(11));
  tournament.startHand();
  return new BrowserTable(tournament, 0, new SeededRNG(12));
}

test("browser tables reproduce deals and bot actions with the same seed", () => {
  const a = BrowserTable.create(741, 0), b = BrowserTable.create(741, 0);
  assert.deepEqual(a.view(), b.view());
  for (let decision = 0; decision < 12; decision++) {
    const view = a.view();
    if (view.tournament_terminal) break;
    if (view.hand_terminal) { a.nextHand(); b.nextHand(); }
    else {
      const action = view.legal_actions.find(action => action.action_id === "CALL") ?? view.legal_actions[0];
      assert.ok(action);
      a.act(action.action_id); b.act(action.action_id);
    }
    assert.deepEqual(a.view(), b.view());
  }
});

test("browser table clones and exported views cannot mutate their source", () => {
  const original = BrowserTable.create(19, 0), baseline = original.view();
  const clone = original.clone();
  const view = clone.view();
  const action = view.legal_actions.find(action => action.action_id === "ALL_IN") ?? view.legal_actions[0];
  assert.ok(action);
  clone.act(action.action_id);
  assert.deepEqual(original.view(), baseline);
  const detached = original.view();
  detached.board.push(51);
  detached.players[0].stack_bb = -100;
  detached.players[0].cards[0] = 51;
  detached.history[0].pot_after = -100;
  detached.active_players.length = 0;
  detached.blinds.big = 999;
  assert.deepEqual(original.view(), baseline);
  original.act(action.action_id);
  assert.deepEqual(original.view(), clone.view(), "Cloned bot RNG must reproduce the same continuation");
});

test("browser view hides opponent cards until an actual showdown", () => {
  const table = passiveTable();
  let view = table.view();
  assert.equal(view.players[0].cards.length, 2);
  assert.deepEqual(view.players.slice(1).map(p => p.cards), [[], []]);
  const hand = table.tournament.hand!;
  while (!hand.terminal) hand.act(hand.toCall() ? "CALL" : "CHECK");
  assert.equal(hand.showdown, true);
  view = table.view();
  assert.ok(view.players.every(p => p.cards.length === 2));
  assert.ok(Math.abs(Object.values(view.hand_results_bb).reduce((a, b) => a + b, 0)) < 1e-8);
});

test("winning by folds does not reveal opponent cards", () => {
  const table = passiveTable();
  table.tournament.hand!.act("FOLD");
  table.tournament.hand!.act("FOLD");
  assert.equal(table.tournament.hand!.terminal, true);
  assert.equal(table.tournament.hand!.showdown, false);
  assert.deepEqual(table.view().players.slice(1).map(p => p.cards), [[], []]);
});

test("browser P&L counts settled results rather than uncalled chips committed live", () => {
  const table = passiveTable();
  table.tournament.hand!.act("RAISE", 2);
  assert.equal(table.view().hero_result_bb, 0);
  while (!table.tournament.hand!.terminal) table.tournament.hand!.act(table.tournament.hand!.toCall() ? "CALL" : "CHECK");
  assert.equal(table.view().hero_result_bb, table.tournament.hand!.utility(0));
});

for (const seed of [1, 7, 42]) {
  test(`bounded browser tournament ${seed} conserves chips through elimination`, () => {
    const table = BrowserTable.create(seed, 0);
    let previousActive = 3;
    let decisions = 0;
    while (!table.view().tournament_terminal && decisions < 500) {
      const view = table.view();
      assert.ok(Math.abs(view.players.reduce((s, p) => s + p.stack_bb, view.pot_bb) - view.total_chips_bb) < 1e-8);
      assert.ok(view.active_players.length <= previousActive);
      previousActive = view.active_players.length;
      assert.equal(view.policy.status, "unavailable");
      assert.equal(view.policy.probabilities, null);
      if (view.hand_terminal) table.nextHand();
      else {
        assert.equal(view.actor, 0);
        const action = view.legal_actions.find(a => a.action_id === "ALL_IN")
          ?? view.legal_actions.find(a => a.action_id === "CALL") ?? view.legal_actions[0];
        assert.ok(action);
        table.act(action.action_id);
      }
      table.tournament.assertInvariants();
      decisions++;
    }
    const final = table.view();
    assert.equal(final.tournament_terminal, true, `Tournament exceeded 500 decisions for seed ${seed}`);
    assert.equal(final.active_players.length, 1);
    assert.ok(final.players.some(p => p.player_id === final.winner && Math.abs(p.stack_bb * final.chip_unit_big_blind - 75) < 1e-8));
    assert.throws(() => table.nextHand(), /already won/);
  });
}

test("browser table surfaces invalid identities, actions and card IDs", () => {
  assert.throws(() => BrowserTable.create(1, 9), /not seated/);
  const table = BrowserTable.create(1, 0);
  assert.throws(() => table.act("RAISE"), /Illegal canonical action/);
  assert.throws(() => table.nextHand(), /before settlement/);
  assert.equal(cardLabel(0), "2♠");
  assert.equal(cardLabel(51), "A♣");
  assert.throws(() => cardLabel(52), /Invalid card/);
});


test("browser schedule levels apply between hands and preserve physical chips", () => {
  const schedule = new BlindSchedule([
    new BlindStage(1, { small: 0.5, big: 1 }),
    new BlindStage(2, { small: 1, big: 2 }),
    new BlindStage(3, { small: 2, big: 4 }),
  ]);
  const t = new TournamentState({ 0: 25, 1: 25, 2: 25 }, 0, new SeededRNG(9), schedule);
  for (let number = 1; number <= 4; number++) {
    const h = t.startHand();
    const expectedBig = number === 1 ? 1 : number === 2 ? 2 : 4;
    assert.equal(h.blinds.big, expectedBig);
    assert.equal(h.hand_number, number);
    assert.equal(h.blind_level_index, Math.min(number - 1, 2));
    const before = h.blinds.big;
    h.act("FOLD"); h.act("FOLD");
    assert.equal(h.blinds.big, before, "The settled hand retains its own blind unit");
    t.assertInvariants();
    assert.equal(t.total_chips, 75);
  }
});

test("browser public amounts use current BB while historical results keep each hand's unit", () => {
  const schedule = new BlindSchedule([
    new BlindStage(1, { small: 0.5, big: 1 }), new BlindStage(2, { small: 1, big: 2 }),
  ]);
  const t = new TournamentState({ 0: 25, 1: 25, 2: 25 }, 0, new SeededRNG(9), schedule);
  t.startHand();
  const table = new BrowserTable(t, 1, new SeededRNG(4));
  t.hand!.act("FOLD"); t.hand!.act("FOLD");
  assert.equal(table.view().hero_result_bb, -0.5);
  t.startHand();
  const view = table.view();
  assert.equal(view.chip_unit_big_blind, 2);
  assert.equal(view.total_chips_bb, 37.5);
  assert.equal(view.pot_bb, 1.5);
  assert.equal(view.to_call_bb, 1);
  assert.equal(view.hero_result_bb, -0.5, "Past P&L does not rescale at a level change");
  assert.deepEqual(view.blinds, { small: 0.5, big: 1 });
  assert.equal(view.history[0].amount_to, 0.5);
  assert.equal(view.history[1].amount_to, 1);
  const raise = view.legal_actions.find(a => a.action_id === "RAISE_2.0X");
  assert.equal(raise?.amount_to, 2);
  const internal = t.hand!;
  assert.equal(internal.blinds.big, 2);
  assert.equal(internal.pot, 3);
  internal.act("FOLD"); internal.act("FOLD");
  const final = table.view();
  assert.equal(final.hand_results_bb[2], -0.5);
  assert.equal(final.hand_results_bb[0], 0.5);
});

test("conditional cEV observations contain only current hand at the current blind level", () => {
  const schedule = new BlindSchedule([
    new BlindStage(1, { small: 0.5, big: 1 }), new BlindStage(2, { small: 1, big: 2 }),
  ]);
  const t = new TournamentState({ 0: 25, 1: 25, 2: 25 }, 0, new SeededRNG(17), schedule);
  t.startHand(); t.hand!.act("FOLD"); t.hand!.act("FOLD");
  const hand = t.startHand();
  const obs = observe(t);
  assert.deepEqual(obs, observe(hand));
  assert.equal(obs.version, 3);
  assert.equal(obs.objective, "hand_chip_delta");
  assert.equal(JSON.parse(obs.recall).length, 1);
  assert.equal(JSON.parse(obs.recall)[0].hand, 2);
  assert.ok(obs.history.every(token => token[0] === 1 / 25));
  assert.equal(obs.numeric[NUMERIC_NAMES.indexOf("blind_level_index")], 1);
  assert.equal(obs.numeric[NUMERIC_NAMES.indexOf("chip_unit_big_blind")], 2);
  assert.equal(obs.numeric[NUMERIC_NAMES.indexOf("pot")], 1.5 / 25);
  assert.equal(obs.numeric[NUMERIC_NAMES.indexOf("hand_number")], 2 / 25);
});

test("browser blind schedule rejects invalid stages instead of repairing them", () => {
  assert.throws(() => new BlindStage(0, { small: 0.5, big: 1 }));
  assert.throws(() => new BlindSchedule([]));
  assert.throws(() => new BlindSchedule([new BlindStage(2, { small: 0.5, big: 1 })]));
  assert.throws(() => new BlindSchedule([new BlindStage(1, { small: 0.5, big: 1 }),
    new BlindStage(1, { small: 1, big: 2 })]));
  assert.throws(() => new BlindSchedule([new BlindStage(1, { small: 1, big: 2 }),
    new BlindStage(2, { small: 0.5, big: 1 })]));
  assert.throws(() => new BlindSchedule([new BlindStage(1, { small: 0.5, big: 1 }),
    new BlindStage(2, { small: 0.5, big: 1 })]));
  assert.throws(() => BlindSchedule.fixed().forHand(0));
  assert.throws(() => new TournamentState({ 0: 25, 1: 25 }, 0, new SeededRNG(0), BlindSchedule.fixed(), "split"));
});

test("explicit asynchronous opponent policies drive actions and survive cloning", async () => {
  const { ACTION_IDS } = await import("../ui/src/lib/poker/actions.js");
  const seen:number[]=[];
  const policy=async (observation:ReturnType<typeof observe>) => {
    seen.push(observation.hero);
    assert.equal('deck' in observation,false);
    const selected=observation.legal_mask[ACTION_IDS.indexOf('FOLD')] ? ACTION_IDS.indexOf('FOLD') : ACTION_IDS.indexOf('CHECK');
    assert.ok(observation.legal_mask[selected]);
    return ACTION_IDS.map((_,i)=>Number(i===selected));
  };
  const table=await BrowserTable.createWithPolicy(12,2,'published',policy);
  assert.ok(seen.length>0);
  assert.ok(seen.every(p=>p!==2));
  assert.equal(table.opponentProfile,'published');
  assert.equal(table.clone().opponentProfile,'published');
  assert.ok(table.tournament.hand!.history.filter(e=>e.action!=='BLIND').every(e=>e.action==='FOLD'||e.action==='CHECK'));
  await assert.rejects(()=>BrowserTable.createWithPolicy(12,2,'published',async ()=>Array(13).fill(0)),/Invalid opponent/);
});
