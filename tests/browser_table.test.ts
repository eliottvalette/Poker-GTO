import assert from "node:assert/strict";
import test from "node:test";
import { BrowserTable, cardLabel } from "../ui/src/lib/game";
import { SeededRNG, TournamentState } from "../ui/src/lib/poker/engine";

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
      assert.ok(Math.abs(view.players.reduce((s, p) => s + p.stack_bb, view.pot_bb) - 75) < 1e-8);
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
    assert.ok(final.players.some(p => p.player_id === final.winner && Math.abs(p.stack_bb - 75) < 1e-8));
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
