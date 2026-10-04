import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import test from "node:test";
import { HandState, TournamentState, SeededRNG, BlindSchedule, BlindStage } from "../ui/src/lib/poker/engine";
import { rank7 } from "../ui/src/lib/poker/evaluator";
import { ACTION_IDS, legalActions } from "../ui/src/lib/poker/actions";
import { NUMERIC_NAMES, observe } from "../ui/src/lib/poker/observation";

type Command = { category: string; amount_to: number | null };
type Snapshot = Record<string, unknown>;
type HandFixture = {
  name: string; stacks: number[]; button: number; deck: number[];
  blinds: {small: number; big: number}; hand_number: number; blind_level_index: number; initial: Snapshot;
  steps: { action: Command; expected: Snapshot }[];
};
type TournamentFixture = {
  name: string; stacks: number[]; button: number;
  schedule: {first_hand: number; blinds: {small: number; big: number}}[];
  steps: { operation: "start" | "action"; deck?: number[]; action?: Command; expected: Snapshot }[];
};
type Fixtures = {
  version: number; actions: string[]; numeric_names: string[];
  hands: HandFixture[]; tournaments: TournamentFixture[];
  category_cards: number[][]; evaluations: { a: number[]; b: number[]; comparison: number }[];
};

const fixtures: Fixtures = JSON.parse(readFileSync(resolve("tests/fixtures/browser_parity.json"), "utf8"));

/** Compare floating arithmetic without weakening structural or ordering contracts. */
function equivalent(actual: unknown, expected: unknown, path = "state"): void {
  if (typeof expected === "number") {
    assert.equal(typeof actual, "number", `${path}: numeric type`);
    assert.ok(Number.isFinite(actual as number), `${path}: finite value`);
    assert.ok(Math.abs((actual as number) - expected) <= 1e-8,
      `${path}: actual=${actual}, expected=${expected}`);
  } else if (Array.isArray(expected)) {
    assert.ok(Array.isArray(actual), `${path}: expected array`);
    assert.equal((actual as unknown[]).length, expected.length, `${path}: length`);
    expected.forEach((value, index) => equivalent((actual as unknown[])[index], value, `${path}[${index}]`));
  } else if (expected !== null && typeof expected === "object") {
    assert.ok(actual !== null && typeof actual === "object", `${path}: expected object`);
    const left = actual as Record<string, unknown>;
    const right = expected as Record<string, unknown>;
    assert.deepEqual(Object.keys(left).sort(), Object.keys(right).sort(), `${path}: fields`);
    Object.entries(right).forEach(([key, value]) => equivalent(left[key], value, `${path}.${key}`));
  } else {
    assert.equal(actual, expected, path);
  }
}

function snapshot(state: HandState | TournamentState): Snapshot {
  const hand = state instanceof HandState ? state : state.hand;
  assert.ok(hand, "Started hand required");
  const result: Snapshot = {
    players: Object.values(hand.players).map(p => ({
      player_id: p.player_id, stack: p.stack, position: p.position, cards: p.cards,
      street_bet: p.street_bet, contribution: p.contribution, folded: p.folded, acted_at: p.acted_at,
    })),
    button: hand.button, hand_number: hand.hand_number, blind_level_index: hand.blind_level_index,
    blinds: hand.blinds, board: hand.board, street: hand.street,
    pot: hand.pot, highest: hand.highest, last_full_raise: hand.last_full_raise,
    pending: [...hand.pending].sort((a, b) => a - b), current_player: hand.current_player,
    terminal: hand.terminal, showdown: hand.showdown, awards: hand.awards,
    history: hand.history, legal_actions: legalActions(hand),
    observation: hand.terminal ? null : observe(state),
  };
  // JSON object keys and integer/float formatting differ by language, but recall
  // information must match exactly as a structured sequence.
  if (result.observation !== null) {
    const obs = result.observation as unknown as Record<string, unknown>;
    result.observation = { ...obs, recall: JSON.parse(obs.recall as string) };
  }
  if (state instanceof TournamentState) {
    result.tournament = { button: state.button, hand_number: state.hand_number,
      active: state.active, terminal: state.terminal, winner: state.winner, total_chips: state.total_chips, blind_level_index: state.blind_level_index,
      blinds: state.blinds };
  }
  return result;
}

function compare(state: HandState | TournamentState, expected: Snapshot, label: string): void {
  const value = structuredClone(expected);
  if (value.observation !== null) {
    const obs = value.observation as Record<string, unknown>;
    obs.recall = JSON.parse(obs.recall as string);
  }
  equivalent(snapshot(state), value, label);
  (state instanceof HandState ? state : state.hand!).assertInvariants();
  if (state instanceof TournamentState) state.assertInvariants();
}

function stacksRecord(stacks: number[]): Record<number, number> {
  return Object.fromEntries(stacks.map((stack, id) => [id, stack]));
}

test("canonical browser schemas match generated Python fixtures", () => {
  assert.equal(fixtures.version, 1);
  assert.deepEqual([...ACTION_IDS], fixtures.actions);
  assert.deepEqual([...NUMERIC_NAMES], fixtures.numeric_names);
});

for (const fixture of fixtures.hands) {
  test(`Python/browser hand parity: ${fixture.name}`, () => {
    const hand = HandState.start(stacksRecord(fixture.stacks), fixture.button,
      new SeededRNG(7), fixture.blinds, fixture.deck,
      { hand_number: fixture.hand_number, blind_level_index: fixture.blind_level_index });
    compare(hand, fixture.initial, `${fixture.name}.initial`);
    fixture.steps.forEach((step, index) => {
      hand.act(step.action.category, step.action.amount_to);
      compare(hand, step.expected, `${fixture.name}.step_${index}`);
    });
    assert.equal(hand.terminal, true);
  });
}

for (const fixture of fixtures.tournaments) {
  test(`Python/browser persistent tournament parity: ${fixture.name}`, () => {
    const tournament = new TournamentState(stacksRecord(fixture.stacks), fixture.button,
      new SeededRNG(0), new BlindSchedule(fixture.schedule.map(stage => new BlindStage(stage.first_hand, stage.blinds))));
    fixture.steps.forEach((step, index) => {
      if (step.operation === "start") {
        assert.ok(step.deck);
        tournament.startHand(step.deck);
      } else {
        assert.ok(step.action);
        tournament.hand!.act(step.action.category, step.action.amount_to);
      }
      compare(tournament, step.expected, `${fixture.name}.step_${index}`);
    });
  });
}

test("browser evaluator orders all nine hand categories correctly", () => {
  const values = fixtures.category_cards.map(cards => rank7(cards));
  values.slice(1).forEach((value, index) => {
    assert.ok(values[index] > value, `Category ${index} must beat category ${index + 1}`);
  });
});

test("browser evaluator matches Python ordering on 100 deterministic deal pairs", () => {
  fixtures.evaluations.forEach((entry, index) => {
    assert.equal(Math.sign(rank7(entry.a) - rank7(entry.b)), entry.comparison, `Deal pair ${index}`);
  });
});

test("browser rejects invalid betting and card state instead of repairing it", () => {
  const first = fixtures.hands[0];
  const hand = HandState.start(stacksRecord(first.stacks), first.button, new SeededRNG(0),
    { small: 0.5, big: 1.0 }, first.deck);
  assert.throws(() => hand.act("CHECK"));
  assert.throws(() => hand.act("RAISE", 1.5));
  assert.throws(() => hand.act("CALL", 2));
  assert.throws(() => HandState.start({ 0: 25, 1: 25 }, 0, new SeededRNG(0),
    { small: 0.5, big: 1.0 }, Array(52).fill(0)));
  assert.throws(() => rank7([0, 0, 1, 2, 3, 4, 5]));
  hand.pot += 1;
  assert.throws(() => hand.assertInvariants());
});
