import { ACTION_IDS, legalActions } from "./actions";
import { HandState, TournamentState } from "./engine";

export const STATE_VERSION = 3;
export const HISTORY_WIDTH = 12;
export const POSITIONS = ["BTN", "SB", "BB"] as const;
export const EVENTS = ["BLIND", "FOLD", "CHECK", "CALL", "RAISE", "CARD", "STACK"] as const;
const STREETS = ["PREFLOP", "FLOP", "TURN", "RIVER"] as const;
export const NUMERIC_NAMES = [
  ...["stack", "street_bet", "contribution", "folded", "effective"].flatMap(feature =>
    [0, 1, 2].map(i => `${feature}_${i}`)),
  "pot", "to_call", "highest", "last_full_raise", "hero_bet", "small_blind", "big_blind",
  "player_count", "hero_position", "button", "hand_number", "initial_0", "initial_1", "initial_2",
  ...ACTION_IDS.map(action => `target_${action}`),
  "blind_level_index", "chip_unit_big_blind",
];

export type Observation = {
  version: number;
  hero: number;
  objective: "hand_chip_delta";
  cards: number[];
  street: number;
  numeric: number[];
  legal_mask: boolean[];
  history: number[][];
  recall: string;
};

function sortedJson(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(sortedJson).join(",")}]`;
  if (value !== null && typeof value === "object") {
    const record = value as Record<string, unknown>;
    return `{${Object.keys(record).sort().map(key => `${JSON.stringify(key)}:${sortedJson(record[key])}`).join(",")}}`;
  }
  return JSON.stringify(value);
}

export function validateObservation(obs: Observation): void {
  const known = obs.cards.filter(card => card !== 52);
  if (obs.version !== STATE_VERSION || obs.objective !== "hand_chip_delta"
      || !Number.isInteger(obs.hero) || !Number.isInteger(obs.street) || obs.street < 0 || obs.street > 3
      || obs.cards.length !== 7 || obs.cards.some((card, i) => !Number.isInteger(card) || card < 0 || card > (i < 2 ? 51 : 52))
      || new Set(known).size !== known.length || obs.numeric.length !== NUMERIC_NAMES.length
      || obs.numeric.some(value => !Number.isFinite(value)) || obs.legal_mask.length !== ACTION_IDS.length
      || obs.legal_mask.some(value => typeof value !== "boolean") || !obs.legal_mask.some(Boolean)
      || !obs.history.length || obs.history.some(event => event.length !== HISTORY_WIDTH || event.some(value => !Number.isFinite(value)))
      || !Array.isArray(JSON.parse(obs.recall))) {
    throw new Error(`Invalid structured observation: version=${obs.version}, objective=${obs.objective}, hero=${obs.hero}`);
  }
}

/** Current-hand cEV observation, scaled in current BB and independent of earlier hands. */
export function observe(state: HandState | TournamentState): Observation {
  const singleHand = state instanceof HandState;
  const hand = singleHand ? state : state.hand;
  if (!hand || hand.terminal || hand.current_player === null) throw new Error("Observation requires a live decision");
  const hero = hand.players[hand.current_player];
  const seats = [hero.player_id, ...Object.keys(hand.players).map(Number).filter(id => id !== hero.player_id)];
  const players = seats.map(id => hand.players[id]);
  const actions = new Map(legalActions(hand).map(action => [action.action_id, action]));
  const scale = 25 * hand.blinds.big;
  const numeric: number[] = [];
  for (const feature of ["stack", "street_bet", "contribution", "folded", "effective"] as const) {
    const values = players.map(player => feature === "effective" ? Math.min(hero.stack, player.stack) / scale
      : feature === "folded" ? Number(player.folded) : player[feature] / scale);
    numeric.push(...values, ...Array(3 - values.length).fill(0));
  }
  const handNumber = hand.hand_number;
  numeric.push(hand.pot / scale, hand.toCall() / scale, hand.highest / scale, hand.last_full_raise / scale,
    hero.street_bet / scale, hand.blinds.small / scale, hand.blinds.big / scale, players.length / 3,
    POSITIONS.indexOf(hero.position) / 2, seats.indexOf(hand.button) / 2, handNumber / 25);
  numeric.push(...seats.map(id => hand.initial_stacks[id] / scale), ...Array(3 - seats.length).fill(0));
  numeric.push(...ACTION_IDS.map(action => (actions.get(action)?.amount_to ?? 0) / scale));
  numeric.push(hand.blind_level_index, hand.blinds.big);
  const identity = seats;
  const history: number[][] = [];
  const cardEvent = EVENTS.indexOf("CARD") / 6;
  const stackEvent = EVENTS.indexOf("STACK") / 6;
  const recall = [{ hand: handNumber, button: hand.button, initial: hand.initial_stacks,
    hole: hero.cards, board: hand.board, shown_cards: {}, final_stacks: null, events: hand.history }];
  const handToken = 1 / 25;
  for (const [key, amount] of Object.entries(hand.initial_stacks)) {
    const id = Number(key);
    history.push([handToken, 0, identity.indexOf(id) / 2, POSITIONS.indexOf(hand.players[id].position) / 2,
      stackEvent, amount / scale, 0, 0, 0, 0, 0, 0]);
  }
  for (const card of hero.cards) history.push([handToken, 0, 0, POSITIONS.indexOf(hero.position) / 2,
    cardEvent, 0, 0, 0, 0, 0, card / 51, 1]);
  let revealed = 0;
  for (const event of hand.history) {
    const street = STREETS.indexOf(event.street);
    const required = [0, 3, 4, 5][street];
    for (const card of hand.board.slice(revealed, required)) history.push([handToken, street / 3,
      0, 0, cardEvent, 0, 0, 0, 0, 0, card / 51, 1]);
    revealed = required;
    history.push([handToken, street / 3, identity.indexOf(event.player_id) / 2, POSITIONS.indexOf(event.position) / 2,
      EVENTS.indexOf(event.action) / 6, event.amount_to / scale, event.amount_added / scale,
      event.pot_before / scale, event.pot_after / scale, event.highest_before / scale, 0, 0]);
  }
  for (let i = revealed; i < hand.board.length; i++) history.push([handToken, (i < 3 ? 1 : i - 1) / 3,
    0, 0, cardEvent, 0, 0, 0, 0, 0, hand.board[i] / 51, 1]);
  const observation: Observation = { version: STATE_VERSION, hero: hero.player_id,
    objective: "hand_chip_delta",
    cards: [...hero.cards, ...hand.board, ...Array(5 - hand.board.length).fill(52)],
    street: STREETS.indexOf(hand.street), numeric, legal_mask: ACTION_IDS.map(action => actions.has(action)),
    history, recall: sortedJson(recall) };
  validateObservation(observation);
  return observation;
}
