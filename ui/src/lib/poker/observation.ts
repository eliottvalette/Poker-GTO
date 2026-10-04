import { ACTION_IDS, legalActions } from "./actions";
import { HandState, TournamentState } from "./engine";

export const STATE_VERSION = 2;
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
];

export type Observation = {
  version: number;
  hero: number;
  objective: "hand_chip_delta" | "tournament_winner";
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
  if (obs.version !== STATE_VERSION || !["hand_chip_delta", "tournament_winner"].includes(obs.objective)
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

/** Exact Python infoset.py feature contract, including the complete tournament history. */
export function observe(state: HandState | TournamentState): Observation {
  const singleHand = state instanceof HandState;
  const hand = singleHand ? state : state.hand;
  if (!hand || hand.terminal || hand.current_player === null) throw new Error("Observation requires a live decision");
  const hero = hand.players[hand.current_player];
  const seats = [hero.player_id, ...Object.keys(hand.players).map(Number).filter(id => id !== hero.player_id)];
  const players = seats.map(id => hand.players[id]);
  const actions = new Map(legalActions(hand).map(action => [action.action_id, action]));
  const numeric: number[] = [];
  for (const feature of ["stack", "street_bet", "contribution", "folded", "effective"] as const) {
    const values = players.map(player => feature === "effective" ? Math.min(hero.stack, player.stack) / 25
      : feature === "folded" ? Number(player.folded) : player[feature] / 25);
    numeric.push(...values, ...Array(3 - values.length).fill(0));
  }
  const handNumber = singleHand ? 1 : state.hand_number;
  numeric.push(hand.pot / 25, hand.toCall() / 25, hand.highest / 25, hand.last_full_raise / 25,
    hero.street_bet / 25, hand.blinds.small / 25, hand.blinds.big / 25, players.length / 3,
    POSITIONS.indexOf(hero.position) / 2, seats.indexOf(hand.button) / 2, handNumber / 25);
  numeric.push(...seats.map(id => hand.initial_stacks[id] / 25), ...Array(3 - seats.length).fill(0));
  numeric.push(...ACTION_IDS.map(action => (actions.get(action)?.amount_to ?? 0) / 25));
  const hands = singleHand ? [hand] : [...state.completed, hand];
  const identity = singleHand ? Object.keys(hand.players).map(Number) : [...state.original_players];
  identity.splice(identity.indexOf(hero.player_id), 1);
  identity.unshift(hero.player_id);
  const history: number[][] = [];
  const recall: unknown[] = [];
  const cardEvent = EVENTS.indexOf("CARD") / 6;
  const stackEvent = EVENTS.indexOf("STACK") / 6;
  for (let index = 0; index < hands.length; index++) {
    const previous = hands[index];
    const n = index + 1;
    const own = previous.players[hero.player_id];
    const privateCards = own?.cards ?? [];
    const previousPlayers = Object.values(previous.players);
    const shown = Object.fromEntries(previousPlayers.filter(player => previous.showdown && !player.folded)
      .map(player => [player.player_id, [...player.cards]]));
    recall.push({ hand: n, button: previous.button, initial: previous.initial_stacks, hole: privateCards,
      board: previous.board, shown_cards: shown,
      final_stacks: previous.terminal ? Object.fromEntries(previousPlayers.map(player => [player.player_id, player.stack])) : null,
      events: previous.history });
    for (const [key, amount] of Object.entries(previous.initial_stacks)) {
      const id = Number(key);
      history.push([n / 25, 0, identity.indexOf(id) / 2, POSITIONS.indexOf(previous.players[id].position) / 2,
        stackEvent, amount / 25, 0, 0, 0, 0, 0, 0]);
    }
    for (const card of privateCards) history.push([n / 25, 0, 0, POSITIONS.indexOf(own.position) / 2,
      cardEvent, 0, 0, 0, 0, 0, card / 51, 1]);
    let revealed = 0;
    for (const event of previous.history) {
      const street = STREETS.indexOf(event.street);
      const required = [0, 3, 4, 5][street];
      for (const card of previous.board.slice(revealed, required)) history.push([n / 25, street / 3,
        0, 0, cardEvent, 0, 0, 0, 0, 0, card / 51, 1]);
      revealed = required;
      history.push([n / 25, street / 3, identity.indexOf(event.player_id) / 2, POSITIONS.indexOf(event.position) / 2,
        EVENTS.indexOf(event.action) / 6, event.amount_to / 25, event.amount_added / 25,
        event.pot_before / 25, event.pot_after / 25, event.highest_before / 25, 0, 0]);
    }
    for (let i = revealed; i < previous.board.length; i++) history.push([n / 25, (i < 3 ? 1 : i - 1) / 3,
      0, 0, cardEvent, 0, 0, 0, 0, 0, previous.board[i] / 51, 1]);
    if (previous.terminal) {
      for (const [key, cards] of Object.entries(shown)) {
        const id = Number(key);
        if (id !== hero.player_id) for (const card of cards) history.push([n / 25, 1, identity.indexOf(id) / 2,
          POSITIONS.indexOf(previous.players[id].position) / 2, cardEvent, 0, 0, 0, 0, 0, card / 51, 1]);
      }
      for (const player of previousPlayers) history.push([n / 25, STREETS.indexOf(previous.street) / 3,
        identity.indexOf(player.player_id) / 2, POSITIONS.indexOf(player.position) / 2,
        stackEvent, player.stack / 25, 0, 0, 0, 0, 0, 0]);
    }
  }
  const observation: Observation = { version: STATE_VERSION, hero: hero.player_id,
    objective: singleHand ? "hand_chip_delta" : "tournament_winner",
    cards: [...hero.cards, ...hand.board, ...Array(5 - hand.board.length).fill(52)],
    street: STREETS.indexOf(hand.street), numeric, legal_mask: ACTION_IDS.map(action => actions.has(action)),
    history, recall: sortedJson(recall) };
  validateObservation(observation);
  return observation;
}
