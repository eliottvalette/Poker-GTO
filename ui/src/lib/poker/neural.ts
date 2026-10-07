/** Versioned neural inputs from hero/public observations; diagnostic recall is excluded. */
import { ACTION_IDS } from "./actions";
import { NUMERIC_NAMES, type Observation } from "./observation";

export const FEATURE_SCHEMA_VERSION = 3;
export const ARCHITECTURE = "cards8_numeric32_historyGRU32_head64_features3";
const CARD_FEATURE_NAMES = ["made_category", "flush_draw", "straight_outs", "flush_outs", "overcards",
  "board_pair_multiplicity", "board_max_suit", "board_rank_span", "ace_suit_blockers"];
export const NEURAL_NUMERIC_NAMES = [...NUMERIC_NAMES, "pot_odds", "call_pot", "hero_stack_pot", "spr_1", "spr_2",
  ...ACTION_IDS.map(a => `target_pot_${a}`), ...CARD_FEATURE_NAMES];

function counts(values: number[]): Map<number, number> {
  const result = new Map<number, number>();
  for (const value of values) result.set(value, (result.get(value) ?? 0) + 1);
  return result;
}
function straight(input: Set<number>): boolean {
  const ranks = new Set(input);
  if (ranks.has(14)) ranks.add(1);
  for (let low = 1; low <= 10; low++) {
    if ([0, 1, 2, 3, 4].every(offset => ranks.has(low + offset))) return true;
  }
  return false;
}
function category(cards: number[]): number {
  const ranks = counts(cards.map(c => Math.floor(c / 4) + 2));
  const suits = counts(cards.map(c => c % 4));
  const flush = [...suits].filter(([, n]) => n >= 5).map(([s]) => s);
  if (flush.some(s => straight(new Set(cards.filter(c => c % 4 === s).map(c => Math.floor(c / 4) + 2))))) return 8;
  const sizes = [...ranks.values()].sort((a, b) => b - a);
  if (sizes[0] === 4) return 7;
  if (sizes[0] >= 3 && sizes.length > 1 && sizes[1] >= 2) return 6;
  if (flush.length) return 5;
  if (straight(new Set(ranks.keys()))) return 4;
  if (sizes[0] >= 3) return 3;
  if (sizes.filter(n => n >= 2).length >= 2) return 2;
  return Number(sizes.some(n => n >= 2));
}

export function neuralObservation(obs: Observation, featureVersion: number = FEATURE_SCHEMA_VERSION): { cards: number[]; numeric: number[]; history: number[][] } {
  if (![2, 3].includes(featureVersion)) throw new Error(`Unsupported feature schema: ${featureVersion}`);
  const count = Math.round(obs.numeric[NUMERIC_NAMES.indexOf("player_count")] * 3);
  const positions = count === 3 ? [0, 1, 2] : [1, 2];
  const stacks = obs.history.slice(0, count);
  const byPosition = new Map(stacks.map(row => [Math.round(row[3] * 2), Math.round(row[2] * 2)]));
  const heroPosition = Math.round(obs.numeric[NUMERIC_NAMES.indexOf("hero_position")] * 2);
  if (![2, 3].includes(count) || stacks.some(row => row[4] !== 1) || byPosition.size !== count
      || positions.some(position => !byPosition.has(position)) || byPosition.get(heroPosition) !== 0
      || new Set(byPosition.values()).size !== count
      || [...byPosition.values()].some(index => index < 0 || index >= count)) {
    throw new Error(`Invalid observable position mapping: players=${count}, heroPosition=${heroPosition}`);
  }
  const offset = positions.indexOf(heroPosition);
  const order = positions.map((_, i) => byPosition.get(positions[(offset + i) % count])!);
  const mapping = new Map(order.map((old, current) => [old, current]));
  const rawNumeric = [...obs.numeric];
  for (const feature of ["stack", "street_bet", "contribution", "folded", "effective", "initial"]) {
    order.forEach((old, current) => {
      rawNumeric[NUMERIC_NAMES.indexOf(`${feature}_${current}`)] = obs.numeric[NUMERIC_NAMES.indexOf(`${feature}_${old}`)];
    });
  }
  const button = NUMERIC_NAMES.indexOf("button");
  rawNumeric[button] = mapping.get(Math.round(obs.numeric[button] * 2))! / 2;
  const normalize = (input: number[]) => {
    const suits = new Map<number, number>();
    const cards = input.map(card => {
      if (card === 52) return card;
      const suit = card % 4;
      if (!suits.has(suit)) suits.set(suit, suits.size);
      return Math.floor(card / 4) * 4 + suits.get(suit)!;
    });
    return { cards, suits };
  };
  let selected = normalize(obs.cards);
  if (featureVersion === 3) {
    const reversed = normalize([obs.cards[1], obs.cards[0], ...obs.cards.slice(2)]);
    const different = selected.cards.findIndex((card, i) => card !== reversed.cards[i]);
    if (different >= 0 && reversed.cards[different] < selected.cards[different]) selected = reversed;
  }
  const { cards, suits } = selected;
  const history = obs.history.map((row, index) => {
    const event = [...row];
    const actor = mapping.get(Math.round(row[2] * 2));
    if (actor === undefined) throw new Error(`History actor outside observed seats: ${row[2]}`);
    event[2] = actor / 2;
    if (event[11] === 1) {
      const card = Math.round(event[10] * 51);
      if (!suits.has(card % 4)) throw new Error(`History contains an unobservable card: ${card}`);
      if (featureVersion === 3 && index >= count && index < count + 2) {
        if (card !== obs.cards[index - count]) throw new Error("Private-card history disagrees with observable cards");
        event[10] = cards[index - count] / 51;
      } else event[10] = (Math.floor(card / 4) * 4 + suits.get(card % 4)!) / 51;
    }
    return event;
  });
  history.splice(0, count, ...history.slice(0, count).sort((a, b) => a[2] - b[2]));
  const state = Object.fromEntries(NUMERIC_NAMES.map((name, index) => [name, rawNumeric[index]]));
  const pot = state.pot, call = Math.min(state.to_call, state.stack_0);
  if (pot <= 0) throw new Error(`Live hand must have a positive pot: ${pot}`);
  const hero = obs.cards.slice(0, 2), board = obs.cards.slice(2).filter(c => c !== 52), known = [...hero, ...board];
  const ranks = new Set(known.map(c => Math.floor(c / 4) + 2));
  const suitCounts = counts(known.map(c => c % 4));
  const boardRanks = counts(board.map(c => Math.floor(c / 4) + 2));
  const boardSuits = counts(board.map(c => c % 4));
  const drawing = board.length === 3 || board.length === 4;
  const remaining = Array.from({ length: 52 }, (_, c) => c).filter(c => !known.includes(c));
  const straightOuts = drawing && !straight(ranks) ? remaining.filter(c => straight(new Set([...ranks, Math.floor(c / 4) + 2]))).length : 0;
  const flushOuts = drawing && Math.max(...suitCounts.values()) < 5 ? remaining.filter(c => suitCounts.get(c % 4) === 4).length : 0;
  const derived = [call / (pot + call), call / pot, state.stack_0 / pot, state.effective_1 / pot, state.effective_2 / pot,
    ...ACTION_IDS.map(a => state[`target_${a}`] / pot), category(known) / 8,
    Number(drawing && Math.max(...suitCounts.values()) === 4), straightOuts / 52, flushOuts / 52,
    hero.filter(c => Math.floor(c / 4) + 2 > (board.length ? Math.max(...boardRanks.keys()) : 14)).length / 2,
    Math.max(0, ...boardRanks.values()) / 4, Math.max(0, ...boardSuits.values()) / 5,
    board.length ? (Math.max(...boardRanks.keys()) - Math.min(...boardRanks.keys())) / 12 : 0,
    hero.filter(c => Math.floor(c / 4) === 12 && (boardSuits.get(c % 4) ?? 0) > 0).length / 2];
  return { cards, numeric: [...rawNumeric, ...derived], history };
}
