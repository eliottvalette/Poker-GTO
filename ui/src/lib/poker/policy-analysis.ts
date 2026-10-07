import { type LoadedAveragePolicy } from "../onnx-policy";
import { applyAction, legalActions } from "./actions";
import { HandState, SeededRNG, STREETS, type Position, type Street } from "./engine";
import { observe } from "./observation";

export type RangeCell = { label: string; combos: number; probabilities: Record<string, number> };
const RANKS = ["A", "K", "Q", "J", "10", "9", "8", "7", "6", "5", "4", "3", "2"];

/** A legal call/check line provides explicit public history for the selected spot. */
export function analysisRoot(count: number, position: Position, street: Street, stacks: number[], big: number, seed: number): HandState {
  if (count !== 2 && count !== 3) throw new Error(`Unsupported player count: ${count}`);
  if (count === 2 && position === "BTN") throw new Error("HU button position is SB, not BTN");
  const hand = HandState.start(Object.fromEntries(stacks.slice(0, count).map((stack, id) => [id, stack])),
    0, new SeededRNG(seed), { small: big / 2, big });
  const target = STREETS.indexOf(street);
  for (let n = 0; n < 100; n++) {
    if (hand.terminal) throw new Error("Selected stacks reach settlement before the requested spot");
    if (STREETS.indexOf(hand.street) === target && hand.actor.position === position) return hand;
    if (STREETS.indexOf(hand.street) > target) throw new Error("Requested position has no live decision on this street");
    const actions = legalActions(hand);
    const passive = actions.find(action => action.action_id === "CHECK" || action.action_id === "CALL");
    if (!passive) throw new Error("No call/check continuation for the requested spot");
    applyAction(hand, passive.action_id);
  }
  throw new Error("Analysis root exceeded 100 public actions");
}

export function holeObservation(hand: HandState, cards: [number, number]) {
  if (new Set(cards).size !== 2 || cards.some(card => !Number.isInteger(card) || card < 0 || card > 51 || hand.board.includes(card))) {
    throw new Error("Hero cards must be two distinct cards absent from the public board");
  }
  const candidate = hand.clone();
  candidate.actor.cards = [...cards];
  // Only the acting player's cards and public information enter observe().
  return observe(candidate);
}

/** Swap a chosen observable holding into the deal without duplicate future cards. */
export function withHeroCards(hand: HandState, cards: [number, number]): HandState {
  holeObservation(hand, cards);
  const next = hand.clone();
  cards.forEach((card, index) => {
    const previous = next.actor.cards[index];
    const swap = (value: number) => value === previous ? card : value === card ? previous : value;
    next.deck = next.deck.map(swap);
    for (const player of Object.values(next.players)) player.cards = player.cards.map(swap) as [number, number];
  });
  next.assertInvariants();
  return next;
}

/** Uniform exact-combo aggregation, conditional on the fixed public board. */
export async function queryRange(hand: HandState, policy: LoadedAveragePolicy,
  cancelled: () => boolean, progress: (done: number) => void): Promise<RangeCell[] | null> {
  const cells: RangeCell[] = Array.from({ length: 169 }, (_, index) => {
    const row = Math.floor(index / 13), col = index % 13;
    return { label: row === col ? RANKS[row] + RANKS[col]
      : RANKS[Math.min(row, col)] + RANKS[Math.max(row, col)] + (row < col ? "s" : "o"),
      combos: 0, probabilities: {} };
  });
  const blocked = new Set(hand.board);
  let done = 0;
  for (let a = 0; a < 52; a++) for (let b = a + 1; b < 52; b++) {
    if (blocked.has(a) || blocked.has(b)) continue;
    if (cancelled()) return null;
    const probabilities = await policy.query(holeObservation(hand, [a, b]));
    const ra = 12 - Math.floor(a / 4), rb = 12 - Math.floor(b / 4);
    const hi = Math.min(ra, rb), lo = Math.max(ra, rb);
    const index = ra === rb ? ra * 13 + rb : a % 4 === b % 4 ? hi * 13 + lo : lo * 13 + hi;
    const cell = cells[index];
    cell.combos++;
    for (const [action, probability] of Object.entries(probabilities)) {
      cell.probabilities[action] = (cell.probabilities[action] ?? 0) + probability;
    }
    done++;
    if (done % 32 === 0) {
      progress(done);
      await new Promise(resolve => setTimeout(resolve, 0));
    }
  }
  for (const cell of cells) for (const action of Object.keys(cell.probabilities)) cell.probabilities[action] /= cell.combos;
  progress(done);
  return cancelled() ? null : cells;
}
