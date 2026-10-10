/** Public posterior factors persist across actions; Hero conditioning belongs to analysis. */
import { HandState, SeededRNG } from "./engine";
import { ACTION_IDS, legalActions, type SolverAction } from "./actions";
import { observe } from "./observation";
import { normalizeRanges, restoreHand, type HandTransport, type Ranges } from "./hybrid";
import { behaviorProbabilities } from "./behavior";

export type BeliefTransition = { before: HandTransport; after: HandTransport; action: SolverAction };

export function recoverHandStart(hand: HandState): HandState {
  // Canonical engine replay only. No policy receives this internal dealt state.
  const dealt = Object.values(hand.players).flatMap(player => player.cards);
  const deck = [...hand.deck, ...[...hand.board].reverse(), ...dealt.reverse()];
  return HandState.start(hand.initial_stacks, hand.button, new SeededRNG(0), hand.blinds, deck,
    { hand_number: hand.hand_number, blind_level_index: hand.blind_level_index });
}

export function handClassMasses(rows: Ranges[number]): Record<string, number> {
  const result: Record<string, number> = {}, ranks = "23456789TJQKA";
  for (const row of rows) {
    const [a,b] = row.cards, high = Math.max(a >> 2, b >> 2), low = Math.min(a >> 2, b >> 2);
    const label = ranks[high]+ranks[low]+(high === low ? "" : a%4 === b%4 ? "s" : "o");
    result[label] = (result[label] ?? 0) + row.probability;
  }
  return result;
}
export function rankClassMasses(classes: Record<string, number>): Record<string, number> {
  const result: Record<string, number> = {};
  for (const [label, mass] of Object.entries(classes)) {
    const ranks = label.slice(0, 2);
    result[ranks] = (result[ranks] ?? 0) + mass;
  }
  return result;
}
export function updateRanges(before: HandState, after: HandState, ranges: Ranges,
                             action: SolverAction, profile: string, interpolate = false, predictions?: Record<string, number[]>): Ranges {
  const actor = before.current_player!;
  const checked = before.clone(); checked.act(action.category, action.amount_to);
  if (JSON.stringify(checked.history) !== JSON.stringify(after.history) || checked.pot !== after.pot
      || checked.board.join() !== after.board.join() || checked.current_player !== after.current_player)
    throw new Error("Observed transition does not match the canonical action");
  const prior = normalizeRanges(before, ranges), legal = legalActions(before);
  const rows = prior[actor].map(row => {
    const view = before.clone(); view.players[actor].cards = row.cards;
    const probabilities = predictions ? predictions[row.cards.join()] : behaviorProbabilities(observe(view), profile);
    if (!probabilities) throw new Error("Missing candidate-hand action likelihood");
    const match = legal.find(a => a.category === action.category && a.amount_to === action.amount_to);
    let likelihood: number | undefined;
    if (match) likelihood = probabilities[ACTION_IDS.findIndex(id => id === match.action_id)];
    else {
      if (!interpolate || action.category !== "RAISE" || action.amount_to === null)
        throw new Error("Off-tree action requires explicit likelihood interpolation");
      const raises = legal.filter(a => a.category === "RAISE").sort((a,b) => a.amount_to! - b.amount_to!);
      for (let i = 1; i < raises.length; i++) {
        const low = raises[i-1], high = raises[i];
        if (low.amount_to! < action.amount_to && action.amount_to < high.amount_to!) {
          const f = (action.amount_to-low.amount_to!)/(high.amount_to!-low.amount_to!);
          likelihood = (1-f)*probabilities[ACTION_IDS.findIndex(id => id === low.action_id)] + f*probabilities[ACTION_IDS.findIndex(id => id === high.action_id)];
        }
      }
      if (likelihood === undefined) throw new Error(`Unsupported off-tree extrapolation: ${action.amount_to}`);
    }
    return { cards: row.cards, probability: row.probability * likelihood };
  });
  const result = normalizeRanges(after, { ...prior, [actor]: rows });
  const seats = Object.keys(result).map(Number);
  let visits = 0;
  function witness(index: number, used: number[]): boolean {
    if (index === seats.length) return true;
    for (const row of result[seats[index]]) {
      if (++visits > 100000) throw new Error("Joint compatibility witness budget exceeded");
      if (!row.cards.some(c => used.includes(c)) && witness(index + 1, [...used, ...row.cards])) return true;
    }
    return false;
  }
  if (!witness(0, after.board)) throw new Error("Observation leaves no compatible joint deal");
  return result;
}

export function replayBeliefs(prior: Ranges, transitions: BeliefTransition[], profiles: Record<number,string>, interpolate: boolean): Ranges {
  let ranges = prior;
  for (const transition of transitions) {
    const before = restoreHand(transition.before), after = restoreHand(transition.after);
    const profile = profiles[before.current_player!];
    if (!profile) throw new Error(`Missing action likelihood profile for seat ${before.current_player}`);
    ranges = updateRanges(before, after, ranges, transition.action, profile, interpolate);
  }
  return ranges;
}

/** Exact private marginals for HU/3-max factorized public beliefs.
 * With the observer fixed, at most two unknown hands remain. Inclusion/exclusion
 * integrates their mutually exclusive cards without enumerating 1,326² pairs.
 */
export function observerMarginals(hand: HandState, publicRanges: Ranges, observer: number): Ranges {
  if (!hand.players[observer]) throw new Error("Range observer is not seated");
  const prior = normalizeRanges(hand, publicRanges), known = hand.players[observer].cards;
  const identity = [...known].sort((a,b)=>a-b).join();
  if (!prior[observer].some(r => r.cards.join() === identity)) throw new Error("Observer holding has no public range support");
  const conditioned = normalizeRanges(hand, Object.fromEntries(Object.entries(prior).map(([seat, rows]) => [seat,
    Number(seat) === observer ? [{ cards: known, probability: 1 }] : rows.filter(r => !r.cards.some(c => known.includes(c)))])));
  const unknown = Object.keys(conditioned).map(Number).filter(p => p !== observer);
  if (unknown.length < 1 || unknown.length > 2) throw new Error("Range marginals require HU or three players");
  if (unknown.length === 1) return conditioned;
  const result: Ranges = { [observer]: conditioned[observer] };
  for (const seat of unknown) {
    const other = conditioned[unknown.find(p => p !== seat)!];
    const mass = Array(52).fill(0) as number[], pairs = new Map<string, number>();
    let total = 0;
    for (const row of other) {
      total += row.probability;
      for (const card of row.cards) mass[card] += row.probability;
      pairs.set(row.cards.join(),row.probability);
    }
    const rows = conditioned[seat].map(row => ({ ...row, probability: row.probability * Math.max(0,
      total - mass[row.cards[0]] - mass[row.cards[1]] + (pairs.get(row.cards.join()) ?? 0)) }));
    const normalizer = rows.reduce((s,r)=>s+r.probability,0);
    if (!(normalizer > 1e-14)) throw new Error("No numerically resolvable compatible joint range mass");
    result[seat] = rows.filter(r=>r.probability>0).map(r=>({...r,probability:r.probability/normalizer}));
  }
  return result;
}

export function rangeStateKey(hand: HandState, observer: number): string {
  return JSON.stringify([hand.hand_number,hand.button,hand.board,hand.history,observer,hand.players[observer]?.cards]);
}

export type RangeSnapshot = {
  sessionId: number; stateKey: string; status: "updating" | "current" | "error";
  profiles: Record<number,string>; marginals?: Ranges; error?: string;
  display?: import("./range-display").RangeDisplay;
};
