/** Observable-only synthetic behaviors; exact tables remain local CFR's responsibility. */
import { ACTION_IDS } from "./actions";
import { rank7 } from "./evaluator";
import { NUMERIC_NAMES, type Observation } from "./observation";

export const PROFILES: Record<string, [number, number]> = {
  conservative: [0, .25], tight_passive: [.12, -.8], loose_passive: [-.12, -.8],
  tight_aggressive: [.12, .8], loose_aggressive: [-.12, .8], push_fold: [.04, 1.4],
};
const cache = new Map<string, number>();
function canonical(cards: number[]): number[] {
  const rename = (input: number[]) => {
    const suits = new Map<number, number>();
    return input.map(c => {
      if (c === 52) return c;
      if (!suits.has(c % 4)) suits.set(c % 4, suits.size);
      return Math.floor(c / 4) * 4 + suits.get(c % 4)!;
    });
  };
  const a = rename(cards), b = rename([cards[1], cards[0], ...cards.slice(2)]);
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return a[i] < b[i] ? a : b;
  return a;
}
export function showdownStrength(cards: number[], count: number): number {
  const normalized = canonical(cards), key = [...normalized, count].join();
  if (cache.has(key)) return cache.get(key)!;
  const known = normalized.filter(c => c < 52);
  let seed = 2166136261, total = 0;
  for (const c of [...normalized, count]) seed = Math.imul(seed ^ c, 16777619) >>> 0;
  for (let sample = 0; sample < 24; sample++) {
    const deck = Array.from({ length: 52 }, (_, c) => c).filter(c => !known.includes(c)), drawn: number[] = [];
    for (let i = 0; i < 2 * (count - 1) + 7 - known.length; i++) {
      seed = (Math.imul(1664525, seed) + 1013904223) >>> 0;
      drawn.push(deck.splice(seed % deck.length, 1)[0]);
    }
    const board = [...known.slice(2), ...drawn.slice(2 * (count - 1))];
    const ranks = [rank7([...known.slice(0, 2), ...board])];
    for (let i = 0; i < count - 1; i++) ranks.push(rank7([...drawn.slice(2*i, 2*i+2), ...board]));
    const best = Math.max(...ranks);
    total += ranks[0] === best ? 1 / ranks.filter(r => r === best).length : 0;
  }
  if (cache.size >= 32768) cache.clear();
  cache.set(key, total / 24);
  return total / 24;
}
export function behaviorProbabilities(obs: Observation, profile: string): number[] {
  if (profile === "uniform") return obs.legal_mask.map(m => Number(m) / obs.legal_mask.filter(Boolean).length);
  if (!(profile in PROFILES)) throw new Error(`Unsupported behavior profile: ${profile}`);
  const values = Object.fromEntries(NUMERIC_NAMES.map((n, i) => [n, obs.numeric[i]]));
  const strength = showdownStrength(obs.cards, Math.round(values.player_count * 3));
  const [tightness, aggression] = PROFILES[profile], pot = values.pot, call = Math.min(values.to_call, values.stack_0);
  const advantage = strength - (call ? call / (pot + call) : 0) - tightness;
  const scores = ACTION_IDS.map((action, i) => {
    if (!obs.legal_mask[i]) return 0;
    let logit: number;
    if (action === "FOLD") logit = -5 * advantage;
    else if (action === "CHECK" || action === "CALL") logit = 3 * advantage - aggression;
    else {
      const added = Math.max(0, values[`target_${action}`] - values.hero_bet);
      logit = 6 * (strength - .5 - tightness) + aggression - 2 * (added ? added / (pot + added) : 0);
    }
    if (profile === "push_fold" && action !== "FOLD" && action !== "ALL_IN") logit -= 5;
    return Math.exp(Math.max(-20, Math.min(20, logit)));
  });
  const sum = scores.reduce((a, b) => a + b, 0);
  return scores.map(v => v / sum);
}
