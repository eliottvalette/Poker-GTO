/** Canonical discrete sizing shared by browser play, policy masks and parity tests. */
import { EPS, type HandState } from "./engine";
export const ACTION_IDS = ["FOLD", "CHECK", "CALL", "RAISE_2.0X", "RAISE_2.5X", "RAISE_3.0X", "RAISE_4.0X",
  "BET_RAISE_33P", "BET_RAISE_50P", "BET_RAISE_75P", "BET_RAISE_100P", "BET_RAISE_150P", "ALL_IN"] as const;
export type ActionCategory = "FOLD" | "CHECK" | "CALL" | "RAISE";
export type SolverAction = { action_id: string; category: ActionCategory; amount_to: number | null };
export function legalActions(hand: HandState): SolverAction[] {
  if (hand.terminal) return [];
  const call = hand.toCall();
  const actions: SolverAction[] = call > EPS ? [
    { action_id: "FOLD", category: "FOLD", amount_to: null },
    { action_id: "CALL", category: "CALL", amount_to: null },
  ] : [{ action_id: "CHECK", category: "CHECK", amount_to: null }];
  if (!hand.canRaise()) return actions;
  const maximum = hand.actor.stack + hand.actor.street_bet;
  const targets: [string, number][] = hand.street === "PREFLOP"
    ? [2, 2.5, 3, 4].map(x => [`RAISE_${x.toFixed(1)}X`, x * Math.max(hand.blinds.big, hand.highest)])
    : [33, 50, 75, 100, 150].map(pct => [`BET_RAISE_${pct}P`, hand.highest + pct / 100 * (hand.pot + call)]);
  const seen: number[] = [];
  for (const [action_id, candidate] of targets) {
    const target = Math.max(candidate, hand.minRaiseTo);
    if (target >= maximum - EPS || seen.some(t => Math.abs(t - target) < EPS)) continue;
    actions.push({ action_id, category: "RAISE", amount_to: target });
    seen.push(target);
  }
  actions.push({ action_id: "ALL_IN", category: "RAISE", amount_to: maximum });
  return actions;
}
export function applyAction(hand: HandState, actionId: string): void {
  const actions = legalActions(hand);
  const action = actions.find(a => a.action_id === actionId);
  if (!action) throw new Error(`Illegal canonical action ${JSON.stringify(actionId)}; expected ${actions.map(a => a.action_id)}`);
  hand.act(action.category, action.amount_to);
}
