import { ACTION_IDS } from "./actions";

/** Shared solver-action palette for policy analysis and live play. */
export const ACTION_COLORS: Record<typeof ACTION_IDS[number], string> = {
  FOLD: "#64748b", CHECK: "#22c55e", CALL: "#16a34a",
  "RAISE_2.0X": "#fbbf24", "RAISE_2.5X": "#f59e0b", "RAISE_3.0X": "#f97316", "RAISE_4.0X": "#ea580c",
  BET_RAISE_33P: "#fb7185", BET_RAISE_50P: "#f43f5e", BET_RAISE_75P: "#e11d48",
  BET_RAISE_100P: "#be123c", BET_RAISE_150P: "#9f1239", ALL_IN: "#7c3aed",
};

export function actionColor(action: string): string {
  if (!Object.hasOwn(ACTION_COLORS, action)) throw new Error(`Unknown solver action: ${action}`);
  return ACTION_COLORS[action as keyof typeof ACTION_COLORS];
}
