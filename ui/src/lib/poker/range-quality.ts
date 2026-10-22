import type { RangeCell } from "./policy-analysis";

/** Detect near-constant learned output without prescribing a poker strategy. */
export function rangeQuality(cells: RangeCell[]): { usable: boolean; maximumActionSpan: number } {
  if (cells.length !== 169 || cells.some(cell => cell.combos < 1)) throw new Error("Complete preflop range required for quality checks");
  const actions = new Set(cells.flatMap(cell => Object.keys(cell.probabilities)));
  let maximumActionSpan = 0;
  for (const action of actions) {
    const values = cells.map(cell => cell.probabilities[action] ?? 0);
    if (values.some(value => !Number.isFinite(value) || value < 0 || value > 1)) throw new Error("Invalid preflop probabilities");
    maximumActionSpan = Math.max(maximumActionSpan, Math.max(...values) - Math.min(...values));
  }
  return { usable: maximumActionSpan >= .15, maximumActionSpan };
}
