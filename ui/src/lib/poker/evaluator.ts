/** Exact five/seven-card rankings. Higher encoded lexicographic scores win. */
function validate(cards: readonly number[], count: number): void {
  if (cards.length !== count || new Set(cards).size !== count || cards.some(c => !Number.isInteger(c) || c < 0 || c >= 52)) {
    throw new Error(`Expected ${count} distinct card IDs 0..51: ${JSON.stringify(cards)}`);
  }
}

function score(category: number, kickers: number[]): number {
  let value = category;
  for (let i = 0; i < 5; i++) value = value * 15 + (kickers[i] ?? 0);
  return value;
}

export function rank5(cards: readonly number[]): number {
  validate(cards, 5);
  const ranks = cards.map(c => Math.floor(c / 4) + 2).sort((a, b) => b - a);
  const counts = new Map<number, number>();
  ranks.forEach(r => counts.set(r, (counts.get(r) ?? 0) + 1));
  const groups = [...counts].sort((a, b) => b[1] - a[1] || b[0] - a[0]);
  const flush = cards.every(c => c % 4 === cards[0] % 4);
  const unique = [...new Set(ranks)];
  const straight = unique.length === 5 && unique[0] - unique[4] === 4 ? unique[0]
    : unique.join(",") === "14,5,4,3,2" ? 5 : 0;
  if (flush && straight) return score(8, [straight]);
  if (groups[0][1] === 4) return score(7, [groups[0][0], groups[1][0]]);
  if (groups[0][1] === 3 && groups[1][1] === 2) return score(6, [groups[0][0], groups[1][0]]);
  if (flush) return score(5, ranks);
  if (straight) return score(4, [straight]);
  if (groups[0][1] === 3) return score(3, [groups[0][0], ...groups.slice(1).map(g => g[0])]);
  if (groups[0][1] === 2 && groups[1][1] === 2) return score(2, [groups[0][0], groups[1][0], groups[2][0]]);
  if (groups[0][1] === 2) return score(1, [groups[0][0], ...groups.slice(1).map(g => g[0])]);
  return score(0, ranks);
}

export function rank7(cards: readonly number[]): number {
  validate(cards, 7);
  let best = -Infinity;
  for (let a = 0; a < 3; a++) for (let b = a + 1; b < 4; b++)
    for (let c = b + 1; c < 5; c++) for (let d = c + 1; d < 6; d++)
      for (let e = d + 1; e < 7; e++) best = Math.max(best, rank5([cards[a], cards[b], cards[c], cards[d], cards[e]]));
  return best;
}
