export const ACTIONS = ["FOLD","CHECK","CALL","RAISE","ALL-IN"] as const;
export type ActionU = typeof ACTIONS[number];
export const ACTIONS_L = ["fold","check","call","raise","all-in"] as const;
export type ActionL = typeof ACTIONS_L[number];

export const ROLES = ["SB","BB","BTN"] as const;
export const PHASES = ["PREFLOP","FLOP","TURN","RIVER","SHOWDOWN"] as const;

export const ACTION_COLORS: Record<ActionL, string> = {
  fold:   "#780000",
  check:  "#C1121F",
  call:   "#FDF0D5",
  raise:  "#669BBC",
  "all-in":"#003049"
};

export type Dist = Partial<Record<ActionU, number>>;
export type GridMix = Record<ActionU, number>;

export function normalize(dist: Dist | undefined): GridMix {
  const v: GridMix = Object.fromEntries(ACTIONS.map(a => [a, dist?.[a] ?? 0])) as GridMix;
  const s = ACTIONS.reduce((acc,a)=>acc+(v[a]||0),0);
  if (s <= 0) {
    const present = ACTIONS.filter(a => dist && a in dist);
    if (present.length === 0) return Object.fromEntries(ACTIONS.map(a=>[a,0])) as GridMix;
    const p = 1/present.length;
    return Object.fromEntries(ACTIONS.map(a=>[a, present.includes(a)?p:0])) as GridMix;
  }
  return Object.fromEntries(ACTIONS.map(a=>[a, (v[a]||0)/s])) as GridMix;
}
