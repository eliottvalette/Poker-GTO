export const ACTIONS = ["FOLD","CHECK","CALL","RAISE","ALL-IN"] as const;
export type ActionU = typeof ACTIONS[number];
export const ACTIONS_L = ["fold","check","call","raise","all-in"] as const;
export type ActionL = typeof ACTIONS_L[number];

// Actions groupées pour la visualisation
export const GROUPED_ACTIONS = ["fold","check_call","raise_allin"] as const;
export type GroupedAction = typeof GROUPED_ACTIONS[number];
export const GROUPED_LABELS = ["Fold", "Check/Call", "Raise/All-in"] as const;

export const ROLES = ["SB","BB","BTN"] as const;
export const PHASES = ["PREFLOP","FLOP","TURN","RIVER","SHOWDOWN"] as const;

export const ACTION_COLORS: Record<ActionL, string> = {
  fold:   "#006DAA",      // Bleu
  check:  "#97e29b",      // Vert
  call:   "#97e29b",      // Même vert que check
  raise:  "#D62828",      // Rouge
  "all-in":"#D62828"      // Même rouge que raise
};

// Couleurs pour les groupes d'actions
export const GROUP_COLORS: Record<GroupedAction, string> = {
  fold: "#006DAA",        // Bleu
  check_call: "#97e29b",  // Vert
  raise_allin: "#D62828"  // Rouge
};

export type Dist = Partial<Record<ActionU, number>>;
export type GridMix = Record<ActionU, number>;
export type GroupedGridMix = Record<GroupedAction, number>;

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

// Fonction pour regrouper les actions
export function groupActions(mix: GridMix): GroupedGridMix {
  return {
    fold: mix.FOLD,
    check_call: mix.CHECK + mix.CALL,
    raise_allin: mix.RAISE + mix["ALL-IN"]
  };
}
