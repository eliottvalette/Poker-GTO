// ui/src/lib/policy.ts
import { unpackInfosetKeyDense } from "./infoset";

export const ACTIONS = ["FOLD","CHECK","CALL","RAISE","ALL-IN"] as const;
export type ActionU = typeof ACTIONS[number];
export const ACTIONS_L = ["FOLD","CHECK","CALL","RAISE","ALL-IN"] as const;
export type ActionL = typeof ACTIONS_L[number];

// Actions groupées pour la visualisation
export const GROUPED_ACTIONS = ["raise_allin","check_call","fold"] as const;
export type GroupedAction = typeof GROUPED_ACTIONS[number];
export const GROUPED_LABELS = ["Raise/All-in", "Check/Call", "Fold"] as const;

export const ROLES = ["SB","BB","BTN"] as const;
export const PHASES = ["PREFLOP","FLOP","TURN","RIVER","SHOWDOWN"] as const;

export const ACTION_COLORS: Record<ActionL, string> = {
  FOLD:   "#006DAA",      // Bleu
  CHECK:  "#97e29b",      // Vert
  CALL:   "#97e29b",      // Même vert que check
  RAISE:  "#D62828",      // Rouge
  "ALL-IN": "#D62828"      // Même rouge que raise
};

// Couleurs pour les groupes d'actions
export const GROUP_COLORS: Record<GroupedAction, string> = {
  raise_allin: "#D62828",  // Rouge
  check_call: "#97e29b",   // Vert
  fold: "#006DAA"          // Bleu
};

export type Dist = Partial<Record<ActionU, number>>;
export type GridMix = Record<ActionU, number>;
export type GroupedGridMix = Record<GroupedAction, number>;
export type VisitCounts = number[];

// Policy types
export type PolicyEntry = { dist: Record<string, number>, visits: number };
export type Policy = Record<string, PolicyEntry>;

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
    raise_allin: mix.RAISE + mix["ALL-IN"],
    check_call: mix.CHECK + mix.CALL,
    fold: mix.FOLD
  };
}

// Fonction pour calculer les statistiques de visites
export function calculateVisitStats(visitCounts: VisitCounts): { min: number; max: number; avg: number } {
  const validVisits = visitCounts.filter(v => v > 0);
  if (validVisits.length === 0) return { min: 0, max: 0, avg: 0 };
  
  return {
    min: Math.min(...validVisits),
    max: Math.max(...validVisits),
    avg: validVisits.reduce((a, b) => a + b, 0) / validVisits.length
  };
}

// Fonction pour calculer les statistiques pondérées par visites
export function calculateWeightedStats(visitCounts: VisitCounts, phaseIdx?: number): { 
  totalVisits: number; 
  avgVisitsPerHand: number; 
  coverage: number; // pourcentage de mains avec des visites
} {
  const totalVisits = visitCounts.reduce((a, b) => a + b, 0);
  const handsWithVisits = visitCounts.filter(v => v > 0).length;
  const coverage = (handsWithVisits / visitCounts.length) * 100;
  
  return {
    totalVisits,
    avgVisitsPerHand: totalVisits / visitCounts.length,
    coverage
  };
}

// Fonction pour calculer les statistiques par phase
export function calculatePhaseStats(policy: Policy | null): {
  [phase: string]: { totalVisits: number; avgVisitsPerHand: number; coverage: number };
} {
  if (!policy) {
    return PHASES.reduce((acc, phase) => {
      acc[phase] = { totalVisits: 0, avgVisitsPerHand: 0, coverage: 0 };
      return acc;
    }, {} as Record<string, { totalVisits: number; avgVisitsPerHand: number; coverage: number }>);
  }

  const phaseVisits: Record<string, number[]> = {};
  PHASES.forEach(phase => {
    phaseVisits[phase] = Array(169).fill(0);
  });

  for (const [kStr, { visits }] of Object.entries(policy)) {
    const f = unpackInfosetKeyDense(kStr);
    if (f.hand < 0 || f.hand >= 169) continue;
    
    const phaseName = PHASES[f.phase];
    if (phaseVisits[phaseName]) {
      phaseVisits[phaseName][f.hand] += visits;
    }
  }

  const result: Record<string, { totalVisits: number; avgVisitsPerHand: number; coverage: number }> = {};
  PHASES.forEach(phase => {
    result[phase] = calculateWeightedStats(phaseVisits[phase]);
  });

  return result;
}
