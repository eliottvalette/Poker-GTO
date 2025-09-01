// ui/src/lib/infoset.ts
// [ HEROBOARD:4 | SPR:8 | RATIO:8 | POT:8 | BOARD:5 | HAND:8 | ROLE:2 | PHASE:3 ]
const POS = { HEROBOARD:0, SPR:4, RATIO:12, POT:20, BOARD:28, HAND:33, ROLE:41, PHASE:43 } as const;
const MASK= { HEROBOARD:(1<<4)-1, SPR:(1<<8)-1, RATIO:(1<<8)-1, POT:(1<<8)-1,
              BOARD:(1<<5)-1, HAND:(1<<8)-1, ROLE:(1<<2)-1, PHASE:(1<<3)-1 } as const;

export type UnpackedKey = {
  phase:number; role:number; hand:number; board:number;
  pot:number; ratio:number; spr:number; heroboard:number;
};

export function unpackInfosetKeyDense(kStr: string): UnpackedKey {
  const n = BigInt(kStr);
  const g = (pos:number, mask:number) => Number((n >> BigInt(pos)) & BigInt(mask));
  return {
    phase:     g(POS.PHASE,     MASK.PHASE),
    role:      g(POS.ROLE,      MASK.ROLE),
    hand:      g(POS.HAND,      MASK.HAND),
    board:     g(POS.BOARD,     MASK.BOARD),
    pot:       g(POS.POT,       MASK.POT),      // qlog pot bucket id
    ratio:     g(POS.RATIO,     MASK.RATIO),    // toCall/pot bucket id
    spr:       g(POS.SPR,       MASK.SPR),      // SPR bucket id
    heroboard: g(POS.HEROBOARD, MASK.HEROBOARD) // relation héros-board
  };
}

export const ROLE_NAMES = ["SB","BB","BTN"] as const;
export const PHASE_NAMES = ["PREFLOP","FLOP","TURN","RIVER","SHOWDOWN"] as const;

// 169 → label (AA, AKo, A2s, etc.)
export function handLabel169(idx: number): string {
  if (idx < 0 || idx >= 169) return String(idx);
  const labs = ["A","K","Q","J","T","9","8","7","6","5","4","3","2"];
  const i = Math.floor(idx / 13), j = idx % 13;
  if (i === j) return `${labs[i]}${labs[j]}`;
  if (i < j)   return `${labs[i]}${labs[j]}s`;
  return `${labs[j]}${labs[i]}o`;
}

// Buckets lisibles
export const BOARD_BUCKET_LABELS = [
  "PF",
  "RB_NP_LO","RB_NP_MID","RB_NP_HI",
  "RB_PR_LO","RB_PR_MID","RB_PR_HI",
  "TT_NP_LO","TT_NP_MID","TT_NP_HI",
  "TT_PR_LO","TT_PR_MID","TT_PR_HI",
  "MONO_NP_LO","MONO_NP_MID","MONO_NP_HI",
  "MONO_PR_LO","MONO_PR_MID","MONO_PR_HI",
] as const;

export const HEROBOARD_LABELS = [
  "AIR","DRAW","PAIR","STRONG_PAIR","OVERPAIR",
  "STRONG_PAIR_DRAW","COMBO_DRAW","MADE_STRAIGHT_FLUSH"
] as const;

// Raccourci lisible pour une clé
export function prettyInfosetRow(kStr: string) {
  const f = unpackInfosetKeyDense(kStr);
  return {
    key: kStr,
    phase: PHASE_NAMES[f.phase] ?? String(f.phase),
    role: ROLE_NAMES[f.role] ?? String(f.role),
    hand: handLabel169(f.hand),
    board: BOARD_BUCKET_LABELS[f.board] ?? String(f.board),
    heroboard: HEROBOARD_LABELS[f.heroboard] ?? String(f.heroboard),
    pot: `bucket ${f.pot}`,
    ratio: `bucket ${f.ratio}`,
    spr: `bucket ${f.spr}`,
    raw: f,
  };
}
