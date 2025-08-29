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
