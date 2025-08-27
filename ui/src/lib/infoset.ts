// Décode la clé dense uint64 [phase:3 | role:2 | hand:8 | board:5 | pot:8 | tocall:8 | last:8 | raises:2]
const POS = { RAISES:0, LAST:2, TOCALL:10, POT:18, BOARD:26, HAND:31, ROLE:39, PHASE:41 } as const;
const MASK = { RAISES:(1<<2)-1, LAST:(1<<8)-1, TOCALL:(1<<8)-1, POT:(1<<8)-1, BOARD:(1<<5)-1, HAND:(1<<8)-1, ROLE:(1<<2)-1, PHASE:(1<<3)-1 } as const;

export type UnpackedKey = {
  phase:number; role:number; hand:number; board:number; pot:number; tocall:number; last:number; raises:number;
};

export function unpackInfosetKeyDense(kStr: string): UnpackedKey {
  const n = BigInt(kStr);
  const g = (pos:number, mask:number) => Number((n >> BigInt(pos)) & BigInt(mask));
  return {
    phase:  g(POS.PHASE,  MASK.PHASE),
    role:   g(POS.ROLE,   MASK.ROLE),
    hand:   g(POS.HAND,   MASK.HAND),
    board:  g(POS.BOARD,  MASK.BOARD),
    pot:    g(POS.POT,    MASK.POT),
    tocall: g(POS.TOCALL, MASK.TOCALL),
    last:   g(POS.LAST,   MASK.LAST),
    raises: g(POS.RAISES, MASK.RAISES),
  };
}
