// ui/src/lib/buckets.ts
export const POT_EDGES_BB = [0,1,2,3,4,5,6,8,10,12,16,20,24,32,40,48,64,80,96,128,160,192,256,320,Infinity];
export const RATIO_EDGES  = [0,0.05,0.125,0.25,0.5,1.0,2.0,Infinity];
export const SPR_EDGES    = [0,0.75,1.25,2.0,3.5,6.0,10.0,Infinity];

function bucketFromEdges(x: number, edges: number[]) {
  let i = 0;
  while (i+1 < edges.length && x >= edges[i+1]) i++;
  return Math.max(0, Math.min(i, edges.length-2));
}

export function qlogPotBucket(potBB: number) {
  return bucketFromEdges(Math.max(0, potBB), POT_EDGES_BB);
}
export function ratioBucket(toCallBB: number, potBB: number) {
  const r = Math.max(0, toCallBB) / Math.max(1, potBB);
  return bucketFromEdges(r, RATIO_EDGES);
}
export function sprBucket(effStackBB: number, potBB: number) {
  const r = Math.max(0, effStackBB) / Math.max(1, potBB);
  return bucketFromEdges(r, SPR_EDGES);
}
