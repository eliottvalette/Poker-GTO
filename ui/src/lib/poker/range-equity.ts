import { observerMarginals } from "./beliefs";
import { normalizeRanges, type Ranges } from "./hybrid";
import { SeededRNG, type HandState } from "./engine";
import { rank7 } from "./evaluator";

export type RangeEquity = { probability: number; standardError: number; samples: number };

/** Showdown win/tie shares, not betting EV or side-pot payout fractions. */
export function rangeEquities(hand: HandState, ranges: Ranges, observer: number,
                              samples = 1024, seed = 1947): Record<number, RangeEquity> {
  if (!Number.isInteger(samples) || samples < 2) throw new Error("Equity requires at least two samples");
  const factors = normalizeRanges(hand, ranges);
  const marginals = observerMarginals(hand, factors, observer);
  const unknown = Object.keys(factors).map(Number).filter(p => p !== observer).sort((a,b)=>a-b);
  const seats = Object.keys(factors).map(Number);
  const live = seats.filter(p => !hand.players[p].folded);
  if (!live.length) throw new Error("No live player for showdown equity");
  const totals = Object.fromEntries(seats.map(p=>[p,0]));
  const squares = {...totals}, rng = new SeededRNG(seed);
  function draw(rows: Ranges[number], blocked: Set<number>): [number, number] {
    const feasible = rows.filter(r=>!r.cards.some(c=>blocked.has(c)));
    const total = feasible.reduce((s,r)=>s+r.probability,0);
    if (!(total > 0)) throw new Error("No compatible private deal for equity");
    let u = rng.next()*total;
    for(const row of feasible) {u-=row.probability; if(u<=0) return row.cards;}
    return feasible[feasible.length-1].cards;
  }
  for(let sample=0;sample<samples;sample++) {
    const hands: Record<number,[number,number]> = {[observer]: hand.players[observer].cards};
    const used = new Set([...hand.board,...hands[observer]]);
    // First draw is the joint marginal; the second is its conditional factor.
    for(let i=0;i<unknown.length;i++) {
      const seat=unknown[i];
      hands[seat]=draw(i===0?marginals[seat]:factors[seat],used);
      hands[seat].forEach(c=>used.add(c));
    }
    const board=[...hand.board], deck=Array.from({length:52},(_,i)=>i).filter(c=>!used.has(c));
    while(board.length<5) {
      const index=Math.floor(rng.next()*deck.length);
      board.push(deck[index]); deck[index]=deck[deck.length-1]; deck.pop();
    }
    const scores=live.map(p=>rank7([...hands[p],...board])), best=Math.max(...scores);
    const winners=live.filter((_,i)=>scores[i]===best);
    for(const seat of winners) {const share=1/winners.length;totals[seat]+=share;squares[seat]+=share*share;}
  }
  return Object.fromEntries(seats.map(seat=>[seat, {probability:totals[seat]/samples,
    standardError:Math.sqrt(Math.max(0,squares[seat]-totals[seat]**2/samples)/(samples-1)/samples),samples}]));
}
