/** Presentation quantities stay separate from the public factors used by search. */
import { handClassMasses } from "./beliefs";
import { type HandState } from "./engine";
import { type Ranges } from "./hybrid";
import { rank5 } from "./evaluator";
import { behaviorProbabilities } from "./behavior";
import { observe } from "./observation";
import { ACTION_IDS } from "./actions";

export type ReactionMix = {fold:number; passive:number; aggressive:number};
export type RangeDisplay = {
  uniformMass: Record<string,number>;
  feasibleCounts: Record<string,number>;
  composition: Record<number,Record<string,number>>;
  reactions: Record<number,Record<string,ReactionMix>>;
  actor: number | null;
  equities?: Record<number, import("./range-equity").RangeEquity>;
};

function category(cards:number[]): number {
  let best=-Infinity;
  for(let a=0;a<cards.length-4;a++) for(let b=a+1;b<cards.length-3;b++)
    for(let c=b+1;c<cards.length-2;c++) for(let d=c+1;d<cards.length-1;d++)
      for(let e=d+1;e<cards.length;e++) best=Math.max(best,rank5([cards[a],cards[b],cards[c],cards[d],cards[e]]));
  return Math.floor(best/15**5);
}
function hasDraw(cards:number[]):boolean {
  const suits=Array(4).fill(0); cards.forEach(c=>suits[c%4]++);
  if(suits.some(n=>n===4)) return true;
  const ranks=new Set(cards.map(c=>Math.floor(c/4)+2));
  if(ranks.has(14)) ranks.add(1);
  for(let low=1;low<=10;low++) if(Array.from({length:5},(_,i)=>low+i).filter(r=>ranks.has(r)).length===4) return true;
  return false;
}
export function rangeDisplay(hand:HandState, marginals:Ranges, observer:number, profiles:Record<number,string>, predictions?:Record<string,number[]>):RangeDisplay {
  const blocked=new Set([...hand.board,...hand.players[observer].cards]);
  const available=[];
  for(let a=0;a<52;a++) for(let b=a+1;b<52;b++) if(!blocked.has(a)&&!blocked.has(b)) available.push({cards:[a,b] as [number,number],probability:1});
  const feasibleCounts=handClassMasses(available);
  const uniformMass=Object.fromEntries(Object.entries(feasibleCounts).map(([k,n])=>[k,n/available.length]));
  const result:RangeDisplay={uniformMass,feasibleCounts,composition:{},reactions:{},actor:hand.terminal?null:hand.current_player};
  for(const [seat,rows] of Object.entries(marginals)) {
    if(Number(seat)===observer) continue;
    const groups:Record<string,number>={"Two pair+":0,"Pair":0,"Draw (unpaired)":0,"High card":0};
    const mixes:Record<string,ReactionMix>={};
    for(const row of rows) {
      if(hand.board.length>=3) {
        const cards=[...row.cards,...hand.board], made=category(cards);
        const group=made>=2?"Two pair+":made===1?"Pair":hand.board.length<5&&hasDraw(cards)?"Draw (unpaired)":"High card";
        groups[group]+=row.probability;
      }
      if(Number(seat)===result.actor) {
        if(!profiles[Number(seat)]) throw new Error(`Missing reaction profile for seat ${seat}`);
        const view=hand.clone(); view.players[Number(seat)].cards=row.cards;
        const p=predictions ? predictions[row.cards.join()] : behaviorProbabilities(observe(view),profiles[Number(seat)]);
        if(!p) throw new Error("Missing candidate reaction prediction");
        const label=Object.keys(handClassMasses([row]))[0];
        const mix=mixes[label]??={fold:0,passive:0,aggressive:0};
        ACTION_IDS.forEach((a,i)=>{mix[a==="FOLD"?"fold":a==="CHECK"||a==="CALL"?"passive":"aggressive"]+=row.probability*p[i];});
      }
    }
    if(hand.board.length>=3) result.composition[Number(seat)]=groups;
    if(Number(seat)===result.actor) {
      const masses=handClassMasses(rows);
      for(const [label,mix] of Object.entries(mixes)) for(const key of ["fold","passive","aggressive"] as const) mix[key]/=masses[label];
      result.reactions[Number(seat)]=mixes;
    }
  }
  return result;
}

/** Posterior mass: rounded-zero cells are gray; blue increases toward the displayed maximum. */
export function rangeColor(probability:number, maximum:number):string {
  if (Number((probability * 100).toFixed(1)) === 0) return "rgb(39,39,42)";
  const t=Math.min(1,Math.max(0,probability/maximum));
  const low=[35,45,62], high=[37,99,235];
  return `rgb(${low.map((c,i)=>Math.round(c+(high[i]-c)*t)).join(",")})`;
}
