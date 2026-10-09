import { bestFive, decisiveCards, type BestFive } from "./evaluator";
import { EPS, type HandState } from "./engine";

export type ShowdownHighlight = { best: number[]; decisive: number[]; name: string; score: number };
const NAMES=["High card","Pair","Two pair","Three of a kind","Straight","Flush","Full house","Four of a kind","Straight flush"];

/** Highlight only winners of contested pots, not returned uncalled bets or folded holdings. */
export function showdownHighlights(hand: HandState): Record<number,ShowdownHighlight> {
  if(!hand.terminal || !hand.showdown || hand.board.length!==5) return {};
  const players=Object.values(hand.players), ranked:Record<number,BestFive>={};
  for(const p of players.filter(p=>!p.folded)) ranked[p.player_id]=bestFive([...hand.board,...p.cards]);
  const result:Record<number,ShowdownHighlight>={};
  for(const level of [...new Set(players.filter(p=>p.contribution>EPS).map(p=>p.contribution))]) {
    const contributors=players.filter(p=>p.contribution>=level-EPS);
    const eligible=contributors.filter(p=>!p.folded);
    if(contributors.length<2 || eligible.length<2) continue;
    const top=Math.max(...eligible.map(p=>ranked[p.player_id].score));
    for(const p of eligible.filter(p=>ranked[p.player_id].score===top)) {
      const best=ranked[p.player_id];
      const decisive=decisiveCards(best,eligible.filter(q=>q.player_id!==p.player_id).map(q=>ranked[q.player_id]));
      const previous=result[p.player_id];
      result[p.player_id]={best:best.cards,decisive:[...new Set([...(previous?.decisive??[]),...decisive])],name:NAMES[best.category],score:best.score};
    }
  }
  return result;
}
