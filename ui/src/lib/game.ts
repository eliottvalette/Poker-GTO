// ui/src/lib/game.ts
// 3-handed NLHE minimal engine en TypeScript, compatible avec la policy UI.

export type Phase = "PREFLOP"|"FLOP"|"TURN"|"RIVER"|"SHOWDOWN";
export type Action = "FOLD"|"CHECK"|"CALL"|"RAISE"|"ALL-IN";

export class Card {
  readonly rank: number; // 2..14
  readonly suit: number; // 0..3
  readonly id: number;   // 0..51
  constructor(rank: number, suit: number) {
    this.rank = rank;
    this.suit = suit;
    this.id = (rank - 2) * 4 + suit;
  }
  toString() {
    const R = {14:"A",13:"K",12:"Q",11:"J",10:"T",9:"9",8:"8",7:"7",6:"6",5:"5",4:"4",3:"3",2:"2"} as const;
    const S = {0:"♠",1:"♥",2:"♦",3:"♣"} as const;
    return `${R[this.rank as 2|3|4|5|6|7|8|9|10|11|12|13|14]}${S[this.suit as 0|1|2|3]}`;
  }
}

export class Deck {
  cards: Card[];
  constructor() {
    this.cards = [];
    for (let r=2;r<=14;r++) for (let s=0;s<4;s++) this.cards.push(new Card(r,s));
  }
  shuffle(rng=Math.random) {
    for (let i=this.cards.length-1;i>0;i--) {
      const j = Math.floor(rng()*(i+1));
      [this.cards[i],this.cards[j]]=[this.cards[j],this.cards[i]];
    }
  }
  pop(): Card {
    const c = this.cards.pop();
    if (!c) throw new Error("Deck empty");
    return c;
  }
  remove(ids: Set<number>) { this.cards = this.cards.filter(c=>!ids.has(c.id)); }
}

export class Player {
  name: string;
  role: 0|1|2; // 0=SB,1=BB,2=BTN
  stack: number;
  cards: Card[];
  is_active: boolean;
  has_folded: boolean;
  is_all_in: boolean;
  current_player_bet: number;
  total_bet: number;
  has_acted: boolean;
  constructor(name: string, role: 0|1|2, stack: number) {
    this.name = name;
    this.role = role;
    this.stack = stack;
    this.cards = [];
    this.is_active = true;
    this.has_folded = false;
    this.is_all_in = false;
    this.current_player_bet = 0;
    this.total_bet = 0;
    this.has_acted = false;
  }
}

export type GameInit = {
  stacks: [number,number,number];
  totalBets: [number,number,number];
  currentBets: [number,number,number];
  active: [boolean,boolean,boolean];
  hasActed: [boolean,boolean,boolean];
  mainPot: number;
  phase: Phase;
  community: Card[];
};

export class PokerGame {
  readonly numPlayers = 3;
  readonly small_blind = 1;
  readonly big_blind = 2;

  players: [Player,Player,Player];
  deck: Deck;
  community: Card[];
  main_pot: number;

  current_phase: Phase;
  current_role: 0|1|2;
  current_maximum_bet: number;
  number_raise_this_game_phase: number;
  last_raiser: number|null;
  last_raise_amount: number;

  initialStacks: Record<string, number>;
  net_stack_changes: Record<string, number>;
  final_stacks: Record<string, number>;

  constructor(init: GameInit) {
    this.players = [
      new Player("Player_0",0,init.stacks[0]),
      new Player("Player_1",1,init.stacks[1]),
      new Player("Player_2",2,init.stacks[2]),
    ];
    // états importés
    for (let i=0;i<3;i++){
      const p = this.players[i];
      p.is_active = init.active[i] ?? true;
      p.has_folded = !p.is_active;
      p.is_all_in = p.is_active && p.stack===0;
      p.current_player_bet = init.currentBets[i] ?? 0;
      p.total_bet = init.totalBets[i] ?? 0;
      p.has_acted = init.hasActed[i] ?? false;
    }

    this.community = [...init.community];
    this.main_pot = init.mainPot;
    this.current_phase = init.phase;
    this.number_raise_this_game_phase = 0;
    this.last_raiser = null;
    this.last_raise_amount = this.big_blind;
    this.current_role = 0;

    this.deck = new Deck();
    this.deck.shuffle();
    const known = new Set<number>(this.community.map(c=>c.id));
    this.deck.remove(known);

    this.current_maximum_bet = Math.max(...this.players.map(p=>p.current_player_bet));

    this.initialStacks = Object.fromEntries(this.players.map(p=>[p.name,p.stack]));
    this.net_stack_changes = Object.fromEntries(this.players.map(p=>[p.name,0]));
    this.final_stacks = Object.fromEntries(this.players.map(p=>[p.name,p.stack]));
  }

  deal_private() {
    for (const p of this.players) {
      if (p.is_active && !p.has_folded && p.cards.length===0) {
        p.cards = [this.deck.pop(), this.deck.pop()];
      }
    }
  }

  deal_blinds() {
    const [sb, bb] = [this.players[0], this.players[1]];
    // SB
    if (sb.stack >= this.small_blind) {
      sb.stack -= this.small_blind;
      sb.current_player_bet = this.small_blind;
      sb.total_bet = this.small_blind;
      this.main_pot += this.small_blind;
      sb.has_acted = false;
    } else {
      sb.current_player_bet = sb.stack;
      sb.total_bet = sb.stack;
      this.main_pot += sb.stack;
      sb.stack = 0; sb.is_all_in = true; sb.has_acted = true;
    }
    this.current_maximum_bet = this.small_blind;
    this.last_raise_amount = this.big_blind;
    this.current_role = 1; // next -> BB

    // BB
    if (bb.stack >= this.big_blind) {
      bb.stack -= this.big_blind;
      bb.current_player_bet = this.big_blind;
      bb.total_bet = this.big_blind;
      this.main_pot += this.big_blind;
      bb.has_acted = false;
    } else {
      bb.current_player_bet = bb.stack;
      bb.total_bet = bb.stack;
      this.main_pot += bb.stack;
      bb.stack = 0; bb.is_all_in = true; bb.has_acted = true;
    }
    this.current_maximum_bet = this.big_blind;
    this.current_role = 2; // BTN parle après blinds
  }

  next_player() {
    const start = this.current_role;
    do {
      this.current_role = ((this.current_role + 1) % 3) as 0|1|2;
      const p = this.players[this.current_role];
      if (p.is_active && !p.has_folded && !p.is_all_in) return;
    } while (this.current_role !== start);
    // si on boucle, on laisse tel quel, checkPhaseCompletion traitera
  }

  update_available_actions(p: Player): Action[] {
    if (this.current_phase==="SHOWDOWN" || p.is_all_in) return [];
    const mask: Record<Action, boolean> = {FOLD:true,CHECK:true,CALL:true,RAISE:true,"ALL-IN":true};
    if (p.current_player_bet < this.current_maximum_bet) mask.CHECK = false;
    if (mask.CHECK) mask.FOLD = false;

    const toCall = this.current_maximum_bet - p.current_player_bet;
    if (toCall <= 0) mask.CALL = false;
    else if (toCall >= p.stack) { mask.CALL = false; mask["ALL-IN"]=true; }

    const minRaiseTo = this.current_maximum_bet===0 ? this.big_blind
      : this.current_maximum_bet + Math.max(this.last_raise_amount, this.big_blind);
    const addReq = minRaiseTo - p.current_player_bet;
    if (addReq <= 0 || p.stack < addReq || this.number_raise_this_game_phase >= 4) mask.RAISE = false;

    return (["FOLD","CHECK","CALL","RAISE","ALL-IN"] as Action[]).filter(a=>mask[a]);
  }

  private deal_community_for_phase() {
    if (this.current_phase==="FLOP") {
      this.community.push(this.deck.pop(), this.deck.pop(), this.deck.pop());
    } else if (this.current_phase==="TURN" || this.current_phase==="RIVER") {
      this.community.push(this.deck.pop());
    }
  }

  advance_phase() {
    this.last_raiser = null;
    if (this.current_phase==="PREFLOP") this.current_phase="FLOP";
    else if (this.current_phase==="FLOP") this.current_phase="TURN";
    else if (this.current_phase==="TURN") this.current_phase="RIVER";
    this.number_raise_this_game_phase = 0;
    this.last_raise_amount = this.big_blind;
    this.deal_community_for_phase();
    this.current_maximum_bet = 0;
    for (const p of this.players) {
      if (!p.is_active) continue;
      p.current_player_bet = 0;
      if (!p.has_folded && !p.is_all_in) p.has_acted = false;
    }
    // postflop: SB parle en premier
    this.current_role = 0;
    if (!this.players[0].is_active || this.players[0].has_folded || this.players[0].is_all_in) this.next_player();
  }

  process_action(p: Player, action: Action) {
    if (p !== this.players[this.current_role]) throw new Error("Not player's turn");
    const legal = this.update_available_actions(p);
    if (!legal.includes(action)) throw new Error(`Illegal action ${action}`);

    if (action==="FOLD") {
      p.has_folded = true;
    } else if (action==="CHECK") {
      // no-op
    } else if (action==="CALL") {
      const toCall = this.current_maximum_bet - p.current_player_bet;
      if (toCall > p.stack) throw new Error("Call > stack (should have been ALL-IN only)");
      p.stack -= toCall; p.current_player_bet += toCall; p.total_bet += toCall; this.main_pot += toCall;
      if (p.stack===0) p.is_all_in = true;
    } else if (action==="RAISE") {
      const prevMax = this.current_maximum_bet;
      const minRaiseTo = prevMax===0 ? this.big_blind : prevMax + Math.max(this.last_raise_amount, this.big_blind);
      const addReq = Math.max(0, minRaiseTo - p.current_player_bet);
      if (addReq<=0 || addReq>p.stack) throw new Error("Invalid raise");
      p.stack -= addReq; p.current_player_bet = minRaiseTo; p.total_bet += addReq; this.main_pot += addReq;
      this.number_raise_this_game_phase += 1;
      this.last_raiser = this.current_role;
      this.last_raise_amount = minRaiseTo - prevMax;
      this.current_maximum_bet = minRaiseTo;
      if (p.stack===0) p.is_all_in = true;
    } else if (action==="ALL-IN") {
      const prevMax = this.current_maximum_bet;
      const add = p.stack;
      const newTo = p.current_player_bet + add;
      const delta = newTo - prevMax;
      if (delta > 0) {
        this.current_maximum_bet = newTo;
        if (delta >= Math.max(this.last_raise_amount, this.big_blind)) {
          this.number_raise_this_game_phase += 1;
          this.last_raiser = this.current_role;
          this.last_raise_amount = delta;
        }
      }
      p.stack = 0; p.current_player_bet = newTo; p.total_bet += add; this.main_pot += add; p.is_all_in = true;
    }
    p.has_acted = true;
    this.check_phase_completion();
    if (this.current_phase!=="SHOWDOWN") this.next_player();
  }

  private everyoneAllInOrCapped(): boolean {
    const live = this.players.filter(p=>p.is_active && !p.has_folded);
    if (live.length<=1) return true;
    const atMostOneLive = live.filter(p=>!p.is_all_in).length <= 1;
    const allCapped = live.every(p=>p.is_all_in || p.current_player_bet===this.current_maximum_bet);
    return atMostOneLive && allCapped;
  }

  check_phase_completion() {
    const live = this.players.filter(p=>p.is_active && !p.has_folded);
    if (live.length===1) return this.handle_showdown();

    if (live.some(p=>p.is_all_in) && this.everyoneAllInOrCapped() && live.length>1) {
      while (this.community.length<5) this.community.push(this.deck.pop());
      return this.handle_showdown();
    }
    if (live.every(p=>p.has_acted) && live.every(p=>p.is_all_in || p.current_player_bet===this.current_maximum_bet)) {
      if (this.current_phase==="RIVER") return this.handle_showdown();
      this.advance_phase();
    }
  }

  private hand_rank_7(cards: Card[]): number[] {
    // Renvoie un vecteur triable: [cat, tiebreakers...], cat: 8=StraightFlush,7=Four,6=Full,5=Flush,4=Straight,3=Trips,2=TwoPair,1=Pair,0=High
    // Implémentation naïve mais correcte pour 7 cartes.
    const ranks = cards.map(c=>c.rank).sort((a,b)=>b-a);
    const suits = cards.map(c=>c.suit);
    const byRank: Record<number, number> = {};
    for (const r of ranks) byRank[r]=(byRank[r]||0)+1;
    // treat A as 1 for A-5 straight
    const uniq = Array.from(new Set(ranks));
    const uniqWithWheel = uniq.includes(14) ? [...uniq, 1] : uniq;
    const isStraight = (): number => {
      let run=1, bestHigh=0;
      for (let i=0;i<uniqWithWheel.length-1;i++){
        if (uniqWithWheel[i]-1===uniqWithWheel[i+1]) { run++; if (run>=5) bestHigh = Math.max(bestHigh, uniqWithWheel[i-3] ?? uniqWithWheel[i]); }
        else run=1;
      }
      return bestHigh; // 0 si pas de straight, sinon high card de la straight (avec 1 pour wheel bas)
    };
    const suitCounts: Record<number, number> = {};
    for (const s of suits) suitCounts[s]=(suitCounts[s]||0)+1;
    const flushSuit = Object.entries(suitCounts).find(([,n])=>n>=5)?.[0];
    const flushCards = flushSuit===undefined ? [] : cards.filter(c=>c.suit===Number(flushSuit)).sort((a,b)=>b.rank-a.rank);
    // Straight flush
    let sfHigh = 0;
    if (flushCards.length>=5) {
      const fr = flushCards.map(c=>c.rank);
      const uniqF = Array.from(new Set(fr));
      const uniqFWheel = uniqF.includes(14) ? [...uniqF, 1] : uniqF;
      let run=1;
      for (let i=0;i<uniqFWheel.length-1;i++){
        if (uniqFWheel[i]-1===uniqFWheel[i+1]) { run++; if (run>=5) sfHigh = Math.max(sfHigh, uniqFWheel[i-3] ?? uniqFWheel[i]); }
        else run=1;
      }
      if (sfHigh>0) return [8, sfHigh]; // Straight flush
    }
    // Quads / Trips / Pairs
    const groups = Object.entries(byRank).map(([r,c])=>({r:+r,c})).sort((a,b)=> b.c===a.c ? b.r-a.r : b.c-a.c);
    if (groups[0]?.c===4) {
      const kicker = ranks.find(r=>r!==groups[0].r)!;
      return [7, groups[0].r, kicker];
    }
    if (groups[0]?.c===3 && (groups[1]?.c||0)>=2) {
      return [6, groups[0].r, groups[1].r]; // full
    }
    if (flushCards.length>=5) {
      return [5, ...flushCards.slice(0,5).map(c=>c.rank)];
    }
    const straightHigh = isStraight();
    if (straightHigh>0) {
      return [4, straightHigh];
    }
    if (groups[0]?.c===3) {
      const kickers = ranks.filter(r=>r!==groups[0].r).slice(0,2);
      return [3, groups[0].r, ...kickers];
    }
    if (groups[0]?.c===2 && groups[1]?.c===2) {
      const [hi,lo] = [groups[0].r, groups[1].r].sort((a,b)=>b-a);
      const kicker = ranks.find(r=>r!==hi && r!==lo)!;
      return [2, hi, lo, kicker];
    }
    if (groups[0]?.c===2) {
      const kickers = ranks.filter(r=>r!==groups[0].r).slice(0,3);
      return [1, groups[0].r, ...kickers];
    }
    return [0, ...ranks.slice(0,5)];
  }

  handle_showdown() {
    this.current_phase = "SHOWDOWN";
    this.current_maximum_bet = 0;
    // compléter board
    while (this.community.length<5) this.community.push(this.deck.pop());
    const live = this.players.filter(p=>p.is_active && !p.has_folded);
    if (live.length===1) {
      live[0].stack += this.main_pot; this.main_pot=0;
    } else {
      // side pots simples par niveaux de mise
      const contrib = new Map<Player, number>(this.players.map(p=>[p,p.total_bet]));
      const levels = Array.from(new Set(Array.from(contrib.values()))).sort((a,b)=>a-b);
      let prev = 0;
      const board = this.community;
      const scores = new Map<Player, number[]>(live.map(p=>[p, this.hand_rank_7([...board, ...p.cards])]));
      for (const L of levels) {
        const cap = L - prev; if (cap<=0) { prev=L; continue; }
        const eligAll = this.players.filter(p=>(contrib.get(p)??0) >= L);
        const eligLive = eligAll.filter(p=>live.includes(p));
        const potAmt = cap * eligAll.length;
        if (potAmt>0 && eligLive.length>0) {
          eligLive.sort((a,b)=>{
            const sa = scores.get(a)!, sb = scores.get(b)!;
            // compare lexicographically
            for (let i=0;i<Math.max(sa.length,sb.length);i++){
              const va = sa[i]??-1, vb = sb[i]??-1;
              if (va!==vb) return vb-va;
            }
            return 0;
          });
          const best = scores.get(eligLive[0])!;
          const winners = eligLive.filter(p=>{
            const s = scores.get(p)!;
            if (s.length!==best.length) return false;
            for (let i=0;i<s.length;i++) if (s[i]!==best[i]) return false;
            return true;
          });
          const share = potAmt / winners.length;
          for (const w of winners) w.stack += share;
          this.main_pot -= potAmt;
        }
        prev = L;
      }
      if (this.main_pot < 1e-9) this.main_pot = 0;
    }
    this.net_stack_changes = Object.fromEntries(this.players.map(p=>[p.name, p.stack - (this.initialStacks[p.name]??0)]));
    this.final_stacks = Object.fromEntries(this.players.map(p=>[p.name, p.stack]));
  }
}

// -------- Infoset builder minimal pour policy --------
const PHASE_TO_ID: Record<Phase, number> = {PREFLOP:0,FLOP:1,TURN:2,RIVER:3,SHOWDOWN:4};
export function boardBucket(board: Card[]): number {
  const n = board.length; if (n===0) return 0;
  const ranks = board.map(c=>c.rank);
  const suits = board.map(c=>c.suit);
  const maxSuit = Math.max(...Array.from(new Set(suits)).map(s=>suits.filter(x=>x===s).length));
  const suitTex = ((n===3&&maxSuit===3)||(n===4&&maxSuit>=4)||(n===5&&maxSuit>=5)) ? 2 : ((n>=3&&maxSuit===n-1)?1:0);
  const paired = ranks.some(r=>ranks.filter(x=>x===r).length>=2) ? 1 : 0;
  const hi = Math.max(...ranks); const hiClass = hi>=12?2:hi>=10?1:0;
  return suitTex*6 + paired*3 + hiClass;
}
export function heroVsBoardBucket(hero: Player, board: Card[]): number {
  if (board.length===0) return 0;
  const rB = board.map(c=>c.rank), sB = board.map(c=>c.suit);
  const rH = hero.cards.map(c=>c.rank), sH = hero.cards.map(c=>c.suit);
  let pairType = 0;
  if (rH.some(r=>rB.includes(r))) {
    const hiH = Math.max(...rH), hiB = Math.max(...rB), secB = [...rB].sort((a,b)=>a-b)[rB.length-2]??-1;
    if (rB.includes(hiH)) pairType = hiH>=hiB?3 : hiH>=secB?2 : 1;
  } else if (rH[0]===rH[1]) {
    pairType = rH[0] > Math.max(...rB) ? 4 : 1;
  }
  let flushDraw = 0;
  for (const s of new Set(sH)) {
    const need = 5 - sB.filter(x=>x===s).length;
    if (need<=2) flushDraw = need===1?2:1;
  }
  let straight_draw = 0;
  const all = Array.from(new Set([...rB, ...rH])).sort((a,b)=>a-b);
  for (let st=2; st<=10; st++){
    const window = new Set([st,st+1,st+2,st+3,st+4]);
    const overlap = all.filter(x=>window.has(x)).length;
    if (overlap===5) straight_draw = 3;
    else if (overlap===4) straight_draw = Math.max(straight_draw,2);
    else if (overlap===3) straight_draw = Math.max(straight_draw,1);
  }
  if (flushDraw===0 && straight_draw===0 && pairType===0) return 0;
  if (pairType>=3 && (flushDraw||straight_draw)) return 7;
  if (flushDraw===2 && straight_draw>=2) return 8;
  if (straight_draw===3 || flushDraw===2) return 9;
  if (pairType===4) return 6;
  if (pairType>0) return 5;
  if (straight_draw>0 || flushDraw>0) return 4;
  return 0;
}

const POS = { HEROBOARD:0, SPR:4, RATIO:12, POT:20, BOARD:28, HAND:33, ROLE:41, PHASE:43 } as const;
const MASK= { HEROBOARD:(1<<4)-1, SPR:(1<<8)-1, RATIO:(1<<8)-1, POT:(1<<8)-1,
              BOARD:(1<<5)-1, HAND:(1<<8)-1, ROLE:(1<<2)-1, PHASE:(1<<3)-1 } as const;

export function packU64(fields: Record<keyof typeof POS, number>): bigint {
  let v = BigInt(0);
  const keys = Object.keys(POS) as (keyof typeof POS)[];
  for (const k of keys) {
    const mask = MASK[k];
    const pos  = POS[k];
    v |= (BigInt(fields[k] & mask) << BigInt(pos));
  }
  return v;
}

// 169 hand index
export function hand169Index(c1: Card, c2: Card): number {
  const order = [14,13,12,11,10,9,8,7,6,5,4,3,2];
  const idx = new Map(order.map((r,i)=>[r,i]));
  const i = idx.get(c1.rank)!; const j = idx.get(c2.rank)!;
  const suited = c1.suit===c2.suit;
  if (i===j) return i*13+i;
  if (suited) return Math.min(i,j)*13 + Math.max(i,j);
  return Math.max(i,j)*13 + Math.min(i,j);
}

// buckets for pot, ratio, spr
const POT_EDGES = [0,1,2,3,4,5,6,8,10,12,16,20,24,32,40,48,64,80,96,128,160,192,256,320,Infinity];
const RATIO_EDGES= [0,0.05,0.125,0.25,0.5,1.0,2.0,Infinity];
const SPR_EDGES  = [0,0.75,1.25,2.0,3.5,6.0,10.0,Infinity];
function bucketFromEdges(x: number, edges: number[]) {
  let i=0; while (i+1<edges.length && x>=edges[i+1]) i++; return Math.max(0, Math.min(i, edges.length-2));
}
export function qlogPotBucket(potBB: number) { return bucketFromEdges(Math.max(0,potBB), POT_EDGES); }
export function ratioBucket(toCallBB: number, potBB: number) {
  const r = Math.max(0,toCallBB)/Math.max(1,potBB); return bucketFromEdges(r, RATIO_EDGES);
}
export function sprBucket(effStackBB: number, potBB: number) {
  const r = Math.max(0,effStackBB)/Math.max(1,potBB); return bucketFromEdges(r, SPR_EDGES);
}

export function buildInfosetKeyFast(game: PokerGame, hero: Player): string {
  const phase_id = PHASE_TO_ID[game.current_phase];
  const role_id = hero.role;
  const hand = hand169Index(hero.cards[0], hero.cards[1]);
  const board = boardBucket(game.community);
  const pot = game.main_pot;
  const tocall = Math.max(0, game.current_maximum_bet - hero.current_player_bet);
  const live = game.players.filter(p=>p.is_active && !p.has_folded);
  const eff = Math.min(...live.filter(p=>p!==hero).map(p=>Math.min(hero.stack, p.stack)), hero.stack);
  const pot_q = qlogPotBucket(pot);
  const ratio_q = ratioBucket(tocall, pot);
  const spr_q = sprBucket(eff, pot);
  const hb = heroVsBoardBucket(hero, game.community);
  const n = packU64({PHASE:phase_id, ROLE:role_id, HAND:hand, BOARD:board, POT:pot_q, RATIO:ratio_q, SPR:spr_q, HEROBOARD:hb});
  return n.toString();
}
