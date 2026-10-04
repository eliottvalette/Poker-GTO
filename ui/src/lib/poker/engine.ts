/** Browser no-limit Hold'em/tournament rules, parity-tested against the Python training engine. */
import { rank7 } from "./evaluator";
import { applyAction, type ActionCategory } from "./actions";

export const EPS = 1e-9;
export const STREETS = ["PREFLOP", "FLOP", "TURN", "RIVER"] as const;
export type Street = typeof STREETS[number];
export type Position = "BTN" | "SB" | "BB";
export type BlindLevel = { small: number; big: number };
export type HandPlayer = { player_id: number; stack: number; position: Position; cards: [number, number];
  street_bet: number; contribution: number; folded: boolean; acted_at: number | null };
export type ActionEvent = { street: Street; player_id: number; position: Position; action: ActionCategory | "BLIND";
  amount_to: number; amount_added: number; pot_before: number; pot_after: number; highest_before: number };

function ids(record: Record<number, unknown>): number[] { return Object.keys(record).map(Number); }
function sum(values: number[]): number { return values.reduce((a, b) => a + b, 0); }
function close(a: number, b: number): boolean { return Math.abs(a - b) <= Math.max(EPS, 1e-9 * Math.max(Math.abs(a), Math.abs(b))); }
function validateBlinds(blinds: BlindLevel): void {
  if (!Number.isFinite(blinds.small) || !Number.isFinite(blinds.big) || !(0 < blinds.small && blinds.small < blinds.big)) {
    throw new Error(`Invalid blinds: ${JSON.stringify(blinds)}; expected 0 < SB < BB`);
  }
}
function validateSeats(stacks: Record<number, number>): void {
  if (ids(stacks).some(i => !Number.isInteger(i) || i < 0 || i > 2)) throw new Error(`Expected player IDs 0..2: ${JSON.stringify(stacks)}`);
}

/** Explicit hand-index schedule; chip values are never rescaled between levels. */
export class BlindStage {
  readonly first_hand: number;
  readonly blinds: Readonly<BlindLevel>;
  constructor(firstHand: number, blinds: BlindLevel) {
    if (!Number.isInteger(firstHand) || firstHand < 1) throw new Error(`Blind stage first_hand must be positive: ${firstHand}`);
    validateBlinds(blinds);
    this.first_hand = firstHand;
    this.blinds = Object.freeze({ ...blinds });
    Object.freeze(this);
  }
}

export class BlindSchedule {
  readonly stages: readonly BlindStage[];
  constructor(stages: readonly BlindStage[]) {
    if (!stages.length || stages[0].first_hand !== 1) throw new Error("Blind schedule must begin at hand 1");
    this.stages = Object.freeze(stages.map(stage => new BlindStage(stage.first_hand, stage.blinds)));
    for (let i = 1; i < this.stages.length; i++) {
      const previous = this.stages[i - 1], stage = this.stages[i];
      if (stage.first_hand <= previous.first_hand) throw new Error("Blind stage first_hand values must strictly increase");
      if (stage.blinds.small < previous.blinds.small || stage.blinds.big < previous.blinds.big
          || (stage.blinds.small === previous.blinds.small && stage.blinds.big === previous.blinds.big)) {
        throw new Error("Blind stages must be non-decreasing with at least one blind increasing");
      }
    }
    Object.freeze(this);
  }
  static fixed(blinds: BlindLevel = { small: 0.5, big: 1 }): BlindSchedule {
    return new BlindSchedule([new BlindStage(1, blinds)]);
  }
  forHand(handNumber: number): [number, BlindLevel] {
    if (!Number.isInteger(handNumber) || handNumber < 1) throw new Error(`Hand number must be positive: ${handNumber}`);
    let index = 0;
    for (let i = 1; i < this.stages.length && this.stages[i].first_hand <= handNumber; i++) index = i;
    return [index, { ...this.stages[index].blinds }];
  }
}

/** Explicit simulation preset, not an official Expresso blind timetable. */
export const DEFAULT_SIMULATION_SCHEDULE = new BlindSchedule([
  new BlindStage(1, { small: 0.5, big: 1 }), new BlindStage(11, { small: 1, big: 2 }),
  new BlindStage(21, { small: 2, big: 4 }), new BlindStage(31, { small: 4, big: 8 }),
  new BlindStage(41, { small: 8, big: 16 }), new BlindStage(51, { small: 16, big: 32 }),
]);

/** Cloneable deterministic PRNG; supplied-deck parity does not depend on Python's RNG algorithm. */
export class SeededRNG {
  private state: number;
  constructor(seed: number) {
    if (!Number.isSafeInteger(seed)) throw new Error(`Seed must be a safe integer: ${seed}`);
    this.state = seed >>> 0;
  }
  next(): number {
    this.state = (this.state + 0x6D2B79F5) >>> 0;
    let value = this.state;
    value = Math.imul(value ^ value >>> 15, value | 1);
    value ^= value + Math.imul(value ^ value >>> 7, value | 61);
    return ((value ^ value >>> 14) >>> 0) / 4294967296;
  }
  clone(): SeededRNG { const rng = new SeededRNG(0); rng.state = this.state; return rng; }
  shuffle<T>(array: T[]): void {
    for (let i = array.length - 1; i > 0; i--) {
      const j = Math.floor(this.next() * (i + 1));
      [array[i], array[j]] = [array[j], array[i]];
    }
  }
}

export class HandState {
  players: Record<number, HandPlayer> = {};
  button = 0;
  hand_number = 1;
  blind_level_index = 0;
  blinds: BlindLevel = { small: 0.5, big: 1 };
  deck: number[] = [];
  total_chips = 0;
  initial_stacks: Record<number, number> = {};
  board: number[] = [];
  street: Street = "PREFLOP";
  pot = 0;
  highest = 1;
  last_full_raise = 1;
  pending = new Set<number>();
  current_player: number | null = null;
  history: ActionEvent[] = [];
  terminal = false;
  showdown = false;
  awards: Record<number, number> = {};

  static start(stacks: Record<number, number>, button: number, rng: SeededRNG,
    blinds: BlindLevel = { small: 0.5, big: 1 }, deck?: number[],
    metadata: { hand_number: number; blind_level_index: number } = { hand_number: 1, blind_level_index: 0 }): HandState {
    validateSeats(stacks);
    validateBlinds(blinds);
    if (!Number.isInteger(metadata.hand_number) || metadata.hand_number < 1
        || !Number.isInteger(metadata.blind_level_index) || metadata.blind_level_index < 0) {
      throw new Error(`Invalid hand metadata: ${JSON.stringify(metadata)}`);
    }
    const seats = ids(stacks);
    if (![2, 3].includes(seats.length) || !seats.includes(button)) throw new Error(`Expected 2/3 active seats and live button: ${JSON.stringify(stacks)}, ${button}`);
    if (Object.values(stacks).some(s => !Number.isFinite(s) || s <= 0)) throw new Error(`Active stacks must be finite and positive: ${JSON.stringify(stacks)}`);
    const cards = deck === undefined ? Array.from({ length: 52 }, (_, i) => i) : [...deck];
    if (cards.length !== 52 || new Set(cards).size !== 52 || cards.some(c => !Number.isInteger(c) || c < 0 || c >= 52)) throw new Error("Deck must be a permutation of card IDs 0..51");
    if (deck === undefined) rng.shuffle(cards);
    const order = [...seats.slice(seats.indexOf(button)), ...seats.slice(0, seats.indexOf(button))];
    const roles: Position[] = seats.length === 2 ? ["SB", "BB"] : ["BTN", "SB", "BB"];
    const hand = new HandState();
    hand.hand_number = metadata.hand_number; hand.blind_level_index = metadata.blind_level_index;
    hand.button = button; hand.blinds = { ...blinds }; hand.deck = cards;
    hand.initial_stacks = { ...stacks }; hand.total_chips = sum(Object.values(stacks));
    hand.highest = blinds.big; hand.last_full_raise = blinds.big;
    for (const i of seats) hand.players[i] = { player_id: i, stack: stacks[i], position: roles[order.indexOf(i)],
      cards: [hand.draw(), hand.draw()], street_bet: 0, contribution: 0, folded: false, acted_at: null };
    for (const [role, amount] of [["SB", blinds.small], ["BB", blinds.big]] as const) {
      const player = Object.values(hand.players).find(p => p.position === role)!;
      const before = hand.pot;
      hand.pay(player, Math.min(player.stack, amount));
      hand.history.push({ street: "PREFLOP", player_id: player.player_id, position: role, action: "BLIND",
        amount_to: player.street_bet, amount_added: player.street_bet, pot_before: before, pot_after: hand.pot, highest_before: 0 });
    }
    hand.pending = new Set(seats.filter(i => hand.players[i].stack > EPS));
    hand.progress(hand.previous(button));
    hand.assertInvariants();
    return hand;
  }
  clone(): HandState {
    const clone = Object.assign(new HandState(), this);
    clone.players = Object.fromEntries(Object.entries(this.players).map(([i, p]) => [i, { ...p, cards: [...p.cards] as [number, number] }]));
    clone.blinds = { ...this.blinds }; clone.deck = [...this.deck]; clone.initial_stacks = { ...this.initial_stacks };
    clone.board = [...this.board]; clone.pending = new Set(this.pending); clone.history = this.history.map(e => ({ ...e })); clone.awards = { ...this.awards };
    return clone;
  }
  previous(seat: number): number { return this.adjacent(seat, -1); }
  next(seat: number): number { return this.adjacent(seat, 1); }
  private adjacent(seat: number, direction: number): number {
    const seats = ids(this.players), offset = seats.indexOf(seat);
    if (offset < 0) throw new Error(`Unknown seat ${seat}: ${seats}`);
    return seats[(offset + direction + seats.length) % seats.length];
  }
  get actor(): HandPlayer {
    if (this.terminal || this.current_player === null) throw new Error("Terminal hand has no acting player");
    return this.players[this.current_player];
  }
  toCall(player: HandPlayer = this.actor): number {
    const opponents = Object.values(this.players).filter(p => p.player_id !== player.player_id && !p.folded);
    const callable = opponents.some(p => p.stack > EPS) ? this.highest
      : Math.min(this.highest, Math.max(0, ...opponents.map(p => p.street_bet)));
    return Math.max(0, callable - player.street_bet);
  }
  get minRaiseTo(): number { return this.highest + this.last_full_raise; }
  canRaise(): boolean {
    const p = this.actor;
    return Object.values(this.players).some(q => q.player_id !== p.player_id && !q.folded && q.stack > EPS)
      && (p.acted_at === null || this.highest - p.acted_at >= this.last_full_raise - EPS)
      && p.stack + p.street_bet > this.highest + EPS;
  }
  private draw(): number { const card = this.deck.pop(); if (card === undefined) throw new Error("Deck exhausted"); return card; }
  private pay(p: HandPlayer, amount: number): void {
    if (!Number.isFinite(amount) || amount < -EPS || amount > p.stack + EPS) throw new Error(`Invalid payment ${amount}; player ${p.player_id} stack=${p.stack}`);
    p.stack -= amount; p.street_bet += amount; p.contribution += amount; this.pot += amount;
  }
  act(category: string, amountTo: number | null = null): void {
    const p = this.actor, call = this.toCall(p), highestBefore = this.highest, before = this.pot;
    if (category !== "RAISE" && amountTo !== null) throw new Error(`${category} must not specify amount_to=${amountTo}`);
    if (category === "FOLD") {
      if (call <= EPS) throw new Error("FOLD is excluded when CHECK is available");
      p.folded = true;
    } else if (category === "CHECK") {
      if (call > EPS) throw new Error(`Cannot check; to_call=${call}`);
    } else if (category === "CALL") {
      if (call <= EPS) throw new Error(`Cannot call; to_call=${call}`);
      this.pay(p, Math.min(call, p.stack));
    } else if (category === "RAISE") {
      if (amountTo === null || !Number.isFinite(amountTo) || !this.canRaise()) throw new Error(`Invalid raise ${amountTo}; raising rights=${this.canRaise()}`);
      const maximum = p.stack + p.street_bet;
      if (amountTo <= this.highest + EPS || amountTo > maximum + EPS) throw new Error(`Raise-to ${amountTo} must be in (${this.highest}, ${maximum}]`);
      const full = amountTo >= this.minRaiseTo - EPS;
      if (!full && !close(amountTo, maximum)) throw new Error(`Raise-to ${amountTo} below minimum ${this.minRaiseTo}; only all-in allowed`);
      this.pay(p, amountTo - p.street_bet); this.highest = amountTo;
      if (full) {
        this.last_full_raise = amountTo - highestBefore;
        this.pending = new Set(ids(this.players).filter(i => i !== p.player_id && !this.players[i].folded && this.players[i].stack > EPS));
      } else {
        for (const q of Object.values(this.players)) if (q.player_id !== p.player_id && !q.folded && q.stack > EPS && q.street_bet < this.highest - EPS) this.pending.add(q.player_id);
      }
    } else throw new Error(`Unknown engine category ${JSON.stringify(category)}; expected FOLD/CHECK/CALL/RAISE`);
    p.acted_at = this.highest; this.pending.delete(p.player_id);
    this.history.push({ street: this.street, player_id: p.player_id, position: p.position, action: category as ActionCategory,
      amount_to: p.street_bet, amount_added: this.pot - before, pot_before: before, pot_after: this.pot, highest_before: highestBefore });
    this.progress(p.player_id); this.assertInvariants();
  }
  private progress(after: number): void {
    const live = Object.values(this.players).filter(p => !p.folded);
    if (live.length === 1) { this.settle(); return; }
    const able = live.filter(p => p.stack > EPS);
    if (able.length <= 1) this.pending = new Set(able.filter(p => this.toCall(p) > EPS).map(p => p.player_id));
    if (this.pending.size) {
      let seat = after;
      for (let n = 0; n < ids(this.players).length; n++) { seat = this.next(seat); if (this.pending.has(seat)) { this.current_player = seat; return; } }
      throw new Error(`Pending seats not in hand: ${[...this.pending]}`);
    }
    if (able.length <= 1 || this.street === "RIVER") {
      if (live.length > 1) while (this.board.length < 5) this.board.push(this.draw());
      this.settle(); return;
    }
    this.street = STREETS[STREETS.indexOf(this.street) + 1];
    for (let n = 0; n < (this.street === "FLOP" ? 3 : 1); n++) this.board.push(this.draw());
    this.highest = 0; this.last_full_raise = this.blinds.big;
    Object.values(this.players).forEach(p => { p.street_bet = 0; p.acted_at = null; });
    this.pending = new Set(able.map(p => p.player_id)); this.progress(this.button);
  }
  private settle(): void {
    const live = Object.values(this.players).filter(p => !p.folded);
    this.showdown = live.length > 1;
    this.awards = Object.fromEntries(ids(this.players).map(i => [i, 0]));
    if (live.length === 1) this.awards[live[0].player_id] = this.pot;
    else {
      if (this.board.length !== 5) throw new Error(`Showdown requires five board cards: ${this.board}`);
      const ranks = Object.fromEntries(live.map(p => [p.player_id, rank7([...p.cards, ...this.board])]));
      let previous = 0;
      const levels = [...new Set(Object.values(this.players).filter(p => p.contribution > EPS).map(p => p.contribution))].sort((a, b) => a - b);
      for (const level of levels) {
        const contributors = Object.values(this.players).filter(p => p.contribution >= level - EPS);
        const amount = (level - previous) * contributors.length;
        let winners = contributors;
        if (contributors.length !== 1) {
          const eligible = contributors.filter(p => !p.folded);
          if (!eligible.length) throw new Error(`No eligible player for side pot at ${level}`);
          const best = Math.max(...eligible.map(p => ranks[p.player_id]));
          winners = eligible.filter(p => ranks[p.player_id] === best);
        }
        for (const p of winners) this.awards[p.player_id] += amount / winners.length;
        previous = level;
      }
    }
    if (!close(sum(Object.values(this.awards)), this.pot)) throw new Error(`Pot distribution mismatch: pot=${this.pot}, awards=${JSON.stringify(this.awards)}`);
    for (const i of ids(this.awards)) this.players[i].stack += this.awards[i];
    this.pot = 0; this.terminal = true; this.pending.clear(); this.current_player = null;
  }
  utility(player: number): number {
    if (!this.terminal || !this.players[player]) throw new Error(`Hand utility requires terminal hand and valid player: ${player}`);
    return this.players[player].stack - this.initial_stacks[player];
  }
  assertInvariants(): void {
    if (!Number.isInteger(this.hand_number) || this.hand_number < 1
        || !Number.isInteger(this.blind_level_index) || this.blind_level_index < 0) throw new Error("Invalid hand metadata");
    if (!STREETS.includes(this.street) || (!this.terminal && this.board.length !== [0, 3, 4, 5][STREETS.indexOf(this.street)])) throw new Error(`Invalid street/board contract: ${this.street}, ${this.board}`);
    if (!this.players[this.button] || ![2, 3].includes(ids(this.players).length)) throw new Error(`Invalid live seats/button: ${ids(this.players)}, ${this.button}`);
    for (const i of this.pending) if (!this.players[i] || this.players[i].folded || this.players[i].stack <= EPS) throw new Error(`Invalid pending actors: ${[...this.pending]}`);
    if (Object.values(this.players).some(p => p.street_bet > p.contribution + EPS)) throw new Error("Street contributions exceed total hand contributions");
    const values = [this.pot, this.highest, this.last_full_raise, ...Object.values(this.players).flatMap(p => [p.stack, p.street_bet, p.contribution])];
    if (values.some(v => !Number.isFinite(v) || v < -EPS)) throw new Error(`Nonfinite/negative chip state: ${values}`);
    const chips = sum(Object.values(this.players).map(p => p.stack)) + this.pot;
    if (!close(chips, this.total_chips)) throw new Error(`Chip conservation failed: ${chips} != ${this.total_chips}`);
    if (!this.terminal && !close(sum(Object.values(this.players).map(p => p.contribution)), this.pot)) throw new Error("Pot must equal total hand contributions");
    const cards = [...Object.values(this.players).flatMap(p => p.cards), ...this.board, ...this.deck];
    if (cards.length !== 52 || new Set(cards).size !== 52 || cards.some(c => !Number.isInteger(c) || c < 0 || c >= 52)) throw new Error(`Duplicate or missing cards: ${cards}`);
    if (!this.terminal && (this.current_player === null || !this.pending.has(this.current_player))) throw new Error(`Actor ${this.current_player} not pending: ${[...this.pending]}`);
  }
}

export class TournamentState {
  stacks: Record<number, number>;
  button: number;
  blinds: BlindLevel;
  blind_level_index = 0;
  blind_schedule: BlindSchedule;
  readonly payout: "winner_take_all";
  rng: SeededRNG;
  hand_number = 0;
  hand: HandState | null = null;
  completed: HandState[] = [];
  total_chips: number;
  original_players: number[];
  constructor(stacks: Record<number, number> = { 0: 25, 1: 25, 2: 25 }, button = 0,
    rng = new SeededRNG(0), blindSchedule: BlindSchedule = DEFAULT_SIMULATION_SCHEDULE, payout = "winner_take_all") {
    validateSeats(stacks);
    if (!(blindSchedule instanceof BlindSchedule)) throw new Error("Tournament requires an explicit BlindSchedule");
    if (payout !== "winner_take_all") throw new Error(`Unsupported tournament payout: ${payout}`);
    this.payout = "winner_take_all"; this.blind_schedule = blindSchedule;
    this.stacks = { ...stacks }; this.button = button; this.blinds = blindSchedule.forHand(1)[1]; this.rng = rng;
    if (![2, 3].includes(ids(stacks).length) || Object.values(stacks).some(v => !Number.isFinite(v) || v < 0)) throw new Error(`Expected 2/3 finite nonnegative tournament stacks: ${JSON.stringify(stacks)}`);
    if (!this.active.includes(button)) throw new Error(`Button ${button} must be active: ${this.active}`);
    this.total_chips = sum(Object.values(stacks)); this.original_players = ids(stacks);
  }
  get active(): number[] {
    return this.hand?.terminal ? ids(this.hand.players).filter(i => this.hand!.players[i].stack > EPS) : ids(this.stacks).filter(i => this.stacks[i] > EPS);
  }
  get terminal(): boolean { return this.active.length === 1; }
  get winner(): number | null { return this.terminal ? this.active[0] : null; }
  get current_player(): number | null { return this.hand?.current_player ?? null; }
  clone(): TournamentState {
    const clone = Object.assign(Object.create(TournamentState.prototype) as TournamentState, this);
    clone.stacks = { ...this.stacks }; clone.blinds = { ...this.blinds }; clone.rng = this.rng.clone();
    clone.hand = this.hand?.clone() ?? null; clone.completed = this.completed.map(h => h.clone()); clone.original_players = [...this.original_players];
    return clone;
  }
  startHand(deck?: number[]): HandState {
    if (this.hand) {
      if (!this.hand.terminal) throw new Error("Cannot start another hand before settlement");
      for (const p of Object.values(this.hand.players)) this.stacks[p.player_id] = p.stack;
      this.completed.push(this.hand.clone());
      const seats = ids(this.stacks), active = seats.filter(i => this.stacks[i] > EPS);
      const nextLive = (after: number) => {
        const offset = seats.indexOf(after);
        for (let n = 1; n <= seats.length; n++) { const seat = seats[(offset + n) % seats.length]; if (active.includes(seat)) return seat; }
        throw new Error(`No active seat after ${after}: ${JSON.stringify(this.stacks)}`);
      };
      if (ids(this.hand.players).length === 3 && active.length === 2) {
        const previousBB = Object.values(this.hand.players).find(p => p.position === "BB")!.player_id;
        const nextBB = nextLive(previousBB); this.button = active.find(i => i !== nextBB)!;
      } else this.button = nextLive(this.button);
      this.hand = null;
    }
    if (this.terminal) throw new Error(`Tournament already won by player ${this.winner}`);
    this.hand_number++;
    [this.blind_level_index, this.blinds] = this.blind_schedule.forHand(this.hand_number);
    this.hand = HandState.start(Object.fromEntries(this.active.map(i => [i, this.stacks[i]])), this.button, this.rng, this.blinds, deck,
      { hand_number: this.hand_number, blind_level_index: this.blind_level_index });
    this.assertInvariants(); return this.hand;
  }
  act(actionId: string): void {
    if (!this.hand) throw new Error("Tournament has no current hand; call startHand");
    applyAction(this.hand, actionId); this.assertInvariants();
  }
  utility(player: number): number {
    if (!this.original_players.includes(player) || !this.terminal) throw new Error(`Winner utility requires terminal tournament and valid player: ${player}`);
    return Number(player === this.winner) - 1 / this.original_players.length;
  }
  assertInvariants(): void {
    if (this.hand) this.hand.assertInvariants();
    const actual = this.hand ? sum(Object.values(this.hand.players).map(p => p.stack)) + this.hand.pot : sum(Object.values(this.stacks));
    if (!close(actual, this.total_chips)) throw new Error(`Tournament chip conservation: ${actual} != ${this.total_chips}`);
  }
}
