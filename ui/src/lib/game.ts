/** Browser-owned table state; Python is used only for offline training and parity tests. */
import { SeededRNG, TournamentState, type ActionEvent } from "./poker/engine";
import { applyAction, legalActions, type SolverAction } from "./poker/actions";

export type { ActionEvent, SolverAction };
export type PublicPlayer = {
  player_id: number; stack_bb: number; active: boolean; position: string | null;
  folded: boolean; bet_bb: number; cards: number[];
};
export type TableView = {
  hero: number; hand_number: number; button: number; active_players: number[];
  players: PublicPlayer[]; board: number[]; street: string; pot_bb: number;
  to_call_bb: number; actor: number | null; hand_terminal: boolean;
  tournament_terminal: boolean; winner: number | null;
  hand_results_bb: Record<number, number>; hero_result_bb: number;
  total_chips_bb: number; blind_level_index: number; chip_unit_big_blind: number; blinds: { small: number; big: number };
  legal_actions: SolverAction[]; history: ActionEvent[];
  policy: { status: "unavailable" | "experimental"; reason: string; probabilities: Record<string, number> | null };
};

export class BrowserTable {
  constructor(public tournament: TournamentState, public readonly hero: number, private rng: SeededRNG) {
    if (!tournament.original_players.includes(hero)) throw new Error(`Hero ${hero} is not seated`);
  }

  static create(seed: number, hero = 2): BrowserTable {
    const tournament = new TournamentState({ 0: 25, 1: 25, 2: 25 }, 0, new SeededRNG(seed));
    tournament.startHand();
    const table = new BrowserTable(tournament, hero, new SeededRNG(seed + 1));
    table.advanceBots();
    return table;
  }

  clone(): BrowserTable {
    return new BrowserTable(this.tournament.clone(), this.hero, this.rng.clone());
  }

  act(action: string): void {
    if (this.tournament.current_player !== this.hero) throw new Error(`Hero ${this.hero} is not acting`);
    this.tournament.act(action);
    this.advanceBots();
  }

  nextHand(): void {
    if (this.tournament.terminal) throw new Error(`Tournament already won by player ${this.tournament.winner}`);
    this.tournament.startHand();
    this.advanceBots();
  }

  private advanceBots(): void {
    const hand = this.tournament.hand;
    if (!hand) throw new Error("Table has no current hand");
    let decisions = 0;
    while (!hand.terminal && hand.current_player !== this.hero) {
      if (++decisions > 100) throw new Error("Bot action budget exceeded: 100 decisions per hand");
      const actions = legalActions(hand);
      if (actions.length === 0) throw new Error(`No legal actions for player ${hand.current_player}`);
      applyAction(hand, actions[Math.floor(this.rng.next() * actions.length)].action_id);
    }
    this.tournament.assertInvariants();
  }

  view(): TableView {
    const t = this.tournament, hand = t.hand;
    if (!hand) throw new Error("Table has no current hand");
    const active = t.active;
    const bigBlind = hand.blinds.big;
    const players = t.original_players.map(id => {
      const p = hand.players[id];
      return {
        player_id: id, stack_bb: (p?.stack ?? t.stacks[id]) / bigBlind, active: active.includes(id),
        position: p?.position ?? null, folded: p?.folded ?? false, bet_bb: (p?.street_bet ?? 0) / bigBlind,
        cards: p && (id === this.hero || (hand.showdown && !p.folded)) ? [...p.cards] : [],
      };
    });
    return {
      hero: this.hero, hand_number: t.hand_number, button: t.button, active_players: [...active],
      players, board: [...hand.board], street: hand.street, pot_bb: hand.pot / bigBlind,
      to_call_bb: hand.terminal ? 0 : hand.toCall() / bigBlind, actor: hand.current_player,
      hand_terminal: hand.terminal, tournament_terminal: t.terminal, winner: t.winner,
      hand_results_bb: hand.terminal ? Object.fromEntries(t.original_players.map(id => [id, hand.players[id] ? hand.utility(id) / bigBlind : 0])) : {},
      // Historical sum uses each settled hand's own BB unit.
      hero_result_bb: [...t.completed, ...(hand.terminal ? [hand] : [])]
        .reduce((total, previous) => total + (previous.players[this.hero] ? previous.utility(this.hero) / previous.blinds.big : 0), 0),
      total_chips_bb: t.total_chips / bigBlind, blind_level_index: hand.blind_level_index,
      chip_unit_big_blind: bigBlind, blinds: { small: hand.blinds.small / bigBlind, big: 1 },
      legal_actions: hand.current_player === this.hero ? legalActions(hand).map(action => ({ ...action,
        amount_to: action.amount_to === null ? null : action.amount_to / bigBlind })) : [],
      history: hand.history.map(event => ({ ...event, amount_to: event.amount_to / bigBlind,
        amount_added: event.amount_added / bigBlind, pot_before: event.pot_before / bigBlind,
        pot_after: event.pot_after / bigBlind, highest_before: event.highest_before / bigBlind })),
      policy: { status: "unavailable", reason: "No compatible average policy loaded", probabilities: null },
    };
  }
}

export function cardLabel(card: number): string {
  if (!Number.isInteger(card) || card < 0 || card > 51) throw new Error(`Invalid card ID ${card}`);
  return "23456789TJQKA"[Math.floor(card / 4)] + ["♠", "♥", "♦", "♣"][card % 4];
}
