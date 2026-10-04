/** Live table contract. Python owns poker transitions and legal action sizing. */
export type SolverAction = { action_id: string; category: string; amount_to: number | null };
export type PublicPlayer = {
  player_id: number; stack_bb: number; active: boolean; position: string | null;
  folded: boolean; bet_bb: number; cards: number[];
};
export type ActionEvent = {
  street: string; player_id: number; position: string; action: string;
  amount_to: number; amount_added: number; pot_before: number; pot_after: number;
  highest_before: number;
};
export type TableView = {
  version: 2; session_id: string; revision: number; hero: number; hand_number: number;
  button: number; active_players: number[]; players: PublicPlayer[]; board: number[];
  street: string; pot_bb: number; to_call_bb: number; actor: number | null;
  hand_terminal: boolean; tournament_terminal: boolean; winner: number | null;
  hand_results_bb: Record<number, number>; hero_result_bb: number; total_chips_bb: number; blinds: { small: number; big: number };
  legal_actions: SolverAction[]; history: ActionEvent[];
  policy: { status: "unavailable" | "experimental"; reason: string; probabilities: Record<string, number> | null };
};

export function parseTableView(value: unknown): TableView {
  if (!value || typeof value !== "object") throw new Error("Missing engine table response");
  const view = value as TableView;
  if (view.version !== 2 || !Array.isArray(view.players) || !Array.isArray(view.legal_actions)
      || !Array.isArray(view.history) || !Array.isArray(view.board)
      || typeof view.session_id !== "string" || !Number.isInteger(view.revision)
      || typeof view.hand_terminal !== "boolean" || typeof view.tournament_terminal !== "boolean"
      || !view.policy || !["unavailable", "experimental"].includes(view.policy.status)) {
    throw new Error("Incompatible table response: expected Python contract v2");
  }
  for (const player of view.players) {
    if (!Number.isInteger(player.player_id) || !Number.isFinite(player.stack_bb)
        || !Number.isFinite(player.bet_bb) || player.stack_bb < -1e-9 || !Array.isArray(player.cards)) {
      throw new Error(`Invalid public player ${JSON.stringify(player)}`);
    }
    player.cards.forEach(cardLabel);
  }
  view.board.forEach(cardLabel);
  if (!view.hand_results_bb || typeof view.hand_results_bb !== "object"
      || !Number.isFinite(view.hero_result_bb)) {
    throw new Error("Missing settled hand/tournament results in table response");
  }
  if (view.hand_terminal && view.players.some(p => !Number.isFinite(view.hand_results_bb[p.player_id]))) {
    throw new Error("Terminal table response requires a hand result for every player ID");
  }
  for (const action of view.legal_actions) {
    if (typeof action.action_id !== "string" || typeof action.category !== "string"
        || (action.amount_to !== null && !Number.isFinite(action.amount_to))) {
      throw new Error(`Invalid server action ${JSON.stringify(action)}`);
    }
  }
  const chips = view.players.reduce((sum, p) => sum + p.stack_bb, 0) + view.pot_bb;
  if (!Number.isFinite(chips) || !Number.isFinite(view.total_chips_bb)
      || Math.abs(chips - view.total_chips_bb) > 1e-7) {
    throw new Error(`Engine chip conservation failed: ${chips} != ${view.total_chips_bb}`);
  }
  return view;
}

export async function tableRequest(operation: "new" | "action" | "next", payload: Record<string, unknown>): Promise<TableView> {
  const response = await fetch("/api/table", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ operation, ...payload }), signal: AbortSignal.timeout(10000),
  });
  const body = await response.json();
  if (!response.ok) throw new Error(body.error ?? `Engine request failed: ${response.status}`);
  return parseTableView(body);
}

export function cardLabel(card: number): string {
  if (!Number.isInteger(card) || card < 0 || card > 51) throw new Error(`Invalid card ID ${card}`);
  return "23456789TJQKA"[Math.floor(card / 4)] + ["♠", "♥", "♦", "♣"][card % 4];
}

export function policyPercentage(view: TableView, actionId: string): string | null {
  if (view.policy.status === "unavailable" || view.policy.probabilities === null) return null;
  const probability = view.policy.probabilities[actionId];
  if (!Number.isFinite(probability) || probability < 0 || probability > 1) {
    throw new Error(`Invalid average-policy probability for ${actionId}: ${probability}`);
  }
  return `${(probability * 100).toFixed(1)}%`;
}

export async function closeTable(sessionId: string): Promise<void> {
  const response = await fetch("/api/table", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ operation: "close", session_id: sessionId }), signal: AbortSignal.timeout(10000),
  });
  const body = await response.json();
  if (!response.ok || body.closed !== true) {
    throw new Error(body.error ?? `Failed to close table ${sessionId}: HTTP ${response.status}`);
  }
}
