import assert from "node:assert/strict";

const origin = process.env.POKER_UI_ORIGIN ?? "http://127.0.0.1:3100";
async function command(operation, payload = {}) {
  const response = await fetch(`${origin}/api/table`, {
    method: "POST", headers: { "Content-Type": "application/json", Origin: origin },
    body: JSON.stringify({ operation, ...payload }),
  });
  const body = await response.json();
  assert.equal(response.status, 200, JSON.stringify(body));
  return body;
}

const page = await fetch(origin);
assert.equal(page.status, 200);
assert.match(await page.text(), /Test Live/);
let view = await command("new", { seed: 1, hero: 2, opponents: "shove_fold" });
const sessionId = view.session_id;
let sawHeadsUp = false;
let commands = 0;
try {
  while (!view.tournament_terminal && commands < 100) {
    assert.equal(view.version, 2);
    assert.equal(view.policy.status, "unavailable");
    assert.ok(Math.abs(view.players.reduce((sum, p) => sum + p.stack_bb, 0) + view.pot_bb - 75) < 1e-7);
    sawHeadsUp ||= view.active_players.length === 2;
    if (view.hand_terminal) {
      assert.ok(view.players.every(p => Number.isFinite(view.hand_results_bb[p.player_id])));
      view = await command("next", { session_id: sessionId, revision: view.revision });
    } else {
      const ids = view.legal_actions.map(a => a.action_id);
      const action = ids.includes("ALL_IN") ? "ALL_IN" : ids.includes("CALL") ? "CALL" : ids[0];
      view = await command("action", { session_id: sessionId, revision: view.revision, action });
    }
    commands += 1;
  }
  assert.equal(view.tournament_terminal, true);
  assert.equal(sawHeadsUp, true);
  assert.equal(view.active_players.length, 1);
  assert.equal(Math.max(...view.players.map(p => p.stack_bb)), 75);
  const denied = await fetch(`${origin}/api/table`, {
    method: "POST", headers: { "Content-Type": "application/json", Origin: "http://other.example" },
    body: JSON.stringify({ operation: "new" }),
  });
  assert.equal(denied.status, 403);
  console.log(JSON.stringify({ commands, hands: view.hand_number, sawHeadsUp, winner: view.winner }));
} finally {
  assert.equal((await command("close", { session_id: sessionId })).closed, true);
}
