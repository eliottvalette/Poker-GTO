/** Bounded local calculation; canonical engine owns every rule and terminal award. */
import { HandState, SeededRNG } from "./engine";
import { ACTION_IDS, legalActions, type SolverAction } from "./actions";
import { observe } from "./observation";
import { behaviorProbabilities } from "./behavior";

export type WeightedHand = { cards: [number, number]; probability: number };
export type Ranges = Record<number, WeightedHand[]>;
export type HybridBudget = { maxNodes: number; iterations: number; samples: number; maxDepth: number; maxJointCandidates: number; seed: number };
export const HYBRID_BUDGETS: Record<string, HybridBudget> = {
  FAST: { maxNodes: 5000, iterations: 10, samples: 16, maxDepth: 1, maxJointCandidates: 1000, seed: 0 },
  NORMAL: { maxNodes: 100000, iterations: 100, samples: 64, maxDepth: 4, maxJointCandidates: 10000, seed: 0 },
  DEEP: { maxNodes: 1000000, iterations: 1000, samples: 512, maxDepth: 6, maxJointCandidates: 100000, seed: 0 },
};
export type HybridAnalysis = {
  legalActions: SolverAction[]; probabilities: Record<string, number>; ev: Record<string, number>;
  evUnits: "physical_chips/hand_delta"; method: string; nodes: number; samples: number; iterations: number;
  standardErrors: Record<string, number | null>; warnings: string[]; ranges: Ranges; ownPublicRange: WeightedHand[];
  strategyByHand?: Record<string, Record<string, number>>;
  baselineProbabilities?: Record<string, number>;
};
type Deal = { hands: Record<number, [number, number]>; probability: number };
type Node = { player: number | null; key: string; actions: SolverAction[]; children: Node[]; utilities?: number[] };
const sum = (values: number[]) => values.reduce((a, b) => a + b, 0);
const sortedCards = (cards: [number, number]): [number, number] => [...cards].sort((a, b) => a - b) as [number, number];
const keyOf = (cards: [number, number]) => sortedCards(cards).join(",");

export function uniformRanges(hand: HandState): Ranges {
  const entries: WeightedHand[] = [];
  for (let a = 0; a < 52; a++) for (let b = a + 1; b < 52; b++) {
    if (!hand.board.includes(a) && !hand.board.includes(b)) entries.push({ cards: [a, b], probability: 1 });
  }
  return Object.fromEntries(Object.keys(hand.players).map(p => [p, entries.map(e => ({ cards: [...e.cards], probability: 1 / entries.length }))]));
}

export function normalizeRanges(hand: HandState, ranges: Ranges): Ranges {
  if (Object.keys(hand.players).sort().join() !== Object.keys(ranges).sort().join()) throw new Error("Ranges must cover every seated player, including folds");
  return Object.fromEntries(Object.entries(ranges).map(([seat, rows]) => {
    if (!Array.isArray(rows)) throw new Error(`Range ${seat} must be an array`);
    const seen = new Set<string>();
    const valid = rows.map(row => {
      if (!Array.isArray(row.cards) || row.cards.length !== 2 || row.cards[0] === row.cards[1]
          || row.cards.some(c => !Number.isInteger(c) || c < 0 || c >= 52)
          || !Number.isFinite(row.probability) || row.probability < 0) throw new Error(`Invalid range entry for ${seat}`);
      const key = keyOf(row.cards);
      if (seen.has(key)) throw new Error(`Duplicate combination ${key}`);
      seen.add(key);
      return { cards: sortedCards(row.cards), probability: row.cards.some(c => hand.board.includes(c)) ? 0 : row.probability };
    }).filter(row => row.probability > 0);
    const total = sum(valid.map(r => r.probability));
    if (!(total > 0) || !Number.isFinite(total)) throw new Error(`Range ${seat} has no feasible mass`);
    return [seat, valid.map(row => ({ ...row, probability: row.probability / total }))];
  }));
}

function world(publicState: HandState, hands: Deal["hands"], rng?: SeededRNG): HandState {
  const state = publicState.clone();
  for (const [p, cards] of Object.entries(hands)) state.players[Number(p)].cards = sortedCards(cards);
  const dead = [...state.board, ...Object.values(hands).flat()];
  if (new Set(dead).size !== dead.length) throw new Error("Incompatible joint deal");
  state.deck = Array.from({ length: 52 }, (_, c) => c).filter(c => !dead.includes(c));
  rng?.shuffle(state.deck);
  state.assertInvariants();
  return state;
}

function enumerate(ranges: Ranges): Deal[] {
  const seats = Object.keys(ranges).map(Number), rows: Deal[] = [];
  function add(index: number, hands: Deal["hands"], used: number[], probability: number) {
    if (index === seats.length) { rows.push({ hands, probability }); return; }
    const seat = seats[index];
    for (const row of ranges[seat]) if (!row.cards.some(c => used.includes(c)))
      add(index + 1, { ...hands, [seat]: row.cards }, [...used, ...row.cards], probability * row.probability);
  }
  add(0, {}, [], 1);
  const total = sum(rows.map(r => r.probability));
  if (!total) throw new Error("Ranges have no compatible joint deal");
  return rows.map(r => ({ ...r, probability: r.probability / total }));
}

function privateRangesFor(hand: HandState, ranges: Ranges): Ranges {
  const hero = hand.current_player!, cards = hand.actor.cards;
  return normalizeRanges(hand, Object.fromEntries(Object.entries(ranges).map(([seat, rows]) => [seat,
    Number(seat) === hero ? [{ cards, probability: 1 }] : rows.filter(row => !row.cards.some(c => cards.includes(c)))])));
}

class Work {
  nodes = 0;
  constructor(readonly budget: HybridBudget) {}
  visit() { if (this.nodes >= this.budget.maxNodes) throw new Error(`Hybrid node budget exhausted: ${this.nodes}`); this.nodes++; }
}

function river(publicState: HandState, ranges: Ranges, work: Work, chance = false): HybridAnalysis {
  const seats = Object.keys(ranges).map(Number), hero = publicState.current_player!;
  const rng = new SeededRNG(work.budget.seed);
  let chanceSamples = 0, rolloutLeaves = 0;
  const regrets = new Map<string, number[]>(), masses = new Map<string, number[]>(), masks = new Map<string, boolean[]>();
  function build(state: HandState, depth = 0): Node {
    work.visit();
    if (chance && depth >= work.budget.maxDepth && !state.terminal) {
      state = state.clone(); rolloutLeaves++;
      while (!state.terminal) {
        work.visit();
        const actions = legalActions(state), action = actions[Math.floor(rng.next() * actions.length)];
        state.act(action.category, action.amount_to);
      }
    }
    if (state.terminal) return { player: null, key: "", actions: [], children: [], utilities: seats.map(p => state.utility(p)) };
    const obs = observe(state), key = JSON.stringify(obs), actions = legalActions(state);
    if (!regrets.has(key)) { regrets.set(key, ACTION_IDS.map(() => 0)); masses.set(key, ACTION_IDS.map(() => 0)); masks.set(key, obs.legal_mask); }
    return { player: state.current_player, key, actions, children: actions.map(a => {
      const child = state.clone(); child.act(a.category, a.amount_to); return build(child, depth + 1);
    }) };
  }
  const deals = enumerate(ranges);
  const trees: { deal: Deal; tree: Node }[] = [];
  const remaining = 52 - publicState.board.length - 2 * seats.length;
  const exhaustive = chance && publicState.street === "TURN" && deals.length * remaining <= work.budget.samples;
  const perDeal = chance ? (exhaustive ? remaining : Math.floor(work.budget.samples / deals.length)) : 1;
  if (perDeal < 1) throw new Error(`At least ${deals.length} samples are required to cover each public private deal`);
  for (const deal of deals) {
    const initial = world(publicState, deal.hands);
    for (let index = 0; index < perDeal; index++) {
      const state = initial.clone();
      if (exhaustive) { const card = initial.deck[index]; state.deck = [...initial.deck.filter(c => c !== card), card]; }
      else if (chance) rng.shuffle(state.deck);
      trees.push({ deal: { hands: deal.hands, probability: deal.probability / perDeal }, tree: build(state) });
      if (chance) chanceSamples++;
    }
  }
  const treeNodes = work.nodes;
  for (let iteration = 0; iteration < work.budget.iterations; iteration++) {
    if (work.nodes + 2 * treeNodes > work.budget.maxNodes) throw new Error("Insufficient nodes for complete CFR and final EV evaluation; increase budget");
    const strategies = new Map<string, number[]>(), changes = new Map<string, number[]>(), own = new Map<string, number[]>();
    for (const [key, regret] of regrets) {
      const mask = masks.get(key)!;
      const positive = regret.map((r, i) => mask[i] ? Math.max(r, 0) : 0), total = sum(positive);
      strategies.set(key, total > 0 ? positive.map(v => v / total) : mask.map(m => Number(m) / mask.filter(Boolean).length));
      changes.set(key, ACTION_IDS.map(() => 0));
    }
    function visit(node: Node, reach: number[], chance: number): number[] {
      work.visit();
      if (node.utilities) return node.utilities;
      const p = seats.indexOf(node.player!), strategy = strategies.get(node.key)!;
      const values = node.children.map((child, i) => {
        const next = [...reach]; next[p] *= strategy[ACTION_IDS.indexOf(node.actions[i].action_id as typeof ACTION_IDS[number])];
        return visit(child, next, chance);
      });
      const expected = seats.map((_, i) => sum(values.map((value, a) => strategy[ACTION_IDS.indexOf(node.actions[a].action_id as typeof ACTION_IDS[number])] * value[i])));
      const external = reach.reduce((v, r, i) => i === p ? v : v * r, chance);
      node.actions.forEach((a, i) => { changes.get(node.key)![ACTION_IDS.indexOf(a.action_id as typeof ACTION_IDS[number])] += external * (values[i][p] - expected[p]); });
      own.set(node.key, strategy.map(v => v * reach[p]));
      return expected;
    }
    for (const { deal, tree } of trees) visit(tree, seats.map(() => 1), deal.probability);
    for (const [key, row] of regrets) {
      row.forEach((_, i) => { row[i] += changes.get(key)![i]; masses.get(key)![i] += own.get(key)![i]; });
    }
  }
  const averages = new Map<string, number[]>();
  for (const [key, row] of masses) {
    const total = sum(row), mask = masks.get(key)!;
    averages.set(key, total > 0 ? row.map(v => v / total) : mask.map(m => Number(m) / mask.filter(Boolean).length));
  }
  function evaluate(node: Node): number {
    work.visit();
    if (node.utilities) return node.utilities[seats.indexOf(hero)];
    return sum(node.children.map((child, i) => averages.get(node.key)![ACTION_IDS.indexOf(node.actions[i].action_id as typeof ACTION_IDS[number])] * evaluate(child)));
  }
  const strategyByHand: Record<string, Record<string, number>> = {}, evByHand: Record<string, Record<string, number>> = {}, handMass: Record<string, number> = {};
  for (const { deal, tree } of trees) {
    const hand = keyOf(deal.hands[hero]);
    handMass[hand] = (handMass[hand] ?? 0) + deal.probability;
    strategyByHand[hand] = Object.fromEntries(tree.actions.map(a => [a.action_id, averages.get(tree.key)![ACTION_IDS.indexOf(a.action_id as typeof ACTION_IDS[number])]]));
    evByHand[hand] ??= Object.fromEntries(tree.actions.map(a => [a.action_id, 0]));
    tree.children.forEach((child, i) => { evByHand[hand][tree.actions[i].action_id] += deal.probability * evaluate(child); });
  }
  const selected = keyOf(publicState.actor.cards);
  if (!evByHand[selected]) throw new Error("Known Hero holding has no compatible public-range support");
  const ev = Object.fromEntries(Object.entries(evByHand[selected]).map(([a, v]) => [a, v / handMass[selected]]));
  return { legalActions: legalActions(publicState), probabilities: strategyByHand[selected], ev, evUnits: "physical_chips/hand_delta",
    method: chance ? "public-information-set-CFR/bounded-chance-and-rollout-leaves" : "exact-terminal/full-tree-CFR", nodes: work.nodes, samples: chanceSamples, iterations: work.budget.iterations,
    standardErrors: Object.fromEntries(Object.keys(ev).map(a => [a, null])), ranges: privateRangesFor(publicState, ranges), ownPublicRange: ranges[hero], strategyByHand,
    warnings: ["Belief-conditioned local solve; finite-iteration error is not bounded; no safe re-solving claim.",
      ...(chance ? [`${exhaustive ? "Exhaustive turn rivers" : "Fixed sampled chance forest"}; ${rolloutLeaves} baseline rollout leaves; sampled-forest overfitting is possible.`] : []),
      ...(seats.length === 3 ? ["Multiplayer CFR has no general two-player Nash guarantee."] : [])] };
}

function sampleDeal(ranges: Ranges, rng: SeededRNG): Deal["hands"] {
  for (let attempt = 0; attempt < 10000; attempt++) {
    const hands: Deal["hands"] = {};
    for (const [p, entries] of Object.entries(ranges)) {
      let threshold = rng.next();
      const selected = entries.find(row => { threshold -= row.probability; return threshold < 0; });
      if (!selected) throw new Error("Range probability transport failed");
      hands[Number(p)] = selected.cards;
    }
    const cards = Object.values(hands).flat();
    if (new Set(cards).size === cards.length) return hands;
  }
  throw new Error("Joint range rejection limit reached");
}

function simulation(publicState: HandState, ranges: Ranges, work: Work, profiles: Record<number, string>): HybridAnalysis {
  const hero = publicState.current_player!, rng = new SeededRNG(work.budget.seed), actions = legalActions(publicState);
  const known = keyOf(publicState.actor.cards);
  if (!ranges[hero].some(row => keyOf(row.cards) === known)) throw new Error("Hero outside public range");
  const privateRanges = privateRangesFor(publicState, ranges);
  const records: Record<string, number>[] = [];
  function evaluate(state: HandState, depth: number, branchRng: SeededRNG): number {
    work.visit();
    if (state.terminal) return state.utility(hero);
    if (depth >= work.budget.maxDepth) {
      const child = state.clone();
      while (!child.terminal) {
        work.visit(); const legal = legalActions(child);
        const probabilities = behaviorProbabilities(observe(child), profiles[child.current_player!]);
        let draw = branchRng.next(), action = legal[legal.length - 1];
        for (const candidate of legal) {
          draw -= probabilities[ACTION_IDS.findIndex(id => id === candidate.action_id)];
          if (draw < 0) { action = candidate; break; }
        }
        child.act(action.category, action.amount_to);
      }
      return child.utility(hero);
    }
    const legal = legalActions(state);
    const probabilities = behaviorProbabilities(observe(state), profiles[state.current_player!]);
    return sum(legal.map(action => { const child = state.clone(); child.act(action.category, action.amount_to);
      return probabilities[ACTION_IDS.findIndex(id => id === action.action_id)] * evaluate(child, depth + 1, branchRng); }));
  }
  for (let sample = 0; sample < work.budget.samples; sample++) {
    const state = world(publicState, sampleDeal(privateRanges, rng), rng), seed = Math.floor(rng.next() * 2 ** 32);
    records.push(Object.fromEntries(actions.map(action => {
      const child = state.clone(); child.act(action.category, action.amount_to);
      return [action.action_id, evaluate(child, 1, new SeededRNG(seed))];
    })));
  }
  const ev = Object.fromEntries(actions.map(a => [a.action_id, sum(records.map(r => r[a.action_id])) / records.length]));
  const best = Math.max(...Object.values(ev)), ties = Object.keys(ev).filter(a => Math.abs(ev[a] - best) < 1e-10);
  const ranking = Object.keys(ev).sort((a, b) => ev[b] - ev[a]);
  const precisionWarnings: string[] = [];
  if (records.length < 2) precisionWarnings.push("Insufficient samples to estimate Monte Carlo precision.");
  else if (ranking.length > 1) {
    const [first, second] = ranking, gap = ev[first] - ev[second];
    const pairedSE = Math.sqrt(sum(records.map(r => (r[first] - r[second] - gap) ** 2)) / (records.length - 1) / records.length);
    if (gap <= 1.96 * pairedSE) precisionWarnings.push(`Top action ranking is unresolved at the approximate paired 95% sampling threshold (gap ${gap.toFixed(4)}, SE ${pairedSE.toFixed(4)} chips).`);
  }
  return { legalActions: actions, ev, probabilities: Object.fromEntries(Object.keys(ev).map(a => [a, Number(ties.includes(a)) / ties.length])),
    evUnits: "physical_chips/hand_delta", method: "sampled-chance/depth-limited-profile-rollout", nodes: work.nodes,
    samples: records.length, iterations: 0, ranges: privateRanges, ownPublicRange: ranges[hero],
    standardErrors: Object.fromEntries(Object.keys(ev).map(a => [a, records.length < 2 ? null : Math.sqrt(sum(records.map(r => (r[a] - ev[a]) ** 2)) / (records.length - 1) / records.length)])),
    warnings: [...precisionWarnings, `Behavior-conditioned response: ${JSON.stringify(profiles)}. Synthetic population assumptions, not an equilibrium solve.`,
      "Ranges shown are factors; private deals are jointly conditioned on disjoint cards.",
      "Sampling uncertainty excludes opponent-model uncertainty. Python and browser use different seeded PRNGs."] };
}

export function analyzeHybrid(hand: HandState, inputRanges: Ranges, budget: HybridBudget, searchMode: "behavior" | "public_cfr" | "response" = "behavior", profiles?: Record<number, string>): HybridAnalysis {
  if (hand.terminal) throw new Error("Hybrid analysis requires a live decision");
  const fields = ["maxNodes", "iterations", "samples", "maxDepth", "maxJointCandidates", "seed"];
  if (Object.keys(budget).sort().join() !== fields.sort().join()) throw new Error("Hybrid budget requires every declared limit and seed");
  if (searchMode !== "behavior" && searchMode !== "public_cfr" && searchMode !== "response") throw new Error(`Unsupported search mode: ${searchMode}`);
  for (const [name, value] of Object.entries(budget)) if (!Number.isSafeInteger(value) || (name !== "seed" && value < 1)) throw new Error(`Invalid budget ${name}=${value}`);
  const ranges = normalizeRanges(hand, inputRanges), work = new Work(budget);
  const product = Object.values(ranges).reduce((n, r) => n * r.length, 1);
  if (searchMode === "public_cfr") {
    if (product > budget.maxJointCandidates) throw new Error(`Public CFR requires at most ${budget.maxJointCandidates} private candidates; narrow the explicit ranges or select behavior search`);
    return river(hand, ranges, work, hand.street !== "RIVER");
  }
  const behaviors = profiles ?? Object.fromEntries(Object.keys(hand.players).map(p => [p, "uniform"]));
  if (Object.keys(behaviors).sort().join() !== Object.keys(hand.players).sort().join()) throw new Error("Behavior profile seats must match hand");
  return searchMode !== "response" && hand.street === "RIVER" && product <= budget.maxJointCandidates ? river(hand, ranges, work) : simulation(hand, ranges, work, behaviors);
}

export type HandTransport = Pick<HandState, "players" | "button" | "hand_number" | "blind_level_index" | "blinds" |
  "deck" | "total_chips" | "initial_stacks" | "board" | "street" | "pot" | "highest" | "last_full_raise" |
  "current_player" | "history" | "terminal" | "showdown" | "awards"> & { pending: number[] };
export function transportHand(hand: HandState): HandTransport { return { ...hand, pending: [...hand.pending] }; }
export function restoreHand(raw: HandTransport): HandState {
  const hand = Object.assign(new HandState(), raw, { pending: new Set(raw.pending) });
  hand.assertInvariants(); return hand;
}
