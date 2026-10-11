import { ACTION_IDS } from "./poker/actions";
import { fetchPolicyCatalog, policyArtifactBase } from "./policy-catalog";
import { EVENTS, HISTORY_WIDTH, NUMERIC_NAMES, POSITIONS, STATE_VERSION,
  validateObservation, type Observation } from "./poker/observation";

const INPUTS = ["cards", "street", "position", "numeric", "history", "mask"] as const;
import { numericNames, neuralObservation } from "./poker/neural";

export type PolicyManifest = {
  version: number;
  action_schema_version: number;
  feature_schema_version: number;
  seat_normalization: "hero_then_clockwise_positions";
  suit_normalization: "first_observable_occurrence" | "private_order_minimum";
  traversal_mode: "external_sampling" | "outcome_sampling";
  state_version: number;
  architecture: string;
  model_sha256: string;
  objective: Observation["objective"];
  iteration: number;
  supported_player_counts: number[];
  actions: string[];
  numeric_names: string[];
  positions: string[];
  events: string[];
  normalization_bb: number;
  amount_units: "current_big_blinds";
  utility_units: "initial_big_blind_chips";
  history_scope: "current_hand";
  payout_scope: "winner_take_all";
  card_encoding: string;
  card_slots: number;
  unknown_card: number;
  card_vocabulary: number;
  history_width: number;
  full_history: boolean;
  batch_size: number;
  inputs: string[];
  output: string;
  validation_max_absolute_error: number;
};

export class PolicyCoverageError extends Error {
  constructor(message: string) { super(message); this.name = "PolicyCoverageError"; }
}

export type LoadedAveragePolicy = {
  manifest: PolicyManifest;
  query(observation: Observation): Promise<Record<string, number>>;
  release(): Promise<void>;
};

function equalArray(actual: unknown, expected: readonly unknown[]): boolean {
  return Array.isArray(actual) && actual.length === expected.length && actual.every((value, index) => value === expected[index]);
}

export function validatePolicyManifest(raw: unknown): PolicyManifest {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) throw new Error("Policy manifest must be a JSON object");
  const manifest = raw as PolicyManifest;
  const fields = ["action_schema_version", "feature_schema_version", "suit_normalization", "seat_normalization", "traversal_mode", "version", "state_version", "architecture", "model_sha256", "objective", "iteration",
    "supported_player_counts", "actions", "numeric_names", "positions", "events", "normalization_bb",
    "card_encoding", "card_slots", "unknown_card", "card_vocabulary", "history_width", "full_history",
    "batch_size", "inputs", "output", "validation_max_absolute_error", "amount_units", "utility_units", "history_scope", "payout_scope"];
  if (!equalArray(Object.keys(raw).sort(), fields.sort()) || manifest.version !== 4
      || manifest.action_schema_version !== 1
      || ![2, 3, 4].includes(manifest.feature_schema_version)
      || manifest.suit_normalization !== (manifest.feature_schema_version >= 3 ? "private_order_minimum" : "first_observable_occurrence")
      || manifest.seat_normalization !== "hero_then_clockwise_positions"
      || !["external_sampling", "outcome_sampling"].includes(manifest.traversal_mode) || manifest.state_version !== STATE_VERSION
      || manifest.architecture !== `cards8_numeric32_historyGRU32_head64_features${manifest.feature_schema_version}` || typeof manifest.model_sha256 !== "string"
      || !/^[a-f0-9]{64}$/.test(manifest.model_sha256) || manifest.objective !== "hand_chip_delta"
      || !Number.isInteger(manifest.iteration) || manifest.iteration < 1 || !Array.isArray(manifest.supported_player_counts)
      || !manifest.supported_player_counts.length || manifest.supported_player_counts.some(count => count !== 2 && count !== 3)
      || new Set(manifest.supported_player_counts).size !== manifest.supported_player_counts.length
      || !equalArray(manifest.actions, ACTION_IDS) || !equalArray(manifest.numeric_names, numericNames(manifest.feature_schema_version))
      || !equalArray(manifest.positions, POSITIONS) || !equalArray(manifest.events, EVENTS)
      || manifest.amount_units !== "current_big_blinds" || manifest.utility_units !== "initial_big_blind_chips"
      || manifest.history_scope !== "current_hand" || manifest.payout_scope !== "winner_take_all"
      || manifest.normalization_bb !== 25 || manifest.card_encoding !== "rank_index_times_4_plus_suit_index"
      || manifest.card_slots !== 7 || manifest.unknown_card !== 52 || manifest.card_vocabulary !== 53
      || manifest.history_width !== HISTORY_WIDTH || manifest.full_history !== true || manifest.batch_size !== 1
      || !equalArray(manifest.inputs, INPUTS) || manifest.output !== "probabilities"
      || !Number.isFinite(manifest.validation_max_absolute_error) || manifest.validation_max_absolute_error < 0
      || manifest.validation_max_absolute_error > 1e-5) {
    throw new Error("Incompatible average-policy manifest: expected the exact full-history state/action schema and validated CPU export");
  }
  return structuredClone(manifest);
}

/** Validated average-policy artifact selected by the published catalog. */
export async function loadAveragePolicy(model: ArrayBuffer, rawManifest: unknown, assetOrigin?: string): Promise<LoadedAveragePolicy> {
  const manifest = validatePolicyManifest(rawManifest);
  if (!model.byteLength) throw new Error("Average-policy ONNX model is empty");
  const hash = Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", model)),
    value => value.toString(16).padStart(2, "0")).join("");
  if (hash !== manifest.model_sha256) throw new Error(`ONNX model SHA256 ${hash} does not match selected manifest ${manifest.model_sha256}`);
  const ort = await import("onnxruntime-web/wasm");
  ort.env.wasm.wasmPaths = new URL("/onnxruntime/", assetOrigin ?? globalThis.location.origin).href;
  ort.env.wasm.numThreads = 1;
  const session = await ort.InferenceSession.create(model, { executionProviders: ["wasm"] });
  if (!equalArray(session.inputNames, INPUTS) || !equalArray(session.outputNames, ["probabilities"])) {
    await session.release();
    throw new Error(`ONNX input/output contract mismatch: inputs=${session.inputNames}, outputs=${session.outputNames}`);
  }
  let released = false;
  return {
    manifest,
    async query(observation: Observation): Promise<Record<string, number>> {
      if (released) throw new Error("Average-policy session has already been released");
      validateObservation(observation);
      const count = Math.round(observation.numeric[NUMERIC_NAMES.indexOf("player_count")] * 3);
      if (observation.objective !== manifest.objective || !manifest.supported_player_counts.includes(count)) {
        throw new PolicyCoverageError(`Average-policy coverage unavailable: objective=${observation.objective}, players=${count}; `
          + `model objective=${manifest.objective}, supported players=${manifest.supported_player_counts.join(",")}`);
      }
      const integers = (values: number[]) => new BigInt64Array(values.map(value => BigInt(value)));
      const position = Math.round(observation.numeric[NUMERIC_NAMES.indexOf("hero_position")] * 2);
      const neural = neuralObservation(observation, manifest.feature_schema_version);
      const feeds = {
        cards: new ort.Tensor("int64", integers(neural.cards), [1, 7]),
        street: new ort.Tensor("int64", integers([observation.street]), [1]),
        position: new ort.Tensor("int64", integers([position]), [1]),
        numeric: new ort.Tensor("float32", new Float32Array(neural.numeric), [1, numericNames(manifest.feature_schema_version).length]),
        history: new ort.Tensor("float32", new Float32Array(neural.history.flat()), [1, observation.history.length, HISTORY_WIDTH]),
        mask: new ort.Tensor("bool", new Uint8Array(observation.legal_mask.map(Number)), [1, ACTION_IDS.length]),
      };
      let outputs: Awaited<ReturnType<typeof session.run>> | undefined;
      try {
        outputs = await session.run(feeds);
        const output = outputs.probabilities;
        if (!output || output.type !== "float32" || !equalArray(output.dims, [1, ACTION_IDS.length])) {
          throw new Error("Average-policy ONNX output must be float32 probabilities with shape [1, canonical action count]");
        }
        const probabilities = Array.from(output.data as Float32Array);
        const total = probabilities.reduce((sum, probability) => sum + probability, 0);
        if (probabilities.some((probability, index) => !Number.isFinite(probability) || probability < 0
            || (!observation.legal_mask[index] && probability !== 0)) || Math.abs(total - 1) > 1e-5) {
          throw new Error(`Invalid average-policy probabilities: total=${total}, values=${probabilities.join(",")}`);
        }
        return Object.fromEntries(ACTION_IDS.map((action, index) => [action, probabilities[index]]));
      } finally {
        for (const tensor of Object.values(feeds)) tensor.dispose();
        for (const tensor of Object.values(outputs ?? {})) tensor.dispose();
      }
    },
    async release(): Promise<void> {
      if (released) throw new Error("Average-policy session has already been released");
      await session.release();
      released = true;
    },
  };
}

/** Load the immutable ONNX bundles selected by migrate.py's atomic catalog. */
export async function loadPublishedPolicies(publishedCatalog?: unknown, assetOrigin?: string): Promise<Record<number, LoadedAveragePolicy>> {
  let catalog = publishedCatalog as { version: number; exports: Record<string, { bundle_id: string; iteration: number }> };
  if (publishedCatalog === undefined) {
    catalog = await fetchPolicyCatalog(assetOrigin);
  }
  if (!catalog || catalog.version !== 1 || !catalog.exports || typeof catalog.exports !== "object" || Array.isArray(catalog.exports)
      || Object.keys(catalog.exports).some(track => track !== "3max" && track !== "hu")) {
    throw new Error("Invalid published policy catalog");
  }
  const loaded: Record<number, LoadedAveragePolicy> = {};
  try {
    for (const [track, count] of [["3max", 3], ["hu", 2]] as const) {
      const entry = catalog.exports[track];
      if (!(track in catalog.exports)) continue;
      if (!entry || typeof entry.bundle_id !== "string" || !/^[a-f0-9]{64}$/.test(entry.bundle_id)
          || !Number.isInteger(entry.iteration) || entry.iteration < 1) {
        throw new Error(`Published ${track} policy is invalid; export this track again with migrate.py`);
      }
      const base = policyArtifactBase(track, entry.bundle_id, assetOrigin);
      const manifest = await fetch(`${base}.json`, { signal: AbortSignal.timeout(15000) });
      if (!manifest.ok) throw new Error(`Published ${track} manifest unavailable: HTTP ${manifest.status}`);
      const raw = await manifest.json();
      if (raw.version === 5) {
        loaded[count] = loadStreetPolicies(raw, base, count, entry.iteration, assetOrigin);
      } else {
        const weights = await fetch(`${base}.onnx`, { signal: AbortSignal.timeout(30000) });
        if (!weights.ok) throw new Error(`Published ${track} model unavailable: HTTP ${weights.status}`);
        loaded[count] = await loadAveragePolicy(await weights.arrayBuffer(), raw, assetOrigin);
      }
      if (!equalArray(loaded[count].manifest.supported_player_counts, [count]) || loaded[count].manifest.iteration !== entry.iteration) {
        throw new Error(`Published ${track} policy does not match its catalog entry`);
      }
    }
    return loaded;
  } catch (cause) {
    await Promise.all(Object.values(loaded).map(policy => policy.release()));
    throw cause;
  }
}


const STREETS = ["PREFLOP", "FLOP", "TURN", "RIVER"] as const;
type StreetRoute = {
  player_count: number; street: string; model_kind: string; model_version: number;
  checkpoint_source: string; feature_schema: number; action_schema: number;
  model_hash: string; model_file: string; manifest: PolicyManifest;
};

/** Validate every route before loading any weights; load one session per queried street. */
export function loadStreetPolicies(raw: {
  version: number; layout: string; iteration: number; supported_player_counts: number[];
  routes: Record<string, StreetRoute>;
}, base: string, count: number, iteration: number, assetOrigin?: string): LoadedAveragePolicy {
  if (raw.version !== 5 || raw.layout !== "independent_streets_v1" || raw.iteration !== iteration
      || !equalArray(raw.supported_player_counts, [count]) || !raw.routes
      || !equalArray(Object.keys(raw.routes).sort(), [...STREETS].sort())) {
    throw new Error("Invalid street policy catalog");
  }
  for (const street of STREETS) {
    const route = raw.routes[street];
    const manifest = validatePolicyManifest(route.manifest);
    if (route.player_count !== count || route.street !== street || route.model_kind !== "AVERAGE"
        || !Number.isSafeInteger(route.model_version) || route.model_version < 1 || route.model_version > iteration
        || !/^[a-f0-9]{64}$/.test(route.checkpoint_source) || route.model_hash !== manifest.model_sha256
        || route.feature_schema !== manifest.feature_schema_version || route.action_schema !== manifest.action_schema_version
        || manifest.iteration !== iteration || !equalArray(manifest.supported_player_counts, [count])
        || !/^[a-z0-9_]+\.onnx$/.test(route.model_file)) {
      throw new Error(`Invalid ${count}-player ${street} model route`);
    }
  }
  if (new Set(STREETS.map(street => raw.routes[street].checkpoint_source)).size !== 1
      || new Set(STREETS.map(street => raw.routes[street].model_file)).size !== 4) {
    throw new Error("Street routes must belong to one checkpoint with four distinct artifacts");
  }
  const sessions = new Map<string, Promise<LoadedAveragePolicy>>();
  let released = false;
  return {
    manifest: raw.routes.PREFLOP.manifest,
    async query(observation) {
      if (released) throw new Error("Street policy has been released");
      const street = STREETS[observation.street];
      if (!street) throw new Error(`Invalid public street: ${observation.street}`);
      if (!sessions.has(street)) {
        const route = raw.routes[street];
        sessions.set(street, (async () => {
          const response = await fetch(new URL(route.model_file, base), { signal: AbortSignal.timeout(30000) });
          if (!response.ok) throw new Error(`${street} model unavailable: HTTP ${response.status}`);
          return loadAveragePolicy(await response.arrayBuffer(), route.manifest, assetOrigin);
        })().catch(error => { sessions.delete(street); throw error; }));
      }
      return (await sessions.get(street)!).query(observation);
    },
    async release() {
      if (released) throw new Error("Street policy has already been released");
      released = true;
      const pending = await Promise.allSettled(sessions.values());
      await Promise.all(pending.flatMap(result => result.status === "fulfilled" ? [result.value.release()] : []));
    },
  };
}
