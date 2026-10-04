import { ACTION_IDS } from "./poker/actions";
import { EVENTS, HISTORY_WIDTH, NUMERIC_NAMES, POSITIONS, STATE_VERSION,
  validateObservation, type Observation } from "./poker/observation";

const INPUTS = ["cards", "street", "position", "numeric", "history", "mask"] as const;
const ARCHITECTURE = "cards8_numeric32_historyGRU32_head64_v2";

export type PolicyManifest = {
  version: number;
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
  const fields = ["version", "state_version", "architecture", "model_sha256", "objective", "iteration",
    "supported_player_counts", "actions", "numeric_names", "positions", "events", "normalization_bb",
    "card_encoding", "card_slots", "unknown_card", "card_vocabulary", "history_width", "full_history",
    "batch_size", "inputs", "output", "validation_max_absolute_error"];
  if (!equalArray(Object.keys(raw).sort(), fields.sort()) || manifest.version !== 1 || manifest.state_version !== STATE_VERSION
      || manifest.architecture !== ARCHITECTURE || typeof manifest.model_sha256 !== "string"
      || !/^[a-f0-9]{64}$/.test(manifest.model_sha256) || !["tournament_winner", "hand_chip_delta"].includes(manifest.objective)
      || !Number.isInteger(manifest.iteration) || manifest.iteration < 1 || !Array.isArray(manifest.supported_player_counts)
      || !manifest.supported_player_counts.length || manifest.supported_player_counts.some(count => count !== 2 && count !== 3)
      || new Set(manifest.supported_player_counts).size !== manifest.supported_player_counts.length
      || !equalArray(manifest.actions, ACTION_IDS) || !equalArray(manifest.numeric_names, NUMERIC_NAMES)
      || !equalArray(manifest.positions, POSITIONS) || !equalArray(manifest.events, EVENTS)
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

/** User-selected average-policy artifact; no model is loaded or invented implicitly. */
export async function loadAveragePolicy(model: ArrayBuffer, rawManifest: unknown): Promise<LoadedAveragePolicy> {
  const manifest = validatePolicyManifest(rawManifest);
  if (!model.byteLength) throw new Error("Average-policy ONNX model is empty");
  const hash = Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", model)),
    value => value.toString(16).padStart(2, "0")).join("");
  if (hash !== manifest.model_sha256) throw new Error(`ONNX model SHA256 ${hash} does not match selected manifest ${manifest.model_sha256}`);
  const ort = await import("onnxruntime-web/wasm");
  ort.env.wasm.wasmPaths = "/onnxruntime/";
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
      const feeds = {
        cards: new ort.Tensor("int64", integers(observation.cards), [1, 7]),
        street: new ort.Tensor("int64", integers([observation.street]), [1]),
        position: new ort.Tensor("int64", integers([position]), [1]),
        numeric: new ort.Tensor("float32", new Float32Array(observation.numeric), [1, NUMERIC_NAMES.length]),
        history: new ort.Tensor("float32", new Float32Array(observation.history.flat()), [1, observation.history.length, HISTORY_WIDTH]),
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
