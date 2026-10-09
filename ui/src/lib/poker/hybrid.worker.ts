import { rangeEquities } from "./range-equity";
import { rangeDisplay } from "./range-display";
import { analyzeHybrid, restoreHand, type HandTransport, type HybridBudget, type Ranges } from "./hybrid";
import { updateRanges, observerMarginals, type BeliefTransition } from "./beliefs";
import { behaviorProbabilities } from "./behavior";
import { observe } from "./observation";
import { ACTION_IDS } from "./actions";
import { PolicyBank } from "../policy-bank";
import type { HandState } from "./engine";

const published = new PolicyBank();
let checkedAt = -Infinity;
const predictionCache = new Map<string,number[]>();
async function predictionsFor(state:HandState, rows:Ranges[number]):Promise<Record<string,number[]>> {
  const model=published.models[Object.keys(state.players).length];
  if(!model) throw new Error("Published opponent policy unavailable for this player count");
  const result:Record<string,number[]>={};
  for(const row of rows) {
    const view=state.clone(); view.actor.cards=row.cards;
    const observation=observe(view), key=JSON.stringify([model.manifest.model_sha256,observation]);
    let probabilities=predictionCache.get(key);
    if(!probabilities) {
      const output=await model.query(observation); probabilities=ACTION_IDS.map(a=>output[a]);
      if(predictionCache.size>=20000) predictionCache.delete(predictionCache.keys().next().value!);
      predictionCache.set(key,probabilities);
    }
    result[row.cards.join()]=probabilities;
  }
  return result;
}

self.onmessage = async (event: MessageEvent<{ requestId?: number; state: HandTransport; ranges: Ranges; budget: HybridBudget;
  searchMode?: "behavior" | "public_cfr" | "response"; profiles?: Record<number,string>; likelihoodProfiles?: Record<number,string>;
  observerSeat?: number; transitions?: BeliefTransition[]; interpolate?: boolean; updateOnly?: boolean }>) => {
  try {
    const started = performance.now(), data = event.data;
    const profiles = data.profiles ?? Object.fromEntries(Object.keys(data.state.players).map(p => [p, "uniform"]));
    const likelihoodProfiles = data.likelihoodProfiles ?? profiles;
    if (Object.values(likelihoodProfiles).includes("published") && Date.now()-checkedAt > 60000) {
      if (await published.refresh()) predictionCache.clear();
      checkedAt = Date.now();
    }
    let ranges=data.ranges;
    for(const transition of data.transitions ?? []) {
      const before=restoreHand(transition.before), profile=likelihoodProfiles[before.current_player!];
      const predictions=profile==="published" ? await predictionsFor(before,ranges[before.current_player!]) : undefined;
      ranges=updateRanges(before,restoreHand(transition.after),ranges,transition.action,profile,data.interpolate??false,predictions);
    }
    const marginals = data.observerSeat === undefined ? undefined : observerMarginals(restoreHand(data.state), ranges, data.observerSeat);
    const stateForDisplay=restoreHand(data.state);
    const reactionPredictions=marginals && !stateForDisplay.terminal && stateForDisplay.current_player!==data.observerSeat
      && likelihoodProfiles[stateForDisplay.current_player!]==="published" ? await predictionsFor(stateForDisplay,marginals[stateForDisplay.current_player!]) : undefined;
    const display = marginals && data.observerSeat !== undefined ? rangeDisplay(restoreHand(data.state), marginals, data.observerSeat, likelihoodProfiles, reactionPredictions) : undefined;
    if (display && data.observerSeat !== undefined) display.equities = rangeEquities(stateForDisplay, ranges, data.observerSeat);
    if (data.updateOnly || data.state.terminal) self.postMessage({ requestId: data.requestId, ranges, marginals, display, workerMilliseconds: performance.now()-started });
    else {
      if(Object.values(profiles).includes("published")) throw new Error("Published policies support opponent play and range tracking; choose Reference analysis for continuation search.");
      const result = analyzeHybrid(restoreHand(data.state), ranges, data.budget, data.searchMode, profiles);
      const state = restoreHand(data.state), probabilities = behaviorProbabilities(observe(state), profiles[state.current_player!]);
      result.baselineProbabilities = Object.fromEntries(ACTION_IDS.map((action,i) => [action,probabilities[i]]));
      self.postMessage({ requestId: data.requestId, result, ranges, marginals, display, workerMilliseconds: performance.now() - started });
    }
  } catch (error) {
    self.postMessage({ requestId: event.data.requestId, error: error instanceof Error ? error.message : String(error) });
  }
};
