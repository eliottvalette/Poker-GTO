import {test} from "node:test";
import assert from "node:assert/strict";
import {parseSolverHistory} from "../ui/src/lib/solver-history";
const history = {engine:"solver_knowledge_v1", track:"hu", updated_at:"2026-10-16T12:00:00Z", runs:[{
  at:"2026-10-16T12:00:00Z",visits:4,tables:4,labels:4,resolved_labels:0,failed_roots:0,nodes:100,cpu_seconds:1,student_accepted:false,
  streets:["PREFLOP","FLOP","TURN","RIVER"].map(street=>({street,response_shortfall_bb:.1,river_response_gain_bb:null,iterations:32}))}]};
test("solver history validates explicit track, engine and all street measurements",()=>{
 assert.equal(parseSolverHistory(history,"hu").runs.length,1);
 assert.throws(()=>parseSolverHistory(history,"3max"));
 assert.throws(()=>parseSolverHistory({...history,engine:"independent_streets_v1"},"hu"));
 assert.throws(()=>parseSolverHistory({...history,runs:[{...history.runs[0],streets:[]}]},"hu"));
 assert.throws(()=>parseSolverHistory({...history,runs:[{...history.runs[0],resolved_labels:99}]},"hu"));
});
