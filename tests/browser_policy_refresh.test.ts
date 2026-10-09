import test from "node:test";
import assert from "node:assert/strict";
import { PolicyBank } from "../ui/src/lib/policy-bank";
import { fetchPolicyCatalog, policyArtifactBase } from "../ui/src/lib/policy-catalog";
import type { LoadedAveragePolicy, PolicyManifest } from "../ui/src/lib/onnx-policy";
import type { Observation } from "../ui/src/lib/poker/observation";

test("policy refresh swaps validated models, preserves in-flight queries, and bounds live sessions", async () => {
  let version = 1, releases = 0, loads = 0;
  let finish: (() => void) | undefined;
  const bank = new PolicyBank(async () => ({version: 1, exports: {hu: {bundle_id: String(version), iteration: version}}}),
    async () => {
      loads++; const mine = version;
      if (mine === 3) throw new Error("Bad model hash");
      const model: LoadedAveragePolicy = {
        manifest: { iteration: mine } as PolicyManifest,
        async query() { if (mine === 1) await new Promise<void>(resolve => {finish=resolve;}); return {CALL: mine}; },
        async release() { releases++; },
      };
      return {2: model};
    });
  await bank.refresh();const stable=bank.models[2];
  const old=stable.query({} as Observation);
  version=2;await bank.refresh();assert.equal(releases,0);
  assert.equal(stable.manifest.iteration,2);
  assert.deepEqual(await stable.query({} as Observation),{CALL:2});
  finish!();assert.deepEqual(await old,{CALL:1});assert.equal(releases,1);
  await bank.refresh();assert.equal(loads,2);
  version=3;await assert.rejects(bank.refresh(),/Bad model hash/);
  assert.equal(stable.manifest.iteration,2);
  await bank.close();assert.equal(releases,2);
});

test("Supabase pointers are independent, public, and route each track to immutable artifacts", async () => {
  const previous={...process.env};const original=globalThis.fetch;
  process.env.NEXT_PUBLIC_POLICY_SOURCE="supabase";
  process.env.NEXT_PUBLIC_SUPABASE_URL="https://test.supabase.co";
  process.env.NEXT_PUBLIC_SUPABASE_POLICY_BUCKET="poker-policies";
  const requests:string[]=[];
  globalThis.fetch=async (input) => {
    const url=String(input);requests.push(url);const track=url.includes('/hu/')?'hu':'3max';
    return new Response(JSON.stringify({version:1,track,release_id:(track==='hu'?'a':'b').repeat(64),iteration:9}));
  };
  try {
    const catalog=await fetchPolicyCatalog();
    assert.deepEqual(Object.keys(catalog.exports).sort(),['3max','hu']);
    assert.equal(requests.length,2);
    assert.match(policyArtifactBase('hu',catalog.exports.hu.bundle_id),/object\/public\/poker-policies\/hu\/releases\/a{64}\/average_hu$/);
    globalThis.fetch=async () => new Response('unavailable',{status:503});
    await assert.rejects(fetchPolicyCatalog(),/HTTP 503/);
  } finally { globalThis.fetch=original;process.env=previous; }
});

test("updating HU does not reload the unchanged 3-max policy", async () => {
  let hu=1;const loaded:string[][]=[];let released=0;
  const bank=new PolicyBank(async()=>({version:1,exports:{hu:{bundle_id:String(hu),iteration:hu},'3max':{bundle_id:'fixed',iteration:1}}}),
    async catalog=>{
      loaded.push(Object.keys(catalog.exports).sort());
      return Object.fromEntries(Object.entries(catalog.exports).map(([track,entry])=>[track==='hu'?2:3,{
        manifest:{iteration:entry.iteration} as PolicyManifest,async query(){return {CALL:1};},async release(){released++;},
      }]));
    });
  await bank.refresh();hu=2;await bank.refresh();
  assert.deepEqual(loaded,[['3max','hu'],['hu']]);assert.equal(released,1);
  await bank.close();assert.equal(released,3);
});
