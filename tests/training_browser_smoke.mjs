/** Real static UI rendering with explicit synthetic history transport fixtures. */
import assert from "node:assert/strict";
import { writeFile } from "node:fs/promises";

const cdp = process.env.POKER_CDP_ORIGIN ?? "http://127.0.0.1:9337";
const origin = process.env.POKER_UI_ORIGIN ?? "http://127.0.0.1:3034";
const response = await fetch(`${cdp}/json/new?about:blank`, { method: "PUT" });
if (!response.ok) throw Error(`CDP unavailable: ${response.status}`);
const page = await response.json();
const socket = new WebSocket(page.webSocketDebuggerUrl);
await new Promise((resolve, reject) => { socket.onopen = resolve; socket.onerror = reject; });
let sequence = 0;
const pending = new Map();
socket.onmessage = event => {
  const value = JSON.parse(event.data), task = pending.get(value.id);
  if (!task) return;
  clearTimeout(task.timer); pending.delete(value.id);
  if (value.error) task.reject(Error(JSON.stringify(value.error))); else task.resolve(value.result);
};
const call = (method, params = {}) => new Promise((resolve, reject) => {
  const id = ++sequence;
  const timer = setTimeout(() => { pending.delete(id); reject(Error(`CDP timeout: ${method}`)); }, 20000);
  pending.set(id, { resolve, reject, timer }); socket.send(JSON.stringify({ id, method, params }));
});
async function evaluate(expression) {
  const result = await call("Runtime.evaluate", { expression, returnByValue: true, awaitPromise: true });
  if (result.exceptionDetails) throw Error(JSON.stringify(result.exceptionDetails));
  return result.result.value;
}
async function until(expression) {
  for (let i = 0; i < 100; i++) {
    if (await evaluate(expression)) return;
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  throw Error(`UI timeout: ${expression}; ${await evaluate('document.body.innerText')}`);
}
async function click(label) {
  await evaluate(`(()=>{const b=[...document.querySelectorAll('button')].find(b=>b.textContent.trim()===${JSON.stringify(label)});if(!b)throw Error('Missing button');b.click()})()`);
}
try {
  await call("Page.enable");
  await call("Emulation.setDeviceMetricsOverride", { width: 1360, height: 1050, deviceScaleFactor: 1, mobile: false });
  await call("Page.addScriptToEvaluateOnNewDocument", { source: `
    window.__historyMode='ready'; window.__errors=[]; window.__archives={};
    window.addEventListener('error',e=>window.__errors.push(e.message));
    const original=window.fetch;
    window.fetch=async (...args)=>{
      const url=String(args[0]);
      if(!url.includes('/evaluation/'))return original(...args);
      if(window.__historyMode==='missing')return Promise.resolve(new Response('',{status:404}));
      if(window.__historyMode==='error')return Promise.resolve(new Response('',{status:503}));
      const track=url.includes('/hu/')?'hu':'3max';
      const estimate={n:128,mean:0,m2:0,se:0,interval:[0,0]};
      const run={suite:'hourly-paired-policy-v1',release_id:'a'.repeat(64),model_sha256:'b'.repeat(64),iteration:track==='hu'?42:33,
        evaluated_at:new Date().toISOString(),published_at:new Date().toISOString(),status:'complete',is_reference:true,
        reference:{iteration:1,model_sha256:'b'.repeat(64),suite:'hourly-paired-policy-v1'},completed_groups:128,hands:6144,
        attempted_hands:6144,discarded_hands:0,cpu_seconds:5,wall_seconds:7,gain:estimate,difference:estimate,
        segments:[{profile:'uniform',region:'medium',gain:estimate,difference:estimate}],river:null,warnings:['Synthetic browser test fixture'],units:'BB/100'};
      if(url.endsWith('/index.json')){
        const payload=JSON.stringify({version:1,track,updated_at:new Date().toISOString(),runs:[run]});
        const bytes=await new Response(new Blob([payload]).stream().pipeThrough(new CompressionStream('gzip'))).arrayBuffer();
        window.__archives[track]=bytes;
        const sha256=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),b=>b.toString(16).padStart(2,'0')).join('');
        return Response.json({version:1,track,sha256});
      }
      return new Response(window.__archives[track]);
    };
  ` });
  await call("Page.navigate", { url: origin });
  await until("[...document.querySelectorAll('button')].some(b=>b.textContent==='Training')");
  await until("document.body.innerText.includes('Policies ·') || document.body.innerText.includes('Policy refresh failed')");
  await click("Training");
  await until("document.body.innerText.includes('Iteration 42')");
  assert.equal(await evaluate("document.querySelectorAll('svg[aria-label=\"Policy performance history in big blinds per 100 hands\"]').length"), 1);
  await click("7d"); await click("vs opponent pool");
  await click("3-Max");
  await until("document.body.innerText.includes('Iteration 33')");
  if (process.env.POKER_SCREENSHOT_PATH) {
    const shot = await call("Page.captureScreenshot", { format: "png" });
    await writeFile(process.env.POKER_SCREENSHOT_PATH, Buffer.from(shot.data, "base64"));
  }
  await evaluate("window.__historyMode='missing'"); await click("Refresh");
  await until("document.body.innerText.includes('No evaluations in this period')");
  await evaluate("window.__historyMode='error'"); await click("Refresh");
  await until("document.body.innerText.includes('Training history unavailable: HTTP 503')");
  await evaluate("window.__historyMode='ready'"); await click("Refresh");
  await until("document.body.innerText.includes('Iteration 33')");
  assert.deepEqual(await evaluate("window.__errors"), []);
  console.log("Training UI passed: HU, 3-max, periods, chart, empty state, HTTP failure and recovery.");
} finally {
  socket.close();
  await fetch(`${cdp}/json/close/${page.id}`);
}
