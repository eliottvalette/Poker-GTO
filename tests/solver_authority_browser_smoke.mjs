/** Exercise real worker decisions and assert that Test Live displays their EVs. */
import assert from "node:assert/strict";
import { readFile, writeFile } from "node:fs/promises";

const origin = process.env.POKER_UI_ORIGIN ?? "http://127.0.0.1:3149";
const cdp = process.env.POKER_CDP_ORIGIN ?? "http://127.0.0.1:9449";
const page = await (await fetch(`${cdp}/json/new?about:blank`, { method: "PUT" })).json();
const socket = new WebSocket(page.webSocketDebuggerUrl);
await new Promise((resolve, reject) => { socket.onopen = resolve; socket.onerror = reject; });
let sequence = 0;
const pending = new Map();
socket.onmessage = event => {
  const response = JSON.parse(String(event.data)), task = pending.get(response.id);
  if (!task) return;
  pending.delete(response.id); clearTimeout(task.timer);
  response.error ? task.reject(new Error(JSON.stringify(response.error))) : task.resolve(response.result);
};
function call(method, params = {}) {
  return new Promise((resolve, reject) => {
    const id = ++sequence;
    const timer = setTimeout(() => reject(new Error(`CDP timeout: ${method}`)), 60000);
    pending.set(id, { resolve, reject, timer }); socket.send(JSON.stringify({ id, method, params }));
  });
}
async function evaluate(expression) {
  const response = await call("Runtime.evaluate", { expression, awaitPromise: true, returnByValue: true });
  if (response.exceptionDetails) throw new Error(JSON.stringify(response.exceptionDetails));
  return response.result.value;
}
async function until(expression) {
  const deadline = Date.now() + 60000;
  while (Date.now() < deadline) {
    const value = await evaluate(`Boolean(${expression})`);
    if (value) return value;
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  throw new Error(`Timeout ${expression}: ${await evaluate("document.body.innerText")}`);
}
const report = { live: [], parity: [] };
try {
  await call("Page.enable");
  await call("Page.addScriptToEvaluateOnNewDocument", { source: `
    window.__calculated=[]; window.__workerUrl=null; window.__failures=[];
    window.addEventListener('error',e=>window.__failures.push(e.message));
    const Native=window.Worker; window.Worker=class extends Native {
      constructor(url, options) { super(url, options); window.__workerUrl=String(url); window.__workerOptions=options;
        this.addEventListener('message',e=>{if(e.data.result)window.__calculated.push({result:e.data.result,ms:e.data.workerMilliseconds});});
      }
      postMessage(...args) { if(window.__delayCompute)this.__timer=setTimeout(()=>super.postMessage(...args),1000);else super.postMessage(...args); }
      terminate() { clearTimeout(this.__timer); super.terminate(); }
    };
  ` });
  await call("Page.navigate", { url: origin });
  await until("Array.from(document.querySelectorAll('button')).find(b=>b.textContent.trim()==='Test Live')");
  await evaluate("Array.from(document.querySelectorAll('button')).find(b=>b.textContent.trim()==='Test Live').click()");
  for (const count of [3, 2]) {
    if (count === 2) {
      await until("document.querySelector('[aria-label=\"Table format\"] button:last-child:not(:disabled)')");
      await evaluate("document.querySelector('[aria-label=\"Table format\"] button:last-child').click()");
    }
    // Published opponents can end a hand before Hero acts; advance explicitly.
    await until(`(() => {
      const buttons=Array.from(document.querySelectorAll('[aria-label="Poker actions"] button:not(:disabled)'));
      if(buttons.some(b=>b.innerText.includes('%'))&&Object.keys(window.__calculated.at(-1)?.result.ranges??{}).length===${count})return true;
      const next=Array.from(document.querySelectorAll('button')).find(b=>b.textContent==='Next hand'&&!b.disabled);
      next?.click(); return false;
    })()`);
    const state = await evaluate(`({buttons:Array.from(document.querySelectorAll('[aria-label="Poker actions"] button')).map(b=>b.innerText),calculation:window.__calculated.at(-1),status:Array.from(document.querySelectorAll('[role="status"]')).map(e=>e.innerText).find(text=>/Monte Carlo|Local CFR/.test(text))})`);
    assert.ok(state.calculation.result.nodes > 0);
    assert.ok(state.buttons.every(text => /%.*BB/s.test(text)));
    assert.match(state.status, /Monte Carlo|Local CFR/);
    report.live.push({ count, buttons: state.buttons, milliseconds: state.calculation.ms,
      nodes: state.calculation.result.nodes, method: state.calculation.result.method });
    await evaluate(`(() => {const buttons=Array.from(document.querySelectorAll('[aria-label="Poker actions"] button:not(:disabled)'));(buttons.find(b=>b.textContent.startsWith('FOLD'))??buttons[0]).click();})()`);
    await until(`(() => {
      if(Array.from(document.querySelectorAll('button')).some(b=>b.textContent==='Next hand'&&!b.disabled))return true;
      const actions=Array.from(document.querySelectorAll('[aria-label="Poker actions"] button:not(:disabled)'));
      (actions.find(b=>b.textContent.startsWith('FOLD'))??actions.find(b=>b.textContent.startsWith('CHECK'))??actions[0])?.click();
      return false;
    })()`);
  }
  await evaluate("window.__delayCompute=true");
  await until(`(() => {
    const cancel=Array.from(document.querySelectorAll('button')).find(b=>b.textContent==='Cancel');
    if(cancel){cancel.click();return true;}
    const next=Array.from(document.querySelectorAll('button')).find(b=>b.textContent==='Next hand'&&!b.disabled);next?.click();return false;
  })()`);
  await until("document.body.innerText.includes('Calculation cancelled')");
  assert.equal(await evaluate("Array.from(document.querySelectorAll('[aria-label=\"Poker actions\"] button')).some(b=>b.innerText.includes('%'))"), false);
  await evaluate("window.__delayCompute=false");
  report.cancellation = "No stale probabilities after cancellation";
  const fixtures = JSON.parse(await readFile("tests/fixtures/hybrid_parity.json", "utf8"));
  for (const fixture of fixtures) {
    const profiles = Object.fromEntries(Object.keys(fixture.ranges).map(p => [p, "conservative"]));
    const response = await evaluate(`new Promise((resolve,reject)=>{
      const worker=new Worker(window.__workerUrl,window.__workerOptions);
      worker.onmessage=e=>{worker.terminate();resolve(e.data);};worker.onerror=e=>{worker.terminate();reject(Error(e.message));};
      worker.postMessage(${JSON.stringify({ requestId: 1, state: fixture.state, ranges: fixture.ranges,
        budget: fixture.budget, searchMode: fixture.searchMode, profiles, likelihoodProfiles: profiles })});
    })`);
    assert.equal(response.error, undefined);
    for (const [action, value] of Object.entries(fixture.expected.ev)) {
      assert.ok(Math.abs(response.result.ev[action] - value) < 1e-10);
      assert.ok(Math.abs(response.result.probabilities[action] - fixture.expected.probabilities[action]) < 1e-10);
    }
    report.parity.push({ count: fixture.count, street: fixture.street, nodes: response.result.nodes });
  }
  assert.deepEqual(await evaluate("window.__failures"), []);
  await writeFile("runs/solver-authority/browser.json", JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report));
} finally {
  for (const task of pending.values()) clearTimeout(task.timer);
  socket.close();
}
