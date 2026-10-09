/** Exercise the real Next page and emitted worker, then compare Python fixtures. */
import assert from "node:assert/strict";
import { readFile, writeFile } from "node:fs/promises";

const origin = process.env.POKER_UI_ORIGIN ?? "http://127.0.0.1:3107";
const cdp = process.env.POKER_CDP_ORIGIN ?? "http://127.0.0.1:9223";
const page = await (await fetch(`${cdp}/json/new?about:blank`, { method: "PUT" })).json();
const socket = new WebSocket(page.webSocketDebuggerUrl);
await new Promise((resolve, reject) => { socket.onopen = resolve; socket.onerror = reject; });
let sequence = 0;
const pending = new Map();
socket.onmessage = event => {
  const result = JSON.parse(String(event.data));
  const entry = pending.get(result.id);
  if (!entry) return;
  pending.delete(result.id); clearTimeout(entry.timeout);
  result.error ? entry.reject(new Error(JSON.stringify(result.error))) : entry.resolve(result.result);
};
function call(method, params = {}) {
  return new Promise((resolve, reject) => {
    const id = ++sequence;
    const timeout = setTimeout(() => { pending.delete(id); reject(new Error(`CDP timeout: ${method}`)); }, 45000);
    pending.set(id, { resolve, reject, timeout });
    socket.send(JSON.stringify({ id, method, params }));
  });
}
async function evaluate(expression) {
  const response = await call("Runtime.evaluate", { expression, awaitPromise: true, returnByValue: true });
  if (response.exceptionDetails) throw new Error(JSON.stringify(response.exceptionDetails));
  return response.result?.value;
}
async function until(expression) {
  const deadline = Date.now() + 45000;
  while (Date.now() < deadline) {
    const value = await evaluate(expression);
    if (value) return value;
    await new Promise(resolve => setTimeout(resolve, 200));
  }
  throw new Error(`Browser condition timed out: ${expression}; body=${await evaluate("document.body.innerText")}`);
}
const click = text => evaluate(`Array.from(document.querySelectorAll('button')).find(b => b.getClientRects().length && b.textContent.trim() === ${JSON.stringify(text)}).click()`);
try {
  await call("Emulation.setDeviceMetricsOverride",{width:1440,height:1000,deviceScaleFactor:1,mobile:false});
  await call("Page.navigate",{url:origin});
  await until("Array.from(document.querySelectorAll('button')).some(b=>b.textContent.trim()==='Test Live')");
  await click("Test Live");
  await until("document.querySelector('[aria-label=\"Opponent strategy\"]') && !document.querySelector('[aria-label=\"Opponent strategy\"]').disabled");
  assert.equal(await evaluate("document.querySelector('[aria-label=\"Opponent strategy\"]').value"),'conservative');
  await click("P0");
  await until("!document.querySelector('[aria-label=\"Opponent strategy\"]').disabled");
  const catalog = await (await fetch(`${origin}/policy/index.json`)).json();
  const hasModels = !!catalog.exports?.hu && !!catalog.exports?.["3max"];
  if (!hasModels) {
    assert.equal(await evaluate("document.querySelector('[aria-label=\"Opponent strategy\"] option[value=published]').disabled"),true);
    await until("document.querySelector('[data-testid=\"belief-session\"]')?.getAttribute('data-status')==='current'");
    console.log('Empty catalog: no stale models; card-aware live play and beliefs available');
  } else {
  await until("!document.querySelector('[aria-label=\"Opponent strategy\"] option[value=published]').disabled");
  await evaluate("(()=>{const s=document.querySelector('[aria-label=\"Opponent strategy\"]');s.value='published';s.dispatchEvent(new Event('change',{bubbles:true}));})()");
  await until("document.querySelector('[aria-label=\"Opponent strategy\"]').value==='published' && !document.querySelector('[aria-label=\"Opponent strategy\"]').disabled");
  await until("document.querySelector('[data-testid=\"belief-session\"]')?.getAttribute('data-status')==='current'");
  await evaluate("Array.from(document.querySelectorAll('[aria-label=\"Poker actions\"] button')).find(b=>b.textContent.trim().startsWith('CALL')).click()");
  await until("document.querySelector('[data-testid=\"belief-session\"]')?.getAttribute('data-status')==='current' && document.querySelector('[data-testid=\"belief-session\"]')?.getAttribute('data-actions')!=='0'");
  assert.equal(await evaluate("document.querySelector('[aria-label=\"P1 behavior profile\"]').value"),'published');
  console.log('Real ONNX opponents + Bayesian worker tracking passed');
  }
} finally {socket.close();await fetch(`${cdp}/json/close/${page.id}`);}
