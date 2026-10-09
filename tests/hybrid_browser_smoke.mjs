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
const click = text => evaluate(`Array.from(document.querySelectorAll('button')).find(b => b.textContent.trim() === ${JSON.stringify(text)}).click()`);
const report = { origin, live: [], fixtureParity: [] };
try {
  await call("Emulation.setDeviceMetricsOverride", { width: 1440, height: 1000, deviceScaleFactor: 1, mobile: false });
  await call("Page.navigate", { url: origin });
  await until("Array.from(document.querySelectorAll('button')).some(b => b.textContent.trim() === 'Specific spot')");
  await until("(() => { const button = Array.from(document.querySelectorAll('button')).find(b => b.textContent.trim() === 'Specific spot'); button?.click(); return document.querySelector('[aria-label=\"Hybrid analysis\"]') !== null; })()");
  await evaluate(`window.__hybridNativeWorker = window.Worker; window.Worker = class extends window.__hybridNativeWorker {
    constructor(url, options) { super(url, options); window.__hybridWorkerUrl = String(url); window.__hybridWorkerOptions = options; }
  }`);
  for (const count of [3, 2]) {
    await evaluate(`(() => { const select = Array.from(document.querySelectorAll('label')).find(l => l.textContent.startsWith('Players')).querySelector('select'); select.value = '${count}'; select.dispatchEvent(new Event('change', {bubbles:true})); })()`);
    await new Promise(resolve => setTimeout(resolve, 100));
    await click("Apply state");
    await new Promise(resolve => setTimeout(resolve, 100));
    await click("Analyze locally");
    const text = await until("document.querySelector('[data-testid=\"hybrid-result\"]')?.innerText");
    assert.match(text, /16 samples/);
    assert.match(text, /EV \(current BB\)/);
    report.live.push({ playerCount: count, displayedResult: text });
  }
  const fixtures = JSON.parse(await readFile("tests/fixtures/hybrid_parity.json", "utf8"));
  for (const fixture of fixtures) {
    const message = JSON.stringify({ state: fixture.state, ranges: fixture.ranges, budget: fixture.budget, searchMode: fixture.searchMode });
    const response = await evaluate(`new Promise((resolve, reject) => {
      const worker = new window.__hybridNativeWorker(window.__hybridWorkerUrl, window.__hybridWorkerOptions);
      worker.onmessage = event => { worker.terminate(); resolve(event.data); };
      worker.onerror = event => { worker.terminate(); reject(new Error(event.message)); };
      worker.postMessage(${message});
    })`);
    assert.ok(!response.error, response.error);
    let gap = 0;
    for (const action of Object.keys(fixture.expected.ev)) {
      gap = Math.max(gap, Math.abs(response.result.ev[action] - fixture.expected.ev[action]),
        Math.abs(response.result.probabilities[action] - fixture.expected.probabilities[action]));
    }
    assert.ok(gap < 1e-10);
    report.fixtureParity.push({ playerCount: fixture.count, street: fixture.street, maximumAbsoluteError: gap,
      pythonMilliseconds: fixture.pythonSeconds * 1000, workerMilliseconds: response.workerMilliseconds });
  }
  await writeFile(process.env.POKER_REPORT_PATH ?? "docs/hybrid-browser-smoke.json", JSON.stringify(report, null, 2) + "\n");
  console.log(JSON.stringify(report.fixtureParity));
} finally {
  socket.close();
  await fetch(`${cdp}/json/close/${page.id}`);
}
