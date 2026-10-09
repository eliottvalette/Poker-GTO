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
    await until("!Array.from(document.querySelectorAll('button')).find(b => b.textContent.trim() === 'Analyze locally')?.disabled");
    await click("Analyze locally");
    const text = await until("Array.from(document.querySelectorAll('[data-testid=\"hybrid-result\"]')).find(e=>e.getClientRects().length)?.textContent");
    assert.match(text, /16 samples/);
    assert.match(text, /EV \(current BB\)/);
    report.live.push({ playerCount: count, displayedResult: text });
  }
  report.completeHands = [];
  for (const count of [2, 3]) {
    await evaluate(`(() => { const select = Array.from(document.querySelectorAll('label')).find(l => l.textContent.startsWith('Players')).querySelector('select'); select.value = '${count}'; select.dispatchEvent(new Event('change', {bubbles:true})); })()`);
    await new Promise(resolve => setTimeout(resolve,100));
    await click("Apply state");
    let decisions = 0;
    while (!(await evaluate("document.body.innerText.includes('Hand settled')"))) {
      await until("Array.from(document.querySelectorAll('[data-testid=\"belief-session\"]')).find(e=>e.getClientRects().length)?.textContent.includes('Ranges ready') && !Array.from(document.querySelectorAll('[aria-label=\"Hybrid analysis\"] [role=\"status\"]')).find(e=>e.getClientRects().length)");
      if (decisions > 16) throw new Error("Full-hand decision bound exceeded");
      if (decisions % count === 0) {
        await click("Analyze locally");
        await until("Array.from(document.querySelectorAll('[data-testid=\"hybrid-result\"]')).find(e=>e.getClientRects().length)?.textContent.includes('EV (current BB)')");
      }
      await evaluate("(() => { const buttons = Array.from(document.querySelectorAll('button')); const action = buttons.find(b => b.textContent.trim() === 'CALL') ?? buttons.find(b => b.textContent.trim() === 'CHECK'); if (!action) throw new Error('No passive continuation'); action.click(); })()");
      decisions++;
      await new Promise(resolve => setTimeout(resolve,100));
    }
    await until("Array.from(document.querySelectorAll('[data-testid=\"belief-session\"]')).find(e=>e.getClientRects().length)?.textContent.includes('Ranges ready')");
    report.completeHands.push({count, decisions, session:await evaluate("Array.from(document.querySelectorAll('[data-testid=\"belief-session\"]')).find(e=>e.getClientRects().length).textContent")});
  }
  await click("Apply state");
  await until("!Array.from(document.querySelectorAll('[aria-label=\"Hybrid analysis\"] [role=\"status\"]')).find(e=>e.getClientRects().length)");
  await evaluate("document.querySelector('[aria-label=\"Hybrid analysis\"] input[type=checkbox]').click()");
  await until("!Array.from(document.querySelectorAll('[aria-label=\"Hybrid analysis\"] [role=\"status\"]')).find(e=>e.getClientRects().length)");
  await click("Apply actual raise");
  await until("Array.from(document.querySelectorAll('[data-testid=\"belief-session\"]')).find(e=>e.getClientRects().length)?.getAttribute('data-actions') === '1' && Array.from(document.querySelectorAll('[data-testid=\"belief-session\"]')).find(e=>e.getClientRects().length)?.textContent.includes('Ranges ready')");
  await evaluate("(() => { const s=document.querySelector('[aria-label=\"Decision mode\"]'); s.value='exploitative'; s.dispatchEvent(new Event('change',{bubbles:true})); })()");
  await until("!Array.from(document.querySelectorAll('[aria-label=\"Hybrid analysis\"] [role=\"status\"]')).find(e=>e.getClientRects().length)");
  await click("Analyze locally");
  await until("Array.from(document.querySelectorAll('[data-testid=\"hybrid-result\"]')).find(e=>e.getClientRects().length)?.textContent.includes('Computed behavior baseline')");
  report.offTree = await evaluate("Array.from(document.querySelectorAll('[data-testid=\"belief-session\"]')).find(e=>e.getClientRects().length).textContent");
  await evaluate("(() => { const s=Array.from(document.querySelectorAll('[aria-label=\"Hybrid analysis\"] label')).find(l=>l.textContent.startsWith('Calculation')).querySelector('select'); s.value='DEEP'; s.dispatchEvent(new Event('change',{bubbles:true})); })()");
  await until("!Array.from(document.querySelectorAll('[aria-label=\"Hybrid analysis\"] [role=\"status\"]')).find(e=>e.getClientRects().length)");
  await click("Analyze locally");
  await until("Array.from(document.querySelectorAll('button')).some(b=>b.textContent.trim()==='Cancel calculation')");
  await click("Cancel calculation");
  await until("document.querySelector('[aria-label=\"Hybrid analysis\"] [role=alert]')?.textContent.includes('cancelled')");
  report.cancellation = await evaluate("document.querySelector('[aria-label=\"Hybrid analysis\"] [role=alert]').textContent");
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
  await click("Test Live");
  await until("Array.from(document.querySelectorAll('button')).some(b=>b.getClientRects().length && b.textContent.trim()==='P0' && !b.disabled)");
  await click("P0");
  await until("Array.from(document.querySelectorAll('button')).some(b=>b.getClientRects().length && b.textContent.trim()==='P0' && b.getAttribute('aria-pressed')==='true' && !b.disabled)");
  await until("Array.from(document.querySelectorAll('[data-testid=\"belief-session\"]')).find(e=>e.getClientRects().length)?.textContent.includes('Ranges ready') && Array.from(document.querySelectorAll('button')).some(b=>b.getClientRects().length && b.textContent.trim()==='Analyze locally' && !b.disabled)");
  await click("Analyze locally");
  await until("Array.from(document.querySelectorAll('[data-testid=\"hybrid-result\"]')).find(e=>e.getClientRects().length)?.textContent.includes('EV (current BB)')");
  await evaluate("Array.from(document.querySelectorAll('[aria-label=\"Poker actions\"] button')).find(b=>b.textContent.trim().startsWith('CALL')).click()");
  await until("(() => { const e=Array.from(document.querySelectorAll('[data-testid=\"belief-session\"]')).find(e=>e.getClientRects().length); return e?.textContent.includes('Ranges ready') && e.getAttribute('data-actions') !== '0'; })()");
  report.liveTable = await evaluate("Array.from(document.querySelectorAll('[data-testid=\"belief-session\"]')).find(e=>e.getClientRects().length).textContent");
  report.rangeHover = [];
  const inspectors = await evaluate("Array.from(document.querySelectorAll('button[aria-label^=\"Inspect player\"]')).filter(b=>b.getClientRects().length).map(b=>b.getAttribute('aria-label'))");
  assert.ok(inspectors.length >= 1);
  for (const label of inspectors) {
    const rect = await evaluate(`(() => { const e=document.querySelector('button[aria-label="${label}"]'); e.scrollIntoView({block:'center'}); const r=e.getBoundingClientRect(); return {x:r.x+r.width/2,y:r.y+r.height/2}; })()`);
    await call("Input.dispatchMouseEvent", {type:"mouseMoved", ...rect});
    await until("document.querySelector('[aria-label=\"169 hand classes\"]')?.querySelectorAll('button').length === 169");
    const panel = await evaluate("document.querySelector('[role=dialog][aria-label$=\"range inspector\"]')?.textContent");
    assert.match(panel,/Estimated range/);
    assert.match(panel,/Less likely/);
    assert.match(panel,/More likely/);
    assert.ok(await evaluate("document.querySelector('[aria-label=\"Range color legend\"]') !== null"));
    await evaluate("document.querySelector('[aria-label=\"169 hand classes\"] button').click()");
    await until("document.querySelector('[role=dialog] details')?.open");
    const screenshot = await call("Page.captureScreenshot", {format:"png"});
    await writeFile((process.env.POKER_REPORT_PATH ?? "artifacts/phase2/evaluation/browser-smoke.json").replace(/\.json$/, `-${report.rangeHover.length}.png`), Buffer.from(screenshot.data,"base64"));
    report.rangeHover.push({label, classes:169, combos:await evaluate("document.querySelector('[role=dialog] summary').textContent")});
    await evaluate("document.querySelector('[aria-label=\"Close range inspector\"]').click()");
  }
  await writeFile(process.env.POKER_REPORT_PATH ?? "artifacts/phase2/evaluation/browser-smoke.json", JSON.stringify(report, null, 2) + "\n");
  console.log(JSON.stringify(report.fixtureParity));
} finally {
  socket.close();
  await fetch(`${cdp}/json/close/${page.id}`);
}
