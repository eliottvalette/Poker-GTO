/** Static UI/ONNX smoke test. Start Chrome with remote debugging and the UI separately. */
import assert from "node:assert/strict";
import { access, mkdir, writeFile } from "node:fs/promises";
import { join, resolve } from "node:path";

const cdpOrigin = process.env.POKER_CDP_ORIGIN ?? "http://127.0.0.1:9223";
const uiOrigin = process.env.POKER_UI_ORIGIN ?? "http://127.0.0.1:3000";
const screenshotDirectory = process.env.POKER_SCREENSHOT_DIR;
const modelPath = process.env.POKER_ONNX_MODEL;
const manifestPath = process.env.POKER_ONNX_MANIFEST;
if (!modelPath || !manifestPath) {
  throw new Error("Set POKER_ONNX_MODEL and POKER_ONNX_MANIFEST to the exported ONNX model and JSON manifest paths");
}
const files = [resolve(modelPath), resolve(manifestPath)];
for (const path of files) await access(path);
if (screenshotDirectory) await mkdir(resolve(screenshotDirectory), { recursive: true });

async function cdpEndpoint(path, method = "GET") {
  const response = await fetch(new URL(path, cdpOrigin), { method });
  if (!response.ok) throw new Error(`Chrome CDP ${method} ${path}: HTTP ${response.status}`);
  return response;
}

// A dedicated page keeps this test isolated from other tabs in the debug browser.
const page = await (await cdpEndpoint("/json/new?about:blank", "PUT")).json();
if (!page.webSocketDebuggerUrl || !page.id) throw new Error("Chrome CDP returned no page debugger URL or target ID");
const ws = new WebSocket(page.webSocketDebuggerUrl);
let id = 0;
const pending = new Map();
let closed = false;
function rejectPending(message) {
  for (const task of pending.values()) {
    clearTimeout(task.timeout);
    task.reject(new Error(message));
  }
  pending.clear();
}
ws.addEventListener("close", () => { closed = true; rejectPending("Chrome CDP connection closed"); });
ws.addEventListener("error", () => rejectPending("Chrome CDP WebSocket error"));
ws.addEventListener("message", event => {
  const message = JSON.parse(String(event.data));
  if (!message.id) return;
  const task = pending.get(message.id);
  if (!task) return;
  pending.delete(message.id);
  clearTimeout(task.timeout);
  if (message.error) task.reject(new Error(JSON.stringify(message.error)));
  else task.resolve(message.result);
});

function call(method, params = {}) {
  if (closed || ws.readyState !== WebSocket.OPEN) return Promise.reject(new Error("Chrome CDP is not connected"));
  return new Promise((resolveCall, reject) => {
    const requestId = ++id;
    const timeout = setTimeout(() => {
      pending.delete(requestId);
      reject(new Error(`Chrome CDP request timed out: ${method}`));
    }, 10000);
    pending.set(requestId, { resolve: resolveCall, reject, timeout });
    ws.send(JSON.stringify({ id: requestId, method, params }));
  });
}
async function evaluate(expression) {
  const result = await call("Runtime.evaluate", { expression, returnByValue: true, awaitPromise: true });
  if (result.exceptionDetails) throw new Error(JSON.stringify(result.exceptionDetails));
  return result.result.value;
}
async function wait(expression) {
  for (let attempt = 0; attempt < 100; attempt++) {
    if (await evaluate(expression)) return;
    await new Promise(resolveWait => setTimeout(resolveWait, 100));
  }
  throw new Error(`Browser condition timed out: ${expression}`);
}
async function click(text) {
  const literal = JSON.stringify(text);
  await wait(`[...document.querySelectorAll('button')].some(b=>b.textContent.trim()===${literal}&&!b.disabled)`);
  const point = await evaluate(`(()=>{
    const button=[...document.querySelectorAll('button')].find(b=>b.textContent.trim()===${literal}&&!b.disabled);
    if(!button)throw Error('Missing enabled button '+${literal});
    button.scrollIntoView({block:'center'});
    const rect=button.getBoundingClientRect();return {x:rect.x+rect.width/2,y:rect.y+rect.height/2};
  })()`);
  await call("Input.dispatchMouseEvent", { type: "mousePressed", button: "left", clickCount: 1, ...point });
  await call("Input.dispatchMouseEvent", { type: "mouseReleased", button: "left", clickCount: 1, ...point });
}
async function screenshot(name) {
  if (!screenshotDirectory) return;
  const result = await call("Page.captureScreenshot", { format: "png", captureBeyondViewport: false });
  await writeFile(join(resolve(screenshotDirectory), `${name}.png`), Buffer.from(result.data, "base64"));
}

try {
  await new Promise((resolveOpen, reject) => {
    const timeout = setTimeout(() => reject(new Error("Chrome CDP WebSocket connection timed out")), 10000);
    ws.addEventListener("open", () => { clearTimeout(timeout); resolveOpen(); }, { once: true });
    ws.addEventListener("error", () => { clearTimeout(timeout); reject(new Error("Chrome CDP WebSocket connection failed")); }, { once: true });
  });
  await call("Page.enable");
  await call("Emulation.setDeviceMetricsOverride", { width: 1440, height: 1100, deviceScaleFactor: 1, mobile: false });
  await call("Page.addScriptToEvaluateOnNewDocument", { source: `
    window.__requests=[];window.__errors=[];
    const originalFetch=window.fetch;
    window.fetch=(...args)=>{window.__requests.push(String(args[0]));return originalFetch(...args)};
    window.addEventListener('error',e=>window.__errors.push(e.message));
    window.addEventListener('unhandledrejection',e=>window.__errors.push(String(e.reason)));
    Date.now=()=>41;
  ` });
  await call("Page.navigate", { url: uiOrigin });
  await wait("[...document.querySelectorAll('button')].some(b=>b.textContent.trim()==='New game'&&!b.disabled)");
  assert.ok(await evaluate("Array.isArray(window.__requests)&&Array.isArray(window.__errors)"), "Browser error/request instrumentation must be installed");
  await click("P0");
  await wait("!!document.querySelector('[aria-label=\"Poker actions\"] button:not(:disabled)')");
  await screenshot("ready");

  const document = await call("DOM.getDocument");
  const { nodeId } = await call("DOM.querySelector", { nodeId: document.root.nodeId, selector: "input[type=file]" });
  assert.ok(nodeId, "Policy file input must exist");
  await call("DOM.setFileInputFiles", { nodeId, files });
  await wait("document.body.innerText.includes('Experimental policy') || !!document.querySelector('[role=alert]')");
  assert.equal(await evaluate("document.querySelector('[role=alert]')?.textContent??null"), null);
  assert.ok(await evaluate("document.querySelector('[aria-label=\"Poker actions\"]').innerText.includes('%')"), "Loaded policy must display action probabilities");
  await screenshot("onnx-policy");

  let interactions = 0;
  for (; interactions < 10; interactions++) {
    const text = await evaluate(`(()=>{
      const action=[...document.querySelectorAll('[aria-label="Poker actions"] button')].find(b=>!b.disabled);
      if(action)return action.textContent.trim();
      return [...document.querySelectorAll('button')].find(b=>b.textContent.trim()==='Next hand'&&!b.disabled)?.textContent.trim()??null;
    })()`);
    if (!text) break;
    await click(text);
    await wait("![...document.querySelectorAll('button')].find(b=>b.textContent.trim()==='New game').disabled");
  }
  assert.ok(interactions > 0, "At least one action or hand transition must complete");
  const widths = [1440, 390, 320];
  for (const width of widths) {
    await call("Emulation.setDeviceMetricsOverride", { width, height: 1100, deviceScaleFactor: 1, mobile: false });
    const scrollWidth = await evaluate("document.documentElement.scrollWidth");
    assert.ok(scrollWidth <= width, `Horizontal overflow at ${width}px: ${scrollWidth}px`);
    await screenshot(`viewport-${width}`);
  }
  const results = await evaluate("({errors:window.__errors,requests:window.__requests,alert:document.querySelector('[role=alert]')?.textContent??null})");
  assert.deepEqual(results.errors, []);
  assert.equal(results.alert, null);
  assert.ok(results.requests.every(url => !url.includes("/api/") && !url.includes("8765")), "Static play must not request a poker backend");
  console.log(JSON.stringify({ ...results, interactions, tested_viewport_widths: widths }));
} finally {
  rejectPending("Browser smoke test finished");
  ws.close();
  await cdpEndpoint(`/json/close/${encodeURIComponent(page.id)}`);
}
