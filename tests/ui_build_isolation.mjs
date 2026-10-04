import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { access } from "node:fs/promises";
import { fileURLToPath } from "node:url";

// Run against an already running development server. This test builds production.
const origin = process.env.POKER_UI_ORIGIN ?? "http://127.0.0.1:3100";
const uiRoot = fileURLToPath(new URL("../ui/", import.meta.url));
let refreshes = 0;

async function refresh() {
  const page = await fetch(`${origin}/?refresh_check=${refreshes}`, {
    cache: "no-store", signal: AbortSignal.timeout(10000),
  });
  assert.equal(page.status, 200, "Development page must survive a production build");
  const html = await page.text();
  assert.match(html, /Poker table/);
  const scripts = [...html.matchAll(/<script[^>]+src="([^"]+)"/g)].map(match => match[1]);
  assert.ok(scripts.length > 0, "Page must reference client scripts");
  await Promise.all(["/_next/static/development/_buildManifest.js", ...scripts].map(async path => {
    const response = await fetch(new URL(path, origin), { cache: "no-store", signal: AbortSignal.timeout(10000) });
    assert.equal(response.status, 200, `Missing development asset: ${path}`);
    await response.arrayBuffer();
  }));
  refreshes += 1;
}

await refresh();
const child = spawn(process.execPath, ["node_modules/next/dist/bin/next", "build", "--turbopack"], {
  cwd: uiRoot, stdio: ["ignore", "pipe", "pipe"],
});
let output = "";
let finished = false;
let exitCode;
let spawnError;
child.stdout.on("data", chunk => { output += chunk; });
child.stderr.on("data", chunk => { output += chunk; });
child.on("error", error => { spawnError = error; finished = true; });
child.on("close", code => { exitCode = code; finished = true; });
const deadline = Date.now() + 90000;
try {
  while (!finished) {
    assert.ok(Date.now() < deadline, "Build exceeded the 90-second test budget");
    await refresh();
    await new Promise(resolve => setTimeout(resolve, 300));
  }
  if (spawnError) throw spawnError;
  assert.equal(exitCode, 0, output);
  for (let i = 0; i < 5; i++) await refresh();
  await access(new URL("../ui/.next-dev/static/development/_buildManifest.js", import.meta.url));
  await access(new URL("../ui/.next/BUILD_ID", import.meta.url));
  console.log(JSON.stringify({ refreshes, productionBuild: "passed", developmentAssets: "intact" }));
} finally {
  if (!finished) child.kill("SIGTERM");
}
