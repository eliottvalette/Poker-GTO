import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";

const uiRoot = fileURLToPath(new URL("../", import.meta.url));
const repositoryRoot = fileURLToPath(new URL("../../", import.meta.url));
const mode = process.argv[2];
if (!["dev", "start"].includes(mode)) throw new Error(`Expected dev or start, got ${mode}`);
const engineUrl = "http://127.0.0.1:8765";
if (process.env.POKER_ENGINE_URL && process.env.POKER_ENGINE_URL !== engineUrl) {
  throw new Error(`The integrated application requires POKER_ENGINE_URL=${engineUrl}, got ${process.env.POKER_ENGINE_URL}`);
}
const children = new Set();
let stopping = false;
let ready = false;
let output = "";

function stop(code) {
  if (stopping) return;
  stopping = true;
  process.exitCode = code;
  for (const child of children) child.kill("SIGTERM");
  const timeout = setTimeout(() => {
    for (const child of children) child.kill("SIGKILL");
  }, 5000);
  timeout.unref();
}

function launch(command, args, options) {
  const child = spawn(command, args, options);
  children.add(child);
  child.on("error", error => {
    console.error(`Application process ${command} failed: ${error.message}`);
    children.delete(child);
    stop(1);
  });
  child.on("exit", (code, signal) => {
    children.delete(child);
    if (!stopping) {
      console.error(`Application process ${command} exited: code=${code}, signal=${signal}`);
      stop(code || 1);
    }
  });
  return child;
}

process.on("SIGINT", () => stop(0));
process.on("SIGTERM", () => stop(0));
const startupTimeout = setTimeout(() => {
  console.error("Python poker engine did not become ready within 30 seconds");
  stop(1);
}, 30000);
startupTimeout.unref();

const python = process.env.POKER_PYTHON ?? fileURLToPath(new URL("../../.venv/bin/python", import.meta.url));
const engine = launch(python, ["-u", "-m", "server"], {
  cwd: repositoryRoot, stdio: ["ignore", "pipe", "inherit"], env: process.env,
});
engine.stdout.on("data", chunk => {
  process.stdout.write(chunk);
  output += chunk.toString();
  if (!ready && !stopping && output.includes(`Poker engine listening on ${engineUrl};`)) {
    ready = true;
    clearTimeout(startupTimeout);
    launch(process.execPath, [fileURLToPath(new URL("../node_modules/next/dist/bin/next", import.meta.url)),
      mode, ...(mode === "dev" ? ["--turbopack"] : []), ...process.argv.slice(3)], {
      cwd: uiRoot, stdio: "inherit", env: { ...process.env, POKER_ENGINE_URL: engineUrl },
    });
  }
  if (output.length > 4096) output = output.slice(-4096);
});
