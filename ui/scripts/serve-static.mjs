import { createServer } from "node:http";
import { readFile, stat } from "node:fs/promises";
import { extname, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

const root = fileURLToPath(new URL("../out/", import.meta.url));
const publicRoot = fileURLToPath(new URL("../public/", import.meta.url));
const { values } = parseArgs({ options: {
  port: { type: "string", default: process.env.PORT ?? "3000" },
  hostname: { type: "string", default: "127.0.0.1" },
} });
const port = Number(values.port);
if (!Number.isInteger(port) || port < 1 || port > 65535) throw new Error(`Invalid port ${values.port}`);
await stat(resolve(root, "index.html"));
const contentTypes = { ".html": "text/html; charset=utf-8", ".js": "text/javascript", ".mjs": "text/javascript", ".css": "text/css", ".json": "application/json", ".wasm": "application/wasm", ".svg": "image/svg+xml", ".png": "image/png", ".ico": "image/x-icon", ".woff2": "font/woff2", ".gz": "application/gzip" };
const server = createServer(async (request, response) => {
  try {
    if (!["GET", "HEAD"].includes(request.method)) { response.writeHead(405); response.end(); return; }
    const path = decodeURIComponent(new URL(request.url, "http://localhost").pathname);
    const assetRoot = path.startsWith("/policy/") ? publicRoot : root;
    const file = resolve(assetRoot, `.${path.endsWith("/") ? `${path}index.html` : path}`);
    if (!file.startsWith(assetRoot.endsWith(sep) ? assetRoot : assetRoot + sep)) { response.writeHead(403); response.end(); return; }
    const data = await readFile(file);
    response.writeHead(200, { "Content-Type": contentTypes[extname(file)] ?? "application/octet-stream", "Content-Length": data.length, ...(path === "/policy/index.json" ? { "Cache-Control": "no-store" } : {}) });
    response.end(request.method === "HEAD" ? undefined : data);
  } catch (error) {
    const status = error.code === "ENOENT" || error.code === "EISDIR" ? 404 : error instanceof URIError ? 400 : 500;
    if (status === 500) console.error(error);
    response.writeHead(status); response.end(status === 404 ? "Not found" : "Request failed");
  }
});
server.listen(port, values.hostname, () => console.log(`Static app: http://${values.hostname}:${port}`));
