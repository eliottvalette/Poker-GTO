import { copyFile, mkdir, writeFile } from "node:fs/promises";

const destination = new URL("../public/onnxruntime/", import.meta.url);
await mkdir(destination, { recursive: true });
for (const file of ["ort-wasm-simd-threaded.mjs", "ort-wasm-simd-threaded.wasm"]) {
  await copyFile(new URL(`../node_modules/onnxruntime-web/dist/${file}`, import.meta.url), new URL(file, destination));
}

// A clean checkout has no trained releases. Never replace a published catalog.
const policyRoot = new URL("../public/policy/", import.meta.url);
await mkdir(policyRoot, { recursive: true });
try {
  await writeFile(new URL("index.json", policyRoot), JSON.stringify({ version: 1, active: {}, exports: {} }, null, 2) + "\n", { flag: "wx" });
} catch (error) {
  if (error.code !== "EEXIST") throw error;
}
