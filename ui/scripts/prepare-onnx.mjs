import { copyFile, mkdir } from "node:fs/promises";

const destination = new URL("../public/onnxruntime/", import.meta.url);
await mkdir(destination, { recursive: true });
for (const file of ["ort-wasm-simd-threaded.mjs", "ort-wasm-simd-threaded.wasm"]) {
  await copyFile(new URL(`../node_modules/onnxruntime-web/dist/${file}`, import.meta.url), new URL(file, destination));
}
