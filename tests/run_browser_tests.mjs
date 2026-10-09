/** Compile browser-only sources/tests in isolation and remove generated JS afterward. */
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const output = mkdtempSync(join(tmpdir(), "poker-browser-tests-"));
const tests = ["tests/browser_engine.test.ts", "tests/browser_table.test.ts", "tests/browser_neural.test.ts", "tests/browser_analysis.test.ts", "tests/browser_hybrid.test.ts", "tests/browser_beliefs.test.ts", "tests/browser_range_copy.test.ts", "tests/browser_range_equity.test.ts", "tests/browser_showdown_display.test.ts"];
try {
  const compilation = spawnSync(process.execPath, [
    join(root, "ui/node_modules/typescript/bin/tsc"), ...tests,
    "--target", "ES2022", "--module", "Node16", "--moduleResolution", "Node16",
    "--lib", "ES2023,DOM", "--strict", "--esModuleInterop", "--skipLibCheck",
    "--types", "node", "--typeRoots", join(root, "ui/node_modules/@types"),
    "--rootDir", root, "--outDir", output,
  ], { cwd: root, encoding: "utf8", maxBuffer: 8 * 1024 * 1024 });
  if (compilation.error) throw compilation.error;
  if (compilation.stdout) process.stdout.write(compilation.stdout);
  if (compilation.stderr) process.stderr.write(compilation.stderr);
  if (compilation.status !== 0) process.exitCode = compilation.status ?? 1;
  else {
    const result = spawnSync(process.execPath, ["--test", ...tests.map(path => join(output, path.replace(/\.ts$/, ".js")))],
      { cwd: root, stdio: "inherit" });
    if (result.error) throw result.error;
    process.exitCode = result.status ?? 1;
  }
} finally {
  rmSync(output, { recursive: true, force: true });
}
