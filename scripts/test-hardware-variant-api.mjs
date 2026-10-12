import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import { test } from "node:test";

const read = (file) => JSON.parse(fs.readFileSync(new URL(`../public/${file}`, import.meta.url), "utf8"));

test("canonical API serves an eligible precision variant on declared restricted hardware", () => {
  execFileSync(process.execPath, ["scripts/build-recipes-api.mjs"], { stdio: "pipe" });
  const model = read("PaddlePaddle/PaddleOCR-VL.json");
  const endpoint = model.recommended_command.by_hardware.ascend_310p3;
  assert.equal(endpoint, "/PaddlePaddle/PaddleOCR-VL/hw/ascend_310p3.json");
  const command = read(endpoint.slice(1));
  assert.equal(command.argv[command.argv.indexOf("--dtype") + 1], "float16");
  assert.equal(command.argv[command.argv.indexOf("--tensor-parallel-size") + 1], "1");
  assert.ok(command.argv.includes("PaddlePaddle/PaddleOCR-VL"));
  assert.ok(command.argv.includes("PaddleOCR-VL-0.9B"));
  assert.ok(!command.argv.includes("--enforce-eager"));
  assert.match(command.docker_command, /nightly-main-310p@sha256:/);
  assert.ok(!model.recommended_command.argv.includes("--dtype"));
  // Adding a restricted NPU must not expose it on unrelated GPU recipes.
  const unrelated = read("Qwen/Qwen3-8B.json");
  assert.ok(!unrelated.recommended_command.by_hardware.ascend_310p3);
});
