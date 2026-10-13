import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import { before, test } from "node:test";

const read = (file) => JSON.parse(fs.readFileSync(new URL(`../public/${file}`, import.meta.url), "utf8"));

before(() => {
  execFileSync(process.execPath, ["scripts/build-recipes-api.mjs"], { stdio: "pipe" });
});

test("canonical API serves an eligible precision variant on declared restricted hardware", () => {
  const model = read("PaddlePaddle/PaddleOCR-VL.json");
  const endpoint = model.recommended_command.by_hardware.ascend_310p3;
  assert.equal(endpoint, "/PaddlePaddle/PaddleOCR-VL/hw/ascend_310p3.json");
  const command = read(endpoint.slice(1));
  const dtypeIndex = command.argv.indexOf("--dtype");
  assert.ok(dtypeIndex !== -1, "--dtype argument should be present");
  assert.equal(command.argv[dtypeIndex + 1], "float16");
  const tpIndex = command.argv.indexOf("--tensor-parallel-size");
  assert.ok(tpIndex !== -1, "--tensor-parallel-size argument should be present");
  assert.equal(command.argv[tpIndex + 1], "1");
  assert.ok(command.argv.includes("PaddlePaddle/PaddleOCR-VL"));
  assert.ok(command.argv.includes("PaddleOCR-VL-0.9B"));
  assert.ok(!command.argv.includes("--enforce-eager"));
  assert.match(command.docker_command, /nightly-main-310p@sha256:/);
  assert.ok(!model.recommended_command.argv.includes("--dtype"));
  // Adding a restricted NPU must not expose it on unrelated GPU recipes.
  const unrelated = read("Qwen/Qwen3-8B.json");
  assert.ok(!unrelated.recommended_command.by_hardware.ascend_310p3);
});

test("promoted default checkpoint excludes sibling-only hardware while the parent retains it", () => {
  const parentId = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16";
  const promotedId = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4";
  const parent = read(`${parentId}.json`);
  const promoted = read(`${promotedId}.json`);
  assert.equal(promoted.meta.derived_from, parentId);
  assert.equal(promoted.meta.variant, "default");
  assert.equal(promoted.model.model_id, promotedId);
  assert.equal(promoted.recommended_command.argv[2], promotedId);
  const parentAmd = parent.recommended_command.by_hardware.mi300x;
  assert.ok(parentAmd, "the parent should still offer its BF16 sibling on AMD");
  assert.equal(read(parentAmd.slice(1)).argv[2], parentId);
  assert.ok(!promoted.recommended_command.by_hardware.mi300x);
  assert.ok(!fs.existsSync(new URL(`../public/${promotedId}/hw/mi300x.json`, import.meta.url)),
    "removed promoted hardware must not remain as a stale JSON endpoint");
});

test("every promoted checkpoint keeps its selected variant across hardware and strategies", () => {
  let checked = 0;
  for (const entry of read("models.json")) {
    const recipe = read(entry.json.slice(1));
    if (!recipe.meta.derived_from) continue;
    const expected = recipe.meta.variant;
    assert.equal(recipe.recommended_command.variant, expected, entry.hf_id);
    for (const endpoint of Object.values(recipe.recommended_command.by_hardware)) {
      const command = read(endpoint.slice(1));
      assert.equal(command.variant, expected, endpoint);
      for (const alternative of Object.values(command.alternatives || {})) {
        assert.equal(read(alternative.slice(1)).variant, expected, alternative);
      }
      checked++;
    }
  }
  assert.ok(checked > 0, "the catalog must exercise promoted hardware endpoints");
});
