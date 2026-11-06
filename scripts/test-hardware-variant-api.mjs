import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { after, before, test } from "node:test";
import { fileURLToPath } from "node:url";
import yaml from "js-yaml";
import { resolveSelection } from "../src/lib/recipe-render.js";

const ROOT = fileURLToPath(new URL("..", import.meta.url));
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "recipes-hardware-variant-"));
const channel = path.basename(tmp);
const output = path.join(tmp, "public");
const read = (file) => JSON.parse(fs.readFileSync(path.join(output, file), "utf8"));

after(() => {
  fs.rmSync(tmp, { recursive: true, force: true });
  fs.rmSync(path.join(ROOT, ".content", channel), { recursive: true, force: true });
});

before(() => {
  // Exercise the public API without depending on the order in which the
  // owning repositories merge their model content.
  const fixture = path.join(tmp, "source");
  fs.cpSync(path.join(ROOT, "models"), path.join(fixture, "models"), { recursive: true });
  const paddleFile = path.join(fixture, "models/PaddlePaddle/PaddleOCR-VL.yaml");
  const paddle = yaml.load(fs.readFileSync(paddleFile, "utf8"));
  paddle.meta.hardware = { ...paddle.meta.hardware, ascend_310p3: "supported", ascend_910b3: "supported", ascend_910b4: "supported" };
  paddle.variants.default.supported_hardware = ["arch:sm90"];
  for (const [hw, key, precision, dtype] of [
    ["ascend_310p3", "ascend_fp16", "fp16", "float16"],
    ["ascend_910b3", "ascend_bf16", "bf16", "bfloat16"],
  ]) {
    paddle.variants[key] = {
      precision, vram_minimum_gb: 3, supported_hardware: [hw],
      hardware_overrides: { [hw]: {
        docker_image: hw === "ascend_310p3"
          ? "quay.io/ascend/vllm-ascend:v0.30.0rc2-310p@sha256:930dcb52a6d80460d823367356b012e95b18327470cff2bc2b94063b49b5f175"
          : "quay.io/ascend/vllm-ascend:v0.30.0rc2@sha256:6348403a0ebb19f52bc1b7497de1daeb6df0e4650f9051c169e47a011527a6c3",
        extra_args: ["--dtype", dtype, "--served-model-name", "PaddlePaddle/PaddleOCR-VL", "PaddleOCR-VL-0.9B"],
      } },
    };
  }
  paddle.variants.ascend_bf16.supported_hardware.push("ascend_910b4");
  paddle.variants.ascend_bf16.hardware_overrides.ascend_910b4 = {
    ...paddle.variants.ascend_bf16.hardware_overrides.ascend_910b3,
    docker_image: "quay.io/ascend/vllm-ascend:nightly-main@sha256:4b537a90f25f7d282927b49f474f9d8ecf61fd0bf556c28e9ec1ac54d48371c3",
  };
  fs.writeFileSync(paddleFile, yaml.dump(paddle));
  const sources = path.join(tmp, "sources.yaml");
  fs.writeFileSync(sources, JSON.stringify({
    default_channel: channel,
    channels: { [channel]: { recipes: { local: fixture, role: "core" } } },
  }));
  execFileSync(process.execPath, ["scripts/build-recipes-api.mjs"], {
    cwd: ROOT,
    stdio: "pipe",
    env: {
      ...process.env,
      RECIPES_SOURCES: sources,
      RECIPES_CHANNEL: channel,
      RECIPES_PUBLIC_DIR: output,
      RECIPES_CONTENT_DIR: "",
      RECIPES_OVERRIDES: "",
    },
  });
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
  assert.match(command.docker_command, /v0\.30\.0rc2-310p@sha256:/);
  assert.ok(!model.recommended_command.argv.includes("--dtype"));
  // Adding a restricted NPU must not expose it on unrelated GPU recipes.
  const unrelated = read("Qwen/Qwen3-8B.json");
  assert.ok(!unrelated.recommended_command.by_hardware.ascend_310p3);
  const a2 = read(model.recommended_command.by_hardware.ascend_910b3.slice(1));
  assert.equal(a2.variant, "ascend_bf16");
  assert.ok(a2.argv.includes("bfloat16"));
  assert.equal(a2.hardware_profile.gpu_count, 1);
  assert.ok(!unrelated.recommended_command.by_hardware.ascend_910b3);
});

test("the 32 GB A2 profile renders one NPU without leaking onto unrelated recipes", () => {
  const model = read("PaddlePaddle/PaddleOCR-VL.json");
  const command = read(model.recommended_command.by_hardware.ascend_910b4.slice(1));
  assert.equal(command.variant, "ascend_bf16");
  assert.equal(command.hardware_profile.vram_gb, 32);
  assert.equal(command.hardware_profile.gpu_count, 1);
  assert.equal(command.hardware_profile.restricted, true);
  assert.equal(command.hardware_profile.scalable, false);
  const dtype = command.argv.indexOf("--dtype");
  assert.ok(dtype !== -1);
  assert.equal(command.argv[dtype + 1], "bfloat16");
  const tp = command.argv.indexOf("--tensor-parallel-size");
  assert.ok(tp !== -1);
  assert.equal(command.argv[tp + 1], "1");
  assert.ok(command.argv.includes("PaddlePaddle/PaddleOCR-VL"));
  assert.ok(command.argv.includes("PaddleOCR-VL-0.9B"));
  assert.match(command.docker_command, /nightly-main@sha256:/);
  assert.ok(!read("Qwen/Qwen3-8B.json").recommended_command.by_hardware.ascend_910b4);
});

test("hardware-only page links select the runnable variant without overriding explicit checkpoints", () => {
  const recipe = read("PaddlePaddle/PaddleOCR-VL.json");
  const taxonomy = read("taxonomy.json");
  const strategies = Object.fromEntries(fs.readdirSync(path.join(ROOT, "strategies"))
    .filter((f) => f.endsWith(".yaml"))
    .map((f) => {
      const spec = yaml.load(fs.readFileSync(path.join(ROOT, "strategies", f), "utf8"));
      return [spec.name, spec];
    }));
  const selected = resolveSelection(recipe, new URLSearchParams({ hardware: "ascend_310p3" }), strategies, taxonomy);
  assert.equal(selected.selection.hardware, "ascend_310p3");
  assert.equal(selected.selection.variant, "ascend_fp16");
  assert.ok(selected.rendering.argv.includes("float16"));
  const a2 = resolveSelection(recipe, new URLSearchParams({ hardware: "ascend_910b4" }), strategies, taxonomy);
  assert.equal(a2.selection.hardware, "ascend_910b4");
  assert.equal(a2.selection.variant, "ascend_bf16");
  assert.ok(a2.rendering.argv.includes("bfloat16"));
  const explicit = resolveSelection(recipe, new URLSearchParams({ hardware: "ascend_310p3", variant: "default" }), strategies, taxonomy);
  assert.equal(explicit.selection.variant, "default");
  assert.notEqual(explicit.selection.hardware, "ascend_310p3");
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
  assert.ok(!fs.existsSync(path.join(output, promotedId, "hw/mi300x.json")),
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
