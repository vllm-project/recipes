// Hardware profile rules, shown on Rubin (`vr200`): its image and Docker-only
// install, precision compatibility, and which override blocks apply.
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { test } from "node:test";
import yaml from "js-yaml";
import Ajv from "ajv";
import { buildRecipeSchema } from "../src/lib/recipe-schema.js";
import {
  npuDeviceCountForCommand,
  computeDockerMeta,
  hardwareInstall,
  isPrecisionCompatible,
  pickDefaultHardware,
  resolveCommand,
} from "../src/lib/command-synthesis.js";

const taxonomy = yaml.load(fs.readFileSync("taxonomy.yaml", "utf8"));
const profiles = taxonomy.hardware_profiles;
const vr200 = profiles.vr200;
const strategies = Object.fromEntries(
  fs.readdirSync("strategies").map((f) => [path.basename(f, ".yaml"), yaml.load(fs.readFileSync(`strategies/${f}`, "utf8"))]),
);
const IMAGE = "vllm/vllm-openai:cu134-nightly";

const recipe = (extra = {}) => ({
  meta: { hardware: { vr200: "supported" } },
  model: { model_id: "org/model", base_args: [], base_env: {} },
  variants: { default: { precision: "fp8", vram_minimum_gb: 100 } },
  compatible_strategies: ["single_node_tp"],
  ...extra,
});
const args = (r, hw = "vr200") => {
  const out = resolveCommand(r, "default", "single_node_tp", hw, [], strategies, taxonomy, [], 1);
  return out.command || out.head?.command || JSON.stringify(out);
};

test("the profile's image replaces the default and a recipe's general pin", () => {
  assert.equal(computeDockerMeta(recipe(), {}, vr200, "vr200").image, IMAGE);
  const pinned = recipe({ model: { docker_image: "vllm/vllm-openai:v0.18.0" } });
  assert.equal(computeDockerMeta(pinned, {}, vr200, "vr200").image, IMAGE);
  const map = recipe({ model: { docker_image: { cu129: "a:cu129", cu130: "a:cu130" } } });
  const meta = computeDockerMeta(map, {}, vr200, "vr200");
  assert.equal(meta.image, IMAGE);
  assert.equal(meta.cudaMap, null);
});

test("an exact-GPU override (how a preset pins its image) still wins", () => {
  const variant = { hardware_overrides: { vr200: { docker_image: "vllm/vllm-openai:nightly-abc" } } };
  assert.equal(computeDockerMeta(recipe(), variant, vr200, "vr200").image, "vllm/vllm-openai:nightly-abc");
});

test("other hardware keeps the recipe's pin", () => {
  const pinned = recipe({ model: { docker_image: "vllm/vllm-openai:v0.18.0" } });
  assert.equal(computeDockerMeta(pinned, {}, profiles.gb300, "gb300").image, "vllm/vllm-openai:v0.18.0");
});

test("Rubin has no pip path, whatever the recipe says; other hardware does", () => {
  const r = recipe({ model: { install: { pip: { command: "uv pip install vllm" } } } });
  assert.equal(hardwareInstall(r, {}, vr200, "vr200").pip, false);
  assert.ok(hardwareInstall(r, {}, vr200, "vr200").docker.note);
  assert.equal(hardwareInstall(r, {}, profiles.gb300, "gb300"), null);
});

test("NVFP4 runs on Blackwell and Rubin, not Hopper", () => {
  const nvfp4 = { precision: "nvfp4" };
  assert.equal(isPrecisionCompatible(vr200, nvfp4), true);
  assert.equal(isPrecisionCompatible(profiles.b200, nvfp4), true);
  assert.equal(isPrecisionCompatible(profiles.h200, nvfp4), false);
});

test("a variant's single-generation precision_hardware still narrows", () => {
  const onlyHopper = { precision: "nvfp4", precision_hardware: { generation: "hopper" } };
  assert.equal(isPrecisionCompatible(vr200, onlyHopper), false);
  assert.equal(isPrecisionCompatible(profiles.h200, onlyHopper), true);
});

test("Rubin is never the landing hardware by accident", () => {
  assert.equal(pickDefaultHardware(profiles, { precision: "nvfp4" }, recipe()), "b200");
  assert.equal(pickDefaultHardware(profiles, { precision: "fp8" }, recipe()), "h200");
  const authored = recipe({ meta: { default_hardware: "vr200", hardware: { vr200: "verified" } } });
  assert.equal(pickDefaultHardware(profiles, { precision: "fp8" }, authored), "vr200");
  const undeclared = recipe({ meta: { default_hardware: "vr200" } });
  assert.equal(pickDefaultHardware(profiles, { precision: "fp8" }, undeclared), "h200");
});

test("`rubin` and `nvidia` overrides apply on Rubin, `blackwell` doesn't", () => {
  const blocks = {
    blackwell: { extra_args: ["--bw"] },
    nvidia: { extra_args: ["--nv"] },
  };
  const fallback = args(recipe({ hardware_overrides: blocks }));
  assert.match(fallback, /--nv/);
  assert.doesNotMatch(fallback, /--bw/);
  const own = args(recipe({ hardware_overrides: { ...blocks, rubin: { extra_args: ["--rb"] } } }));
  assert.match(own, /--rb/);
  assert.doesNotMatch(own, /--nv|--bw/);
  assert.match(args(recipe({ hardware_overrides: blocks }), "gb300"), /--bw/);
});

test("the schema accepts vr200 and rubin keys", () => {
  const { properties } = buildRecipeSchema({ taxonomy, strategies });
  const validate = (key, value) => new Ajv({ strict: false }).compile(properties[key])(value);
  assert.equal(validate("hardware_overrides", { rubin: { extra_args: ["--x"] } }), true);
  assert.equal(validate("hardware_overrides", { rubinn: { extra_args: ["--x"] } }), false);
  assert.equal(validate("meta", { title: "t", slug: "s", provider: "p", description: "d", date_added: "2026-11-03", date_updated: "2026-11-03", tasks: ["text"], hardware: { vr200: "supported" } }), true);
});

test("single-node NPU devices follow the final TP and DP values", () => {
  const profile = profiles.ascend_910c;
  const npuRecipe = (tp, dp) => recipe({
    meta: { hardware: { ascend_910c: "supported" } },
    variants: {
      default: {
        precision: "bf16",
        vram_minimum_gb: 1,
        tp,
        extra_args: ["--data-parallel-size", String(dp)],
      },
    },
  });
  const argvFor = (tp, dp, advanced = []) => resolveCommand(
    npuRecipe(tp, dp), "default", "single_node_tp", "ascend_910c", [], strategies, taxonomy, advanced, 1,
  ).argv;
  const count = (argv, strategy = "single_node_tp") => npuDeviceCountForCommand(argv, strategy, profile);

  assert.equal(count(argvFor(1, 1)), 1);
  assert.equal(count(argvFor(2, 1)), 2);
  assert.equal(count(argvFor(2, 8)), 16);
  assert.equal(count(argvFor(2, 1, ["--data-parallel-size", "2"])), 4);
  assert.equal(count(argvFor(2, 1), "multi_node_tp"), 16);
  assert.equal(count(undefined), 16);

  const r = npuRecipe(2, 1);
  const meta = computeDockerMeta(r, r.variants.default, profile, "ascend_910c", count(argvFor(2, 1)));
  assert.equal(meta.npuDeviceCount, 2);
  assert.match(meta.gpuFlags, /--device \/dev\/davinci0\b/);
  assert.match(meta.gpuFlags, /--device \/dev\/davinci1\b/);
  assert.doesNotMatch(meta.gpuFlags, /--device \/dev\/davinci2\b/);
});
