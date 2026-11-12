// Per-GPU feature defaults (`features.<f>.hardware_default_modes`) and the
// boolean-flag dedupe (`--no-X` / `--X` are one flag, last wins).
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { test } from "node:test";
import yaml from "js-yaml";
import { resolveCommand, resolveModeKey } from "../src/lib/command-synthesis.js";
import { defaultFeaturesFor } from "../src/lib/recipe-render.js";
import { contentDir } from "../src/lib/content-root.js";

const read = (file) => yaml.load(fs.readFileSync(path.join(contentDir(), file), "utf8"));
const taxonomy = read("taxonomy.yaml");
const strategies = Object.fromEntries(fs.readdirSync(new URL("../strategies", import.meta.url))
  .filter((file) => file.endsWith(".yaml"))
  .map((file) => { const strategy = read(`strategies/${file}`); return [strategy.name, strategy]; }));

const spec = {
  description: "spec",
  default_mode: "a",
  hardware_default_modes: { mi350p: "b" },
  modes: {
    a: { args: ["--speculative-config", '{"method":"a"}'] },
    b: { args: ["--speculative-config", '{"method":"b"}'], hardware: { default: "unsupported", mi350p: "supported" } },
  },
};
const recipe = {
  hf_id: "test/hw-defaults",
  model: { model_id: "test/hw-defaults", base_args: [] },
  variants: { default: { precision: "bf16", vram_minimum_gb: 100 } },
  features: { spec_decoding: spec },
  opt_in_features: ["spec_decoding"],
  compatible_strategies: ["single_node_tp"],
  hardware_overrides: { amd: { extra_args: ["--no-enable-prefix-caching"] } },
  strategy_overrides: { single_node_tp: { hardware_overrides: { mi350p: { extra_args: ["--enable-prefix-caching"] } } } },
};
const hw = (id) => taxonomy.hardware_profiles[id];
const argv = (id, features) => resolveCommand(recipe, "default", "single_node_tp", id, features, strategies, taxonomy).argv;

test("a hardware default mode turns an opt-in feature on for that GPU only", () => {
  assert.deepEqual(defaultFeaturesFor(recipe, "mi350p", "default"), ["spec_decoding"]);
  assert.deepEqual(defaultFeaturesFor(recipe, "mi355x", "default"), []);
});

test("the hardware default mode outranks the feature default, the user pick outranks both", () => {
  assert.equal(resolveModeKey(spec, "spec_decoding", {}, "default", hw("mi350p"), "mi350p"), "b");
  assert.equal(resolveModeKey(spec, "spec_decoding", {}, "default", hw("mi355x"), "mi355x"), "a");
  assert.equal(resolveModeKey(spec, "spec_decoding", {}, "default", hw("mi350p"), "mi350p", "a"), "a");
});

test("--X after --no-X keeps only --X, and the other GPUs keep --no-X", () => {
  const p = argv("mi350p", []);
  assert.ok(p.includes("--enable-prefix-caching"));
  assert.ok(!p.includes("--no-enable-prefix-caching"));
  const x = argv("mi355x", []);
  assert.ok(x.includes("--no-enable-prefix-caching"));
  assert.ok(!x.includes("--enable-prefix-caching"));
});
