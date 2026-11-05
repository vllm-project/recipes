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
  computeDockerMeta,
  hardwareInstall,
  hwPipOverride,
  isFeatureSupportedOnHardware,
  isModeSupported,
  isPipInstallAvailable,
  isPrecisionCompatible,
  multiNodeFloor,
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

test("a hardware's own pip command outranks the recipe-wide and platform defaults", () => {
  const ironwood = profiles.ironwood;
  const own = { pip: { command: "pip install plugin", note: "n" }, docker: false };
  const variant = { hardware_overrides: { ironwood: { install: own } } };
  const docker_only = recipe({ model: { install: { pip: false } } });

  // TPU has no pip tab, and a Docker-only recipe none anywhere; the command brings it back.
  assert.equal(isPipInstallAvailable(recipe(), {}, ironwood, "ironwood"), false);
  assert.equal(isPipInstallAvailable(docker_only, {}, profiles.gb300, "gb300"), false);
  assert.equal(isPipInstallAvailable(docker_only, variant, ironwood, "ironwood"), true);
  assert.deepEqual(hwPipOverride(hardwareInstall(docker_only, variant, ironwood, "ironwood")), own.pip);
  // Only on that hardware, and `pip: false` is not a command.
  assert.equal(isPipInstallAvailable(docker_only, variant, profiles.gb300, "gb300"), false);
  assert.equal(hwPipOverride({ pip: false }), null);
  // docker: false renders the bare command.
  assert.equal(computeDockerMeta(recipe(), variant, ironwood, "ironwood").dockerEnabled, false);
});

test("the schema takes a per-hardware pip command, and it must name the command", () => {
  const { properties } = buildRecipeSchema({ taxonomy, strategies });
  const validate = new Ajv({ strict: false }).compile(properties.variants);
  const variants = (install) => ({ default: { precision: "fp8", vram_minimum_gb: 1, hardware_overrides: { ironwood: { install } } } });
  assert.equal(validate(variants({ pip: { command: "x", note: "y" }, docker: false })), true);
  assert.equal(validate(variants({ pip: false })), true);
  assert.equal(validate(variants({ pip: { note: "y" } })), false);
  assert.equal(validate(variants({ pip: { command: "x", nope: "y" } })), false);
});

test("a hardware gate reads the most specific key, in every key the schema accepts", () => {
  const off = (hardware, id) => !isFeatureSupportedOnHardware({ hardware }, profiles[id], id);
  assert.equal(off({ ironwood: "unsupported" }, "ironwood"), true);
  assert.equal(off({ tpu: "unsupported" }, "ironwood"), true);
  assert.equal(off({ google: "unsupported" }, "ironwood"), true);
  assert.equal(off({ default: "unsupported" }, "ironwood"), true);
  assert.equal(off({ [`arch:${profiles.b300.compute_arch}`]: "unsupported" }, "b300"), true);
  assert.equal(off({ tpu: "unsupported" }, "b300"), false);
  // A more specific key re-enables what a broader one turned off.
  assert.equal(off({ blackwell: "unsupported", b300: "supported" }, "b300"), false);
  assert.equal(off({ blackwell: "unsupported", b300: "supported" }, "b200"), true);
  assert.equal(isModeSupported({}, profiles.b300, "b300"), true);
});

test("a feature gated off a hardware leaves no companion", () => {
  const features = { helper: { description: "d", args: ["--h"], hardware: { gb300: "unsupported" }, companion: { label: "H", command: "helper" } } };
  const companions = (hw) => resolveCommand(recipe({ features }), "default", "single_node_tp", hw, ["helper"], strategies, taxonomy, [], 1).companions || [];
  assert.equal(companions("gb300").length, 0);
  assert.equal(companions("b300").length, 1);
});

test("multi-node selections open on the floor of the strategy they will run", () => {
  const floors = recipe({
    compatible_strategies: ["multi_node_tp", "multi_node_tep", "pd_cluster"],
    strategy_min_gpus: { multi_node_tp: 16, multi_node_tep: 8 },
  });
  const floor = (requested) => multiNodeFloor(floors, strategies, profiles.gb200, "gb200", requested);
  // GB200 is a 4-GPU tray.
  assert.equal(floor("multi_node_tep"), 2);
  assert.equal(floor("multi_node_tp"), 4);
  // PD pools size themselves.
  assert.equal(floor("pd_cluster"), 2);
  // A strategy this selection can't run falls back to the recommendation's floor.
  assert.equal(floor("single_node_tp"), floor(null));
  assert.equal(floor("nonsense"), floor(null));
});
