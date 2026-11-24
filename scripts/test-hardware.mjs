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
  buildDockerArgv,
  buildDockerRun,
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
  variantRunsOnHardware,
} from "../src/lib/command-synthesis.js";
import { resolveSelection } from "../src/lib/recipe-render.js";

const taxonomy = yaml.load(fs.readFileSync("taxonomy.yaml", "utf8"));
const profiles = taxonomy.hardware_profiles;
const vr200 = profiles.vr200;
const strategies = Object.fromEntries(
  fs.readdirSync("strategies").map((f) => [path.basename(f, ".yaml"), yaml.load(fs.readFileSync(`strategies/${f}`, "utf8"))]),
);
const IMAGE = "vllm/vllm-openai:cu134-nightly";

test("single RTX 4090 is opt-in and selects an offloaded variant that fits", () => {
  const hw = profiles.rtx_4090;
  assert.equal(hw.gpu_count, 1);
  assert.equal(hw.vram_gb, 24);
  assert.equal(hw.compute_arch, "sm89");
  assert.equal(hw.restricted, true);
  assert.equal(hw.scalable, false);

  const r = {
    meta: { tasks: ["omni"], hardware: {} },
    model: { model_id: "org/image", base_args: [] },
    omni: { tasks: ["t2i"] },
    variants: {
      default: { precision: "bf16", vram_minimum_gb: 153 },
      dlo: { precision: "bf16", vram_minimum_gb: 16 },
    },
  };
  const query = new URLSearchParams("hardware=rtx_4090");
  const undeclared = resolveSelection(r, query, strategies, taxonomy);
  assert.notEqual(undeclared.selection.hardware, "rtx_4090");
  assert.ok(!undeclared.options.hardware.includes("rtx_4090"));

  r.meta.hardware.rtx_4090 = "supported";
  const selected = resolveSelection(r, query, strategies, taxonomy);
  assert.equal(selected.selection.hardware, "rtx_4090");
  assert.equal(selected.selection.variant, "dlo");
  assert.ok(selected.options.hardware.includes("rtx_4090"));
  assert.equal(variantRunsOnHardware(hw, r.variants.default, "rtx_4090"), false);
  assert.equal(variantRunsOnHardware(hw, r.variants.dlo, "rtx_4090"), true);
});

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

test("the schema accepts a per-hardware Docker shm size and rejects shell text", () => {
  const { properties } = buildRecipeSchema({ taxonomy, strategies });
  const validate = new Ajv({ strict: false }).compile(properties.variants);
  const variants = (docker_shm_size) => ({
    default: { precision: "fp8", vram_minimum_gb: 1, hardware_overrides: { ascend_910b: { docker_shm_size } } },
  });
  assert.equal(validate(variants("320g")), true);
  assert.equal(validate(variants("0.5g")), true);
  assert.equal(validate(variants("320g --ipc=host")), false);
});

test("an Ascend variant can opt into host IPC for shared Engram tables", () => {
  const hardware_overrides = { ascend_910c: { docker_ipc: "host", docker_shm_size: "320g" } };
  const variant = { precision: "int4", vram_minimum_gb: 64, hardware_overrides };
  const validate = new Ajv({ strict: false }).compile(buildRecipeSchema({ taxonomy, strategies }).properties.variants);
  assert.ok(validate({ default: variant }));
  const argv = ["vllm", "serve", "org/model"];
  for (const hwId of ["ascend_910c", "ascend_910b"]) {
    const meta = computeDockerMeta(recipe(), variant, profiles[hwId], hwId);
    const shell = buildDockerRun({ command: argv.join(" "), ...meta });
    const docker = buildDockerArgv({ argv, meta });
    const hostIpc = hwId === "ascend_910c";
    assert.equal(shell.includes("--ipc=host"), hostIpc);
    assert.equal(docker.includes("--ipc=host"), hostIpc);
    assert.equal(shell.includes("--shm-size=16g"), !hostIpc);
    assert.equal(docker.includes("--shm-size"), !hostIpc);
  }
  for (const key of ["gb200", "npu", "huawei", "default"]) {
    const scoped = { ...variant, hardware_overrides: { [key]: { docker_ipc: "host" } } };
    assert.equal(validate({ default: scoped }), false, key);
  }
  hardware_overrides.ascend_910c.docker_ipc = "container:other";
  assert.equal(validate({ default: variant }), false);
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

const dpStrategies = ["multi_node_tp_dp", "multi_node_dep"];
const dpRecipe = (strategy) => ({
  model: { model_id: "org/model", architecture: "moe", base_args: [] },
  variants: { default: { precision: "bf16", vram_minimum_gb: 1 } },
  compatible_strategies: [strategy],
  strategy_overrides: { [strategy]: { extra_args: [] } },
});
const dpCommands = (r, strategy, hw = "h200", nodes = 4, features = []) => {
  const out = resolveCommand(r, "default", strategy, hw, features, strategies, taxonomy, [], nodes);
  assert.equal(out.workerArgvs.length, nodes - 1);
  return [out.headArgv, ...out.workerArgvs];
};
const dpValue = (argv, flag) => {
  const aliases = { "-dpl": "--data-parallel-size-local", "-dp": "--data-parallel-size" };
  const i = argv.findLastIndex((t) => (aliases[t.split("=")[0]] || t.split("=")[0].replaceAll("_", "-")) === flag);
  return i < 0 ? undefined : argv[i].split("=")[1] ?? argv[i + 1];
};

test("multi-node DP sizes and all follower ranks follow strategy, hardware and preset layouts", () => {
  for (const strategy of dpStrategies) for (const layer of ["strategy", "hardware", "preset"]) {
    const r = dpRecipe(strategy);
    const so = r.strategy_overrides[strategy];
    const layout = ["--tensor-parallel-size", "4", "--data-parallel-size-local", "2"];
    if (layer === "strategy") so.extra_args = layout;
    if (layer === "hardware") so.hardware_overrides = { h200: { extra_args: layout } };
    if (layer === "preset") r.presets = { local: { hardware: "h200", strategy, default: true, args: layout } };
    dpCommands(r, strategy).forEach((argv, node) => {
      assert.deepEqual(["--tensor-parallel-size", "--data-parallel-size", "--data-parallel-size-local", "--data-parallel-start-rank"].map((f) => dpValue(argv, f)),
        ["4", "8", "2", node ? String(node * 2) : undefined], `${strategy}/${layer}/node${node}`);
    });
  }
});

test("multi-node TP+DP overrides a variant's TP and local DP together", () => {
  const strategy = "multi_node_tp_dp";
  const r = dpRecipe(strategy);
  r.variants.default.extra_args = ["--tensor-parallel-size", "4", "--data-parallel-size-local", "2"];
  dpCommands(r, strategy, "h200", 4).forEach((argv, node) => {
    assert.deepEqual(["--tensor-parallel-size", "--data-parallel-size-local", "--data-parallel-size", "--data-parallel-start-rank"].map((flag) => dpValue(argv, flag)),
      ["8", "1", "4", node ? String(node) : undefined], `node${node}`);
  });
});

test("multi-node DP defaults retain the complete command order", () => {
  for (const strategy of dpStrategies) {
    const r = dpRecipe(strategy);
    r.model.base_args = ["--max-model-len", "4096"];
    r.features = { tools: { args: ["--tool-call-parser", "hermes"] } };
    const local = strategy === "multi_node_dep" ? 8 : 1;
    dpCommands(r, strategy, "h200", 4, ["tools"]).forEach((argv, node) => {
      assert.deepEqual(argv, ["vllm", "serve", "org/model", "--max-model-len", "4096",
        ...(strategy === "multi_node_dep" ? ["--enable-expert-parallel", "--data-parallel-hybrid-lb"] : ["--tensor-parallel-size", "8"]),
        "--data-parallel-size", String(4 * local), "--data-parallel-size-local", String(local),
        "--data-parallel-address", "$HEAD_IP", ...(node ? ["--data-parallel-start-rank", String(node * local)] : []),
        "--tool-call-parser", "hermes"]);
    });
  }
});

test("multi-node DEP overrides base local DP and variant global DP but respects later overrides", () => {
  const strategy = "multi_node_dep";
  const r = dpRecipe(strategy);
  r.model.base_args = ["--data-parallel-size-local", "1"];
  // Qwen3.5 Ascend variants carry DP1 for their single-node path.
  r.variants.default.extra_args = ["--data-parallel-size", "1"];
  for (const global of [16, 32]) {
    if (global === 32) r.strategy_overrides[strategy].extra_args = ["--data-parallel-size", "32", "--data-parallel-start-rank", "16"];
    dpCommands(r, strategy, "ascend_950dt", 2).forEach((argv, node) => {
      assert.deepEqual(["--data-parallel-size", "--data-parallel-size-local", "--data-parallel-start-rank"].map((f) => dpValue(argv, f)),
        [String(global), "8", global === 32 ? "16" : node ? "8" : undefined]);
    });
  }
});

test("multi-node DP does not restore flags removed by a preset", () => {
  for (const strategy of dpStrategies) for (const flag of ["--data-parallel-size-local", "--data-parallel-size", "--data-parallel-start-rank"]) {
    const r = dpRecipe(strategy);
    r.presets = { remove: { hardware: "h200", strategy, default: true, remove_args: [flag] } };
    for (const argv of dpCommands(r, strategy)) assert.equal(dpValue(argv, flag), undefined, `${strategy}/${flag}`);
  }
});

test("multi-node DP reads local aliases and never derives NaN from an invalid local value", () => {
  for (const strategy of dpStrategies) for (const local of [["-dpl", "2"], ["--data_parallel_size_local", "2"], ["--data-parallel-size-local=2"], ["--data-parallel-size-local", "invalid"]]) {
    const r = dpRecipe(strategy);
    r.strategy_overrides[strategy].extra_args = ["--tensor-parallel-size", "4", ...local];
    const expected = local.includes("invalid") ? (strategy === "multi_node_dep" ? 8 : 1) : 2;
    dpCommands(r, strategy).forEach((argv, node) => {
      assert.equal(dpValue(argv, "--data-parallel-size"), String(4 * expected));
      assert.equal(dpValue(argv, "--data-parallel-start-rank"), node ? String(node * expected) : undefined);
      assert.equal(dpValue(argv, "--data-parallel-size-local"), local.includes("invalid") ? "invalid" : "2");
    });
  }
});

test("NPU Docker device count is scoped to exact profiles and shared by shell and argv", () => {
  const validate = new Ajv({ strict: false }).compile(buildRecipeSchema({ taxonomy, strategies }).properties.variants);
  const variant = { precision: "bf16", vram_minimum_gb: 3, hardware_overrides: { ascend_910b: { docker_npu_device_count: 1 } } };
  assert.equal(validate({ default: variant }), true);
  for (const count of [0, -1, 1.5, "1", 9]) {
    assert.equal(validate({ default: { ...variant, hardware_overrides: { ascend_910b: { docker_npu_device_count: count } } } }), false);
  }
  for (const key of ["gb200", "npu", "huawei", "default"]) {
    assert.equal(validate({ default: { ...variant, hardware_overrides: { [key]: { docker_npu_device_count: 1 } } } }), false);
  }
  const argv = ["vllm", "serve", "org/model"];
  for (const [selected, count] of [[variant, 1], [{ precision: "bf16", vram_minimum_gb: 3 }, 8]]) {
    const meta = computeDockerMeta(recipe(), selected, profiles.ascend_910b, "ascend_910b");
    const shell = buildDockerRun({ command: argv.join(" "), ...meta });
    const docker = buildDockerArgv({ argv, meta });
    for (let index = 0; index < 8; index++) {
      assert.equal(shell.includes('/dev/davinci' + index), index < count);
      assert.equal(docker.includes('/dev/davinci' + index), index < count);
    }
  }
  const gpuMeta = computeDockerMeta(recipe(), variant, profiles.gb200, "gb200");
  assert.equal(gpuMeta.npuDeviceCount, 0);
});
