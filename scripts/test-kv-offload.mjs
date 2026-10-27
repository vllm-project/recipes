import assert from "node:assert/strict";
import fs from "node:fs";
import { test } from "node:test";
import yaml from "js-yaml";
import { resolveCommand, resolveKvOffloadOption, isKvOffloadBrandSupported, kvOffloadInstallSteps } from "../src/lib/command-synthesis.js";

// The vocabulary (taxonomy + strategies) always comes from this repo's root.
const root = new URL("../", import.meta.url);
const read = (file) => yaml.load(fs.readFileSync(new URL(file, root), "utf8"));
const taxonomy = read("taxonomy.yaml");
const strategies = Object.fromEntries(fs.readdirSync(new URL("strategies/", root))
  .filter((file) => file.endsWith(".yaml"))
  .map((file) => { const strategy = read(`strategies/${file}`); return [strategy.name, strategy]; }));
const hw = (id) => taxonomy.hardware_profiles[id];

const recipe = (extra = {}) => ({
  hf_id: "test/kv-offload",
  model: { model_id: "test/kv-offload", base_args: [] },
  // Bigger than one H100, so single_node_tp shards across the whole node.
  variants: { default: { precision: "bf16", vram_minimum_gb: 200 } },
  compatible_strategies: ["single_node_tp", "multi_node_tp"],
  ...extra,
});
const run = (r, hwId, kvOffload, tax = taxonomy) =>
  resolveCommand(r, "default", "single_node_tp", hwId, [], strategies, tax, [], 1, null, {}, kvOffload);
const flag = (argv, key) => (argv.includes(key) ? argv[argv.lastIndexOf(key) + 1] : undefined);

test("hardware_overrides replace top-level keys and merge env", () => {
  const opt = { description: "base", install: "base", env: { A: "1", B: "1" }, hardware_overrides: { cpu: { install: "cpu", env: { B: "2" } } } };
  const cpu = resolveKvOffloadOption(opt, hw("xeon6"), "xeon6");
  assert.equal(cpu.install, "cpu");
  assert.equal(cpu.description, "base");
  assert.deepEqual(cpu.env, { A: "1", B: "2" });
  assert.equal(resolveKvOffloadOption(opt, hw("h100"), "h100"), opt);
});

test("IAXL hardware gate: Intel / NVIDIA brands minus Grace-based nodes", () => {
  const iaxl = taxonomy.kv_offload.iaxl;
  for (const id of ["xeon6", "h100", "b200", "rtx_pro_6000"]) assert.ok(isKvOffloadBrandSupported(iaxl, hw(id), id), id);
  for (const id of ["gb200", "gb300", "dgx_station_gb300", "dgx_spark_gb10", "mi300x"]) assert.ok(!isKvOffloadBrandSupported(iaxl, hw(id), id), id);
});

test("IAXL on Xeon CPU renders the CPU build, its env and the setvars.sh setup", () => {
  const r = run(recipe(), "xeon6", "iaxl");
  assert.equal(r.env.DEVICE, "cpu");
  assert.equal(r.env.TP_SIZE, flag(r.argv, "--tensor-parallel-size"));
  assert.equal(r.env.IAXL_DSA_GD_ENABLE, undefined);
  assert.match(r.setup, /source setvars\.sh/);
  assert.match(r.setup, /export LD_PRELOAD="\$IAXL_LD_PRELOAD/);
  assert.match(r.argv.join(" "), /KVShrinkConnector/);
  assert.ok(r.argv.includes("--no-enable-prefix-caching"));
});

test("IAXL on NVIDIA fills {tp_size} from the engine's TP and keeps DSA off by default", () => {
  const r = run(recipe(), "h100", "iaxl");
  assert.equal(flag(r.argv, "--tensor-parallel-size"), "8");
  assert.equal(r.env.TP_SIZE, "8");
  assert.equal(r.env.DEVICE, "cuda");
  assert.equal(r.env.IAXL_DSA_GD_ENABLE, "0");
});

test("IAXL is offered by default like LMCache; `unsupported` and Grace nodes leave no trace", () => {
  assert.equal(taxonomy.kv_offload.iaxl.requires_opt_in, undefined);
  assert.match(run(recipe(), "h100", "iaxl").argv.join(" "), /KVShrinkConnector/);
  for (const [r, id] of [[recipe({ kv_offload_support: { iaxl: "unsupported" } }), "h100"], [recipe(), "gb200"]]) {
    const out = run(r, id, "iaxl");
    assert.ok(!out.argv.join(" ").includes("KVShrinkConnector"), id);
    assert.equal(out.setup, undefined, id);
    assert.equal(out.env.TP_SIZE, undefined, id);
  }
});

test("a hardware override of a gating key gates synthesis like the builder", () => {
  const tax = {
    ...taxonomy,
    kv_offload: {
      ...taxonomy.kv_offload,
      gated: { args: ["--gated"], hardware_overrides: { h100: { strategies: ["multi_node_tp"] } } },
    },
  };
  assert.ok(!run(recipe(), "h100", "gated", tax).argv.includes("--gated"));
  assert.ok(run(recipe(), "h200", "gated", tax).argv.includes("--gated"));
});

test("install renders as dependency-style steps: a bare string or { note, command } entries", () => {
  assert.deepEqual(kvOffloadInstallSteps(taxonomy.kv_offload.lmcache), [{ command: taxonomy.kv_offload.lmcache.install }]);
  assert.deepEqual(kvOffloadInstallSteps({}), []);
  for (const id of ["xeon6", "h100"]) {
    const steps = kvOffloadInstallSteps(resolveKvOffloadOption(taxonomy.kv_offload.iaxl, hw(id), id));
    assert.equal(steps.length, 3, id);
    for (const s of steps) assert.ok(s.note && s.command && s.command === s.command.trimEnd(), id);
    assert.match(steps[0].command, /^sudo apt-get install -y /, id);
    assert.match(steps[2].command, new RegExp(`DEVICE=${id === "xeon6" ? "cpu" : "cuda"} `), id);
  }
});
