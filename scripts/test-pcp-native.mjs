import assert from "node:assert/strict";
import { test } from "node:test";
import Ajv from "ajv";
import { resolveCommand } from "../src/lib/command-synthesis.js";
import { applyOverlay, resolveScope } from "../src/lib/overlay-merge.js";
import { validatePresets } from "../src/lib/recipe-presets.js";
import { buildRecipeSchema } from "../src/lib/recipe-schema.js";
import { loadStrategies } from "../src/lib/strategies.js";
import { loadTaxonomy } from "../src/lib/taxonomy.js";

const taxonomy = loadTaxonomy();
const strategies = loadStrategies();
const recipe = {
  hf_id: "example/pcp",
  model: { model_id: "example/pcp" },
  compatible_strategies: ["single_node_tp", "multi_node_tp"],
  variants: { default: { precision: "bf16", vram_minimum_gb: 150 } },
  presets: {
    h100: { label: "PCP", hardware: "h100", strategy: "single_node_tp", pcp: 4,
      points: { one: { label: "Off", pcp: 1 }, two: { label: "Two ranks", tp: 2, pcp: 2 } } },
    // Native TP1 × PCP8 × DCP8 on two four-GPU nodes, as in the
    // source-owned Mistral-Large-4-1T-A52B gb300_pcp8_dcp8 preset.
    gb300: { label: "PCP", hardware: "gb300", strategy: "multi_node_tp", tp: 1, pcp: 8,
      args: ["--decode-context-parallel-size", "8", "--enable-expert-parallel"] },
  },
};
const run = (r, hw, strategy = "single_node_tp", nodes = 1, preset = "baseline", point = null) =>
  resolveCommand(r, "default", strategy, hw, [], strategies, taxonomy,
    [], nodes, null, {}, null, null, undefined, null, "http", null, preset, point);
const flag = (argv, name) => argv.includes(name) ? argv[argv.indexOf(name) + 1] : undefined;

test("native PCP uses the chosen hardware map and reserves devices before automatic TP", () => {
  const r = { ...recipe, strategy_overrides: { single_node_tp: { pcp: { h100: 4, "arch:gfx942": 2 } } } };
  for (const [hw, tp, pcp] of [["h100", 2, 4], ["mi300x", 1, 2]]) {
    const result = run(r, hw);
    assert.equal(result.pcp, pcp);
    assert.equal(result.tp, tp);
    assert.equal(flag(result.argv, "--prefill-context-parallel-size"), String(pcp));
    assert.equal(flag(result.argv, "--tensor-parallel-size"), String(tp));
    assert.deepEqual(result.argv.slice(0, 3), ["vllm", "serve", "example/pcp"]);
    assert.equal(result.env.ASCEND_RT_VISIBLE_DEVICES, undefined);
  }
  const unmatched = run(r, "h200");
  assert.equal(unmatched.pcp, undefined);
  assert.equal(flag(unmatched.argv, "--prefill-context-parallel-size"), undefined);
});

test("native preset points can change or disable PCP without changing the baseline", () => {
  const on = run(recipe, "h100", "single_node_tp", 1, "h100", "two");
  assert.deepEqual([on.tp, on.pcp], [2, 2]);
  const off = run(recipe, "h100", "single_node_tp", 1, "h100", "one");
  assert.deepEqual(off.argv, run(recipe, "h100").argv);
  assert.equal(off.pcp, undefined);
});

test("upstream CUDA and ROCm GSM8K layouts use ordinary vllm serve", () => {
  // vLLM v0.32.0 tests/evals/gsm8k/configs/{GLM-5.2-NVFP4-TP1-PCP4-EP,
  // DeepSeek-V2-Lite-Instruct-FP8-TP1-PCP2-ROCm}.yaml.
  for (const [hardware, model, pcp, moeBackend] of [
    ["b200", "nvidia/GLM-5.2-NVFP4", 4, "flashinfer_cutlass"],
    ["mi355x", "RedHatAI/DeepSeek-Coder-V2-Lite-Instruct-FP8", 2, "triton"],
  ]) {
    const r = { ...recipe, model: { model_id: model }, presets: {
      eval: { label: "GSM8K", hardware, strategy: "single_node_tp", tp: 1, pcp,
        args: ["--enable-expert-parallel", "--moe-backend", moeBackend], env: { VLLM_USE_V2_MODEL_RUNNER: "1" } },
    } };
    const result = run(r, hardware, "single_node_tp", 1, "eval");
    assert.deepEqual(result.argv, ["vllm", "serve", model, "--tensor-parallel-size", "1",
      "--prefill-context-parallel-size", String(pcp), "--enable-expert-parallel", "--moe-backend", moeBackend]);
    assert.equal(result.env.VLLM_USE_V2_MODEL_RUNNER, "1");
  }
});

test("native multi-node PCP spans nodes and keeps the head and each follower consistent", () => {
  const r = run(recipe, "gb300", "multi_node_tp", 2, "gb300");
  assert.throws(() => run(recipe, "gb300", "multi_node_tp", 1, "gb300"), /nodes|node count/i);
  assert.deepEqual([r.tp, r.pcp], [1, 8]);
  for (const [rank, argv] of [r.headArgv, ...r.workerArgvs].entries()) {
    assert.equal(flag(argv, "--tensor-parallel-size"), "1");
    assert.equal(flag(argv, "--prefill-context-parallel-size"), "8");
    assert.equal(flag(argv, "--decode-context-parallel-size"), "8");
    assert.equal(flag(argv, "--nnodes"), "2");
    assert.equal(flag(argv, "--node-rank"), String(rank));
    assert.equal(argv.includes("--headless"), rank > 0);
  }
  const raw = structuredClone(recipe);
  delete raw.presets.gb300.tp;
  delete raw.presets.gb300.pcp;
  raw.presets.gb300.args.unshift("--tensor-parallel-size", "1", "--prefill-context-parallel-size", "8");
  const legacy = run(raw, "gb300", "multi_node_tp", 2, "gb300");
  for (const axis of ["--tensor-parallel-size", "--prefill-context-parallel-size", "--decode-context-parallel-size", "--nnodes", "--node-rank"]) {
    assert.equal(flag(r.headArgv, axis), flag(legacy.headArgv, axis));
    assert.equal(flag(r.workerArgv, axis), flag(legacy.workerArgv, axis));
  }
});

test("reject oversubscription, uneven multi-node ranks, and conflicting late CLI spellings", () => {
  const runGb = (extra, nodes = 2) => run({ ...recipe, presets: {
    gb300: { ...recipe.presets.gb300, args: [], ...extra },
  } }, "gb300", "multi_node_tp", nodes, "gb300");
  assert.throws(() => runGb({ tp: 2, pcp: 8 }), /PCP layout/);
  assert.throws(() => runGb({ tp: 16, pcp: undefined }), /fit evenly/);
  assert.throws(() => runGb({ tp: 1, pcp: 3 }), /fit evenly/);
  assert.throws(() => runGb({ args: ["-tp=2"] }), /conflicts with the declared tp/);
  assert.throws(() => runGb({ args: ["--prefill_context_parallel_size=4"] }), /matching the declared pcp/);
  assert.throws(() => runGb({ args: ["-dcp=4"] }), /PCP launch requires DCP/);
  assert.throws(() => runGb({ args: ["-dcp=8", "--dcp_comm_backend=a2a"] }), /ag_rs/);
  assert.throws(() => runGb({ args: ["--nnodes", "4"] }), /PCP layout/);
  assert.throws(() => runGb({ args: ["-n=4"] }), /PCP layout/);
  assert.throws(() => runGb({ args: ["-r", "3"] }), /PCP layout/);
  const r = runGb({ args: ["-pcp=8", "--tensor_parallel_size", "1"] });
  assert.equal(r.headArgv.filter((arg) => arg === "--prefill-context-parallel-size").length, 1);
});

test("native PCP preset and overlay declarations preserve hardware and strategy scope", () => {
  const schema = buildRecipeSchema({ taxonomy, strategies });
  const validate = new Ajv({ strict: false }).compile(schema.properties.strategy_overrides);
  assert.ok(validate({ single_node_tp: { pcp: 4 }, multi_node_tp: { pcp: { gb300: 8 } } }));
  assert.equal(validate({ multi_node_tp: { pcp: 0 } }), false);
  assert.equal(validate({ single_node_dep: { pcp: 4 } }), false);
  assert.deepEqual(validatePresets(recipe, taxonomy, recipe.compatible_strategies), []);
  const misplaced = { ...recipe, presets: { p: { label: "PCP", hardware: "h100", strategy: "pd_cluster", pcp: 4 } } };
  assert.ok(validatePresets(misplaced, taxonomy, ["pd_cluster"]).some((e) => e.includes(".pcp:")));
  const ctx = resolveScope({ brands: ["Huawei"] }, taxonomy);
  const ov = { strategy_overrides: { single_node_tp: { pcp: { ascend_910b: 4 } } } };
  assert.deepEqual(applyOverlay(recipe, ov, { ctx, source: "vllm-ascend" }).recipe.strategy_overrides, ov.strategy_overrides);
  assert.ok(applyOverlay(recipe, { strategy_overrides: { single_node_tp: { pcp: { h100: 4 } } } }, { ctx, source: "vllm-ascend" }).errors.length);
});
