import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { test } from "node:test";
import yaml from "js-yaml";
import { resolveCommand } from "../src/lib/command-synthesis.js";
import { contentDir } from "../src/lib/content-root.js";

const read = (file) => yaml.load(fs.readFileSync(path.join(contentDir(), file), "utf8"));
const taxonomy = read("taxonomy.yaml");
const strategies = Object.fromEntries(fs.readdirSync(new URL("../strategies", import.meta.url))
  .filter((file) => file.endsWith(".yaml"))
  .map((file) => { const strategy = read(`strategies/${file}`); return [strategy.name, strategy]; }));

const recipe = {
  hf_id: "test/dcp",
  model: { model_id: "test/dcp", base_args: [] },
  variants: { default: { precision: "fp8", vram_minimum_gb: 100 } },
  compatible_strategies: ["multi_node_tp", "multi_node_dep", "pd_cluster"],
  decode_context_parallel: { sizes: { gb200: [8, 16] }, comm_backend: "a2a", match_tp: true },
  strategy_overrides: {
    pd_cluster: {
      prefill: { parallelism: "tp", nodes: { gb200: 4 } },
      decode: { parallelism: "tp", nodes: { gb200: 4 } },
      hardware_overrides: { gb200: { extra_args: ["--decode-context-parallel-size", "16", "--dcp-comm-backend", "a2a"] } },
    },
  },
};
const flag = (argv, key) => (argv.includes(key) ? argv[argv.lastIndexOf(key) + 1] : undefined);
const run = (hw, strategy, nodes, dcp, pools = null) =>
  resolveCommand(recipe, "default", strategy, hw, [], strategies, taxonomy, [], nodes, pools, {}, null, null, undefined, null, "http", dcp);

test("aggregated TP offers only the size equal to TP and defaults to Off", () => {
  const r = run("gb200", "multi_node_tp", 4, null);
  assert.deepEqual(r.dcp, { sizes: [16], baseline: 0, selected: 0, tp: [16] });
  assert.equal(flag(r.headArgv, "--decode-context-parallel-size"), undefined);
});

test("a pick lands on the head and every worker, with the comm backend", () => {
  const r = run("gb200", "multi_node_tp", 4, 16);
  for (const argv of [r.headArgv, ...r.workerArgvs]) {
    assert.equal(flag(argv, "--decode-context-parallel-size"), "16");
    assert.equal(flag(argv, "--dcp-comm-backend"), "a2a");
  }
});

test("a size that does not match TP falls back to the default", () => {
  assert.equal(run("gb200", "multi_node_tp", 4, 8).dcp.selected, 0);
});

test("PD keeps both roles on the same size and Off drops the flags", () => {
  const pools = { prefill: { nodes: 4 }, decode: { nodes: 4 } };
  const on = run("gb200", "pd_cluster", 1, null, pools);
  assert.equal(on.dcp.selected, 16);
  for (const role of ["prefill", "decode"]) assert.equal(flag(on[role].argv, "--decode-context-parallel-size"), "16");
  const off = run("gb200", "pd_cluster", 1, 1, pools);
  for (const role of ["prefill", "decode"]) assert.equal(flag(off[role].argv, "--decode-context-parallel-size"), undefined);
});

test("a DEP decode pool (TP1) cannot carry DCP, so the carried flag is dropped on both roles", () => {
  const r = run("gb200", "pd_cluster", 1, null, { prefill: { nodes: 4 }, decode: { nodes: 4, parallelism: "dep" } });
  assert.deepEqual(r.dcp.sizes, []);
  for (const role of ["prefill", "decode"]) assert.equal(flag(r[role].argv, "--decode-context-parallel-size"), undefined);
});

test("hardware without a sizes entry is untouched", () => {
  const r = run("b200", "multi_node_tp", 2, 8);
  assert.equal(r.dcp, null);
  assert.equal(flag(r.headArgv, "--decode-context-parallel-size"), undefined);
});

test("native TP2 × PCP4 offers DCP4 and DCP8 on an eight-device node", () => {
  const pcpRecipe = {
    ...recipe,
    strategy_overrides: { single_node_tp: { tp: 2, pcp: 4 } },
    decode_context_parallel: { sizes: [2, 4, 8], comm_backend: "ag_rs" },
  };
  const render = (size) => resolveCommand(pcpRecipe, "default", "single_node_tp", "h100",
    [], strategies, taxonomy, [], 1, null, {}, null, null, undefined, null, "http", size);
  assert.deepEqual(render(2).dcp.sizes, [4, 8]);
  assert.equal(flag(render(2).argv, "--decode-context-parallel-size"), undefined);
  for (const size of [4, 8]) {
    const result = render(size);
    assert.equal(flag(result.argv, "--decode-context-parallel-size"), String(size));
    assert.equal(flag(result.argv, "--dcp-comm-backend"), "ag_rs");
  }
  pcpRecipe.decode_context_parallel.match_tp = true;
  assert.deepEqual(render(4).dcp.sizes, []);
  pcpRecipe.decode_context_parallel.match_tp = false;
  pcpRecipe.decode_context_parallel.comm_backend = "a2a";
  assert.deepEqual(render(4).dcp.sizes, []);
});

test("NIXL PCP preserves distinct role DCP defaults and accepts a shared override", () => {
  const prefill = { nodes: 1, parallelism: "tp", tp: 1, pcp: 4,
    hardware_overrides: { gb200: { extra_args: ["--decode-context-parallel-size", "4"] } } };
  const pcpRecipe = {
    ...recipe,
    strategy_overrides: { pd_cluster: { prefill, decode: { nodes: 1, parallelism: "tp", tp: 4 } } },
    decode_context_parallel: { sizes: [2, 4], comm_backend: "ag_rs" },
  };
  const render = (size) => resolveCommand(pcpRecipe, "default", "pd_cluster", "gb200",
    [], strategies, taxonomy, [], 2, null, {}, null, null, undefined, null, "http", size);
  const dcpArg = (r, role) => flag(r[role].argv, "--decode-context-parallel-size");
  for (const size of [null, 2]) {
    const result = render(size);
    assert.deepEqual(result.dcp, { sizes: [4], baseline: null, selected: null,
      tp: [1, 4], pcp: [4, 1], roleDefaults: { prefill: 4, decode: 0 } });
    assert.equal(dcpArg(result, "prefill"), "4");
    assert.equal(dcpArg(result, "decode"), undefined);
  }
  for (const size of [1, 4]) {
    const result = render(size);
    for (const role of ["prefill", "decode"]) {
      assert.equal(dcpArg(result, role), size === 1 ? undefined : "4");
    }
  }
  // NIXL accepts PCP+DCP on the producer only when TP=1 and DCP=PCP.
  Object.assign(prefill, { tp: 2, pcp: 2, hardware_overrides: {} });
  assert.deepEqual(render(2).dcp.sizes, []);
});
