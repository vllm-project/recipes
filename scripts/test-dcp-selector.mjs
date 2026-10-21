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
