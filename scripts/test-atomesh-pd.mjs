import assert from "node:assert/strict";
import fs from "node:fs";
import { test } from "node:test";
import yaml from "js-yaml";
import { resolveCommand } from "../src/lib/command-synthesis.js";
import { renderCommand } from "../src/lib/recipe-render.js";

const read = (file) => yaml.load(fs.readFileSync(new URL(`../${file}`, import.meta.url), "utf8"));
const recipe = read("models/XiaomiMiMo/MiMo-V2.6-Flash-RL.yaml");
const taxonomy = read("taxonomy.yaml");
const strategies = { pd_cluster: read("strategies/pd_cluster.yaml") };
const pools = { prefill: { nodes: 1 }, decode: { nodes: 1 } };
const flag = (argv, key) => argv[argv.lastIndexOf(key) + 1];
const resolve = ({ model = recipe, hw = "gb300", router = "atomesh", connector = null, layout = pools, kv = null, catalog = strategies } = {}) =>
  resolveCommand(model, "default", "pd_cluster", hw, [], catalog, taxonomy, [], 1, layout, {}, kv, null, undefined, router, "http", null, undefined, undefined, connector);

test("ATOMesh uses a pinned router container and native Mooncake workers", () => {
  const result = resolve();
  assert.equal(result.orchestrator, "atomesh");
  assert.equal(result.kvConnector.selected, "mooncake");
  assert.match(result.router.command, /--entrypoint atomesh .*@sha256:[a-f0-9]{64} launch/);
  assert.match(result.router.command, /--backend vllm --pd-disaggregation --policy dp_sticky/);
  assert.match(result.router.command, /--prefill http:\/\/\$PREFILL_NODE_1:8001 8998/);
  assert.match(result.router.command, /--decode http:\/\/\$DECODE_NODE_1:8002/);
  assert.doesNotMatch(result.router.command, /--gpus|--kv-connector|--intra-node-data-parallel-size/);
  for (const role of ["prefill", "decode"]) {
    assert.deepEqual(result[role].argv.slice(0, 2), ["vllm", "serve"]);
    assert.equal(JSON.parse(flag(result[role].argv, "--kv-transfer-config")).kv_connector, "MooncakeConnector");
  }
});

test("ATOMesh follows effective HTTP and bootstrap ports", () => {
  const model = structuredClone(recipe);
  model.strategy_overrides ||= {};
  model.strategy_overrides.pd_cluster = {
    prefill: { extra_args: ["--port", "9101"], env: { VLLM_MOONCAKE_BOOTSTRAP_PORT: "9301" } },
    decode: { extra_args: ["--port", "9102"] },
  };
  const result = resolve({ model });
  assert.match(result.router.command, /PREFILL_NODE_1:9101 9301/);
  assert.match(result.router.command, /DECODE_NODE_1:9102/);
});

test("unsupported transfer selection falls back to Mooncake with an explanation", () => {
  const result = resolve({ connector: "nixl" });
  assert.equal(result.pdRouter, "atomesh");
  assert.equal(result.kvConnector.selected, "mooncake");
  assert.match(result.kvConnector.unavailableReason, /only/);
});

test("ATOMesh rejects unequal TP and DP pools", () => {
  assert.throws(() => resolve({ layout: { prefill: { nodes: 1 }, decode: { nodes: 2 } } }), /PD ATOMesh.*equal TP/);
  assert.throws(() => resolve({ layout: { prefill: { nodes: 2, parallelism: "dep" }, decode: { nodes: 2, parallelism: "dep" } } }), /PD ATOMesh.*DP=1/);
});

test("CPU offload composes with the Mooncake connector", () => {
  const model = structuredClone(recipe);
  model.kv_offload_support = { simple: "supported" };
  const result = resolve({ model, kv: "simple" });
  const config = JSON.parse(flag(result.prefill.argv, "--kv-transfer-config"));
  assert.equal(config.kv_connector, "MultiConnector");
  assert.deepEqual(config.kv_connector_extra_config.connectors.map((child) => child.kv_connector), ["MooncakeConnector", "SimpleCPUOffloadConnector"]);
});

test("AMD availability follows the shared Mooncake catalog", () => {
  const catalog = structuredClone(strategies);
  catalog.pd_cluster.connectors.mooncake.brands = ["NVIDIA", "AMD"];
  assert.equal(resolve({ hw: "mi355x", catalog }).pdRouter, "atomesh");
  catalog.pd_cluster.connectors.mooncake.brands = ["NVIDIA"];
  const result = resolve({ hw: "mi355x", catalog });
  assert.equal(result.pdRouter, "vllm");
  assert.match(result.routerUnavailableReason, /NVIDIA/);
});

test("JSON rendering preserves ATOMesh and the router container command", () => {
  const result = renderCommand(recipe, "default", "pd_cluster", "gb300", 1, [], strategies, taxonomy, pools, null, undefined, undefined, false, { pdRouter: "atomesh" });
  assert.equal(result.pd_router, "atomesh");
  assert.match(result.router.command, /--entrypoint atomesh/);
});

test("the existing default remains vLLM Router with NIXL", () => {
  const result = resolve({ router: null });
  assert.equal(result.pdRouter, "vllm");
  assert.equal(result.kvConnector.selected, "nixl");
});

