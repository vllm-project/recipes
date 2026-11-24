import assert from "node:assert/strict";
import fs from "node:fs";
import { test } from "node:test";
import yaml from "js-yaml";
import { ascendPdRecipe } from "./fixtures/ascend-pd.mjs";
import { renderCommand, resolveSelection } from "../src/lib/recipe-render.js";
import { poolDeploymentMarkdown, selectionMarkdown, selectionUrls } from "../src/lib/recipe-markdown.js";

const read = (file) => yaml.load(fs.readFileSync(new URL("../" + file, import.meta.url), "utf8"));
const taxonomy = read("taxonomy.yaml");
const strategies = Object.fromEntries(["single_node_tp", "multi_node_dep", "pd_cluster"].map((s) => [s, read(`strategies/${s}.yaml`)]));
const id = "vllm-ascend/memcache";
const key = "test_pool";
const hardware = "ascend_950dt";
const query = (changes) => new URLSearchParams({ hardware, variant: "default", strategy: "pd_cluster", preset: key, ...changes });

function model() {
  const r = ascendPdRecipe();
  r.strategy_hardware = { pd_cluster: { [hardware]: "unsupported" } };
  r._poolProfiles = { [id]: {
    label: "Memcache", store: { prefill: { backend: "memcache" }, decode: { backend: "memcache" } },
    inputs: { data_dir: { type: "path", scope: "host", label: "Backing directory" } },
    steps: [
      { id: "inspect", stage: "storage", target: "hosts", title: "Check storage", command: "ls -- {{data_dir}}", check: "Path is accessible" },
      { id: "create", mode: "new", stage: "control", target: "host:p", title: "Create the pool", command: "create-pool", check: "Pool is ready" },
      { id: "connect", mode: "reuse", stage: "control", target: "host:d", title: "Use existing storage", depends_on: ["inspect"], check: "Connection is ready" },
    ],
  } };
  const placements = ["prefill", "decode", "proxy"].map((role) => ({ role, container: role === "prefill" ? "cp" : "cd",
    launch: { argv: role === "proxy" ? ["proxy"] : ["vllm", "serve", "model"], transfer: { kv_connector: "MooncakeConnectorV2" } } }));
  placements[0].devices = [0, 1];
  placements.push({ role: "meta", container: "cp", mode: "new" });
  r.presets[key] = { label: "Qualified pool", strategy: "pd_cluster", hardware, variants: ["default"],
    image: "example/ascend:fixture",
    select: { pd_router: "ascend", pd_connector: "mooncake_ascend", kv_offload: id, frontend: "python" },
    deployment: { hosts: ["p", "d"], containers: { cp: { host: "p" }, cd: { host: "d" } }, placements },
  };
  return r;
}

test("pool permits its own PD strategy; explicit Off, ordinary presets and unrelated KV URLs retain baseline gating", () => {
  const r = model();
  const pick = (params) => resolveSelection(r, params, strategies, taxonomy);
  assert.equal(pick(query({})).rendering.pool_deployment.profile, id);
  for (const overrides of [{ preset: "baseline", kv_offload: id }, { kv_offload: "off" }, { preset: "ascend_pd" }, { variant: "missing" }]) {
    const resolved = pick(query(overrides));
    assert.notEqual(resolved.selection.strategy, "pd_cluster");
    assert.equal(resolved.rendering?.pool_deployment, undefined);
  }
  const invalid = pick(query({ pool_mode: "invalid", decode_nodes: "5" }));
  assert.equal(invalid.rendering.pool_deployment.mode, null);
  assert.deepEqual(invalid.selection.adjusted.map((a) => a.param).sort(), ["decode_nodes", "pool_mode"]);
});

test("static template, shared selection and site exports use the same ordered deployment", () => {
  const r = model();
  const shared = resolveSelection(r, query({}), strategies, taxonomy).rendering;
  const preset = renderCommand(r, "default", "pd_cluster", hardware, 1, [], strategies, taxonomy, null, null, key, null, true);
  assert.deepEqual(shared.pool_deployment, preset.pool_deployment);
  assert.ok(shared.pool_deployment.steps.some((s) => s.mode === "new"));
  assert.ok(shared.pool_deployment.steps.some((s) => s.mode === "reuse"));
  const params = query({ pool_mode: "reuse" });
  const path = "/disk/local host's cache";
  const resolved = resolveSelection(r, params, strategies, taxonomy, { poolInputs: { "hosts.d.data_dir": path } });
  const plan = resolved.rendering.pool_deployment;
  assert.ok(plan.steps.every((s) => s.mode !== "new"));
  const urls = selectionUrls("https://recipes.example", r.hf_id, params, hardware);
  const markdown = selectionMarkdown(r, resolved, { taxonomy, strategies, urls });
  for (const s of plan.steps.filter((s) => s.command)) assert.ok(markdown.includes(s.command));
  assert.ok(markdown.includes(path));
  assert.ok(!markdown.includes("create-pool"));
  assert.ok(!poolDeploymentMarkdown(plan, "p").includes(path));
  assert.ok(!JSON.stringify(urls).includes("local"));
  assert.ok(!JSON.stringify(shared).includes(path));
});

test("topology includes controls hosted for a new pool, their hosts, and each serving role's devices", () => {
  const r = model();
  for (const mode of ["", "new", "reuse"]) {
    const plan = resolveSelection(r, query(mode ? { pool_mode: mode } : {}), strategies, taxonomy).rendering.pool_deployment;
    assert.equal(poolDeploymentMarkdown(plan).includes("MetaService (new pool only) on \`p\` / \`cp\`"), mode !== "reuse");
    assert.equal(poolDeploymentMarkdown(plan, "p").includes("MetaService"), mode !== "reuse");
    assert.ok(!poolDeploymentMarkdown(plan, "d").includes("MetaService"));
    assert.match(poolDeploymentMarkdown(plan), /Prefill \(devices 0, 1\)/);
  }
});
