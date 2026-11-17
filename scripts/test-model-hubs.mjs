import assert from "node:assert/strict";
import { test } from "node:test";
import { loadTaxonomy } from "../src/lib/taxonomy.js";
import { loadStrategies } from "../src/lib/strategies.js";
import { resolveCommand, computeDockerMeta, buildDockerRun, buildDockerArgv } from "../src/lib/command-synthesis.js";
import { renderCommand, resolveSelection } from "../src/lib/recipe-render.js";
import { ascendPdPools, ascendPdRecipe } from "./fixtures/ascend-pd.mjs";
import { selectionMarkdown } from "../src/lib/recipe-markdown.js";

const taxonomy = loadTaxonomy();
const strategies = loadStrategies();
const recipe = { hf_id: "org/model", model: { model_id: "org/model" },
  variants: { default: { precision: "bf16", vram_minimum_gb: 1 }, ms: { precision: "bf16", vram_minimum_gb: 1, model_id: "org/ms", hub: "modelscope" } },
  compatible_strategies: ["single_node_tp", "multi_node_tp", "pd_cluster"] };

test("ModelScope settings follow the variant across workers and Docker forms", () => {
  for (const strategy of ["single_node_tp", "multi_node_tp", "pd_cluster"]) {
    const result = resolveCommand(recipe, "ms", strategy, "h200", [], strategies, taxonomy, [], 2,
      strategy === "pd_cluster" ? { prefill: { nodes: 1 }, decode: { nodes: 1 } } : null);
    const roles = strategy === "pd_cluster" ? [result.prefill, result.decode] : [result];
    for (const role of roles) assert.equal(role.env.VLLM_USE_MODELSCOPE, "True");
  }
  const hf = resolveCommand(recipe, "default", "single_node_tp", "h200", [], strategies, taxonomy);
  assert.equal(hf.env.VLLM_USE_MODELSCOPE, undefined);
  const meta = computeDockerMeta(recipe, recipe.variants.ms, taxonomy.hardware_profiles.h200, "h200", taxonomy);
  const env = { VLLM_USE_MODELSCOPE: "True" };
  assert.match(buildDockerRun({ command: "vllm serve org/ms", env, ...meta }), /\.cache\/modelscope/);
  assert.ok(buildDockerArgv({ argv: ["vllm", "serve", "org/ms"], env, meta }).includes("~/.cache/modelscope:/root/.cache/modelscope"));
  const rendered = renderCommand(recipe, "ms", "single_node_tp", "h200", 1, [], strategies, taxonomy);
  assert.equal(rendered.hub, "modelscope");
  assert.equal(rendered.checkpoint_url, "https://modelscope.cn/models/org/ms");
});

const urls = { page: "https://example.org", json: "https://example.org/json" };

test("JSON and Markdown follow the selected variant and hardware pip path", () => {
  const select = (r, key) => {
    const result = resolveSelection(r, new URLSearchParams({ hardware: "h200", variant: key }), strategies, taxonomy);
    return { install: result.rendering.hub_install, md: selectionMarkdown(r, result, { strategies, taxonomy, urls }) };
  };
  const hf = select(recipe, "default");
  assert.equal(hf.install, undefined);
  assert.doesNotMatch(hf.md, /Checkpoint:|pip install.*modelscope/);
  const ms = select(recipe, "ms");
  assert.equal(ms.install.command, taxonomy.model_hubs.modelscope.install);
  assert.match(ms.md, /Checkpoint: \[ModelScope\]\(https:\/\/modelscope\.cn\/models\/org\/ms\)/);
  assert.ok(ms.md.includes(ms.install.command));
  assert.match(ms.md, /process-wide/);
  for (const override of [{ install: { pip: false } }, { hardware_overrides: { h200: { install: { pip: false } } } }]) {
    const dockerOnly = structuredClone(recipe);
    Object.assign(dockerOnly.variants.ms, override);
    const result = select(dockerOnly, "ms");
    assert.equal(result.install, undefined);
    assert.doesNotMatch(result.md, /pip install.*modelscope/);
  }
});

test("every Ascend PD node uses the ModelScope environment and Docker cache", () => {
  const r = ascendPdRecipe();
  r.variants.default.hub = "modelscope";
  const pools = ascendPdPools();
  const rendered = renderCommand(r, "default", "pd_cluster", "ascend_950dt", 1, [], strategies, taxonomy,
    pools, null, "ascend_pd", undefined, true);
  assert.equal(rendered.pd_router, "ascend");
  for (const role of ["prefill", "decode"]) {
    const nodes = rendered[role].node_commands;
    assert.equal(nodes.length, pools[role].nodes);
    for (const [rank, node] of nodes.entries()) {
      assert.equal(node.node_rank, rank);
      assert.equal(node.env.VLLM_USE_MODELSCOPE, "True");
      assert.match(node.docker_command, /VLLM_USE_MODELSCOPE=['"]?True/);
      assert.ok(node.docker_command.includes(taxonomy.model_hubs.modelscope.cache));
      assert.ok(node.docker_argv.includes("VLLM_USE_MODELSCOPE=True"));
      assert.ok(node.docker_argv.includes(taxonomy.model_hubs.modelscope.cache));
    }
  }
});

test("the selected PD router and adjustment report the ModelScope fallback", () => {
  for (const router of ["dynamo", "smg"]) {
    const query = new URLSearchParams({ hardware: "h200", variant: "ms", strategy: "pd_cluster", pd_router: router });
    const { rendering, selection } = resolveSelection(recipe, query, strategies, taxonomy);
    const reason = taxonomy.model_hubs.modelscope.unsupported_routers[router];
    assert.equal(rendering.pd_router, "vllm");
    assert.equal(rendering.router_unavailable_reason, reason);
    assert.equal(selection.pd_router, "vllm");
    assert.deepEqual(selection.adjusted.find(({ param }) => param === "pd_router"),
      { param: "pd_router", requested: router, used: "vllm", reason });
  }
});

test("a non-hub SMG fallback also reports the router adjustment", () => {
  const noSmg = { ...strategies, pd_cluster: { ...strategies.pd_cluster, smg: null } };
  const query = new URLSearchParams({ hardware: "h200", variant: "default", strategy: "pd_cluster", pd_router: "smg" });
  const { rendering, selection } = resolveSelection(recipe, query, noSmg, taxonomy);
  assert.match(rendering.router_unavailable_reason, /SMG is not configured/);
  assert.deepEqual(selection.adjusted.find(({ param }) => param === "pd_router"), {
    param: "pd_router", requested: "smg", used: "vllm", reason: rendering.router_unavailable_reason,
  });
});
