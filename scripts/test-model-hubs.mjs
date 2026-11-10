import assert from "node:assert/strict";
import { test } from "node:test";
import { loadTaxonomy } from "../src/lib/taxonomy.js";
import { loadStrategies } from "../src/lib/strategies.js";
import { resolveCommand, computeDockerMeta, buildDockerRun, buildDockerArgv } from "../src/lib/command-synthesis.js";
import { renderCommand, resolveSelection } from "../src/lib/recipe-render.js";
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
  const selected = resolveSelection(recipe, new URLSearchParams("hardware=h200&variant=ms"), strategies, taxonomy);
  const md = selectionMarkdown(recipe, selected, { strategies, taxonomy, urls: { page: "https://example.org", json: "https://example.org/json" } });
  assert.match(md, /modelscope\.cn\/models\/org\/ms/);
  assert.match(md, /pip install.*modelscope/);
  assert.match(md, /process-wide/);
});

test("unverified ModelScope tokenizer routers fall back to vLLM router", () => {
  for (const router of ["dynamo", "smg"]) {
    const result = resolveCommand(recipe, "ms", "pd_cluster", "h200", [], strategies, taxonomy, [], 1,
      { prefill: { nodes: 1 }, decode: { nodes: 1 } }, {}, null, null, undefined, router);
    assert.notEqual(result.orchestrator, router);
    assert.match(result.routerUnavailableReason, /ModelScope/);
  }
});
