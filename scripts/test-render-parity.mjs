// The on-demand `.md` / `.json?<query>` routes resolve a page URL's query with
// resolveSelection; the static JSON API renders the same selections at build
// time. Every static per-hardware rendering (recommended + each strategy
// alternative) must come out identical when its selection is passed as a page
// query, or an agent reading `<page>.json?hardware=…` would get a different
// command than `/<hf_id>/hw/<hw>.json` for the same configuration.
//
// Run after `node scripts/build-recipes-api.mjs` (reads public/).

import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { test } from "node:test";
import { getAllRecipes } from "../src/lib/recipes.js";
import { loadStrategies } from "../src/lib/strategies.js";
import { loadTaxonomy } from "../src/lib/taxonomy.js";
import { resolveSelection, synthesizeInstall } from "../src/lib/recipe-render.js";
import { selectionMarkdown } from "../src/lib/recipe-markdown.js";
import { validateContextEntries } from "../src/lib/recipe-context.js";
import { presetSelect, resolvePresetKey } from "../src/lib/recipe-presets.js";
import { computeDockerMeta, effectiveCompatibleStrategies, isHardwareScalable, isStrategySupportedOnHardware, multiNodeFloor, needsMultiNode, isFrontendSupported, resolveFrontend, resolveOmniCommand, restoreVariantDefaults, versionBelow } from "../src/lib/command-synthesis.js";

const PUBLIC = process.env.RECIPES_PUBLIC_DIR ? path.resolve(process.env.RECIPES_PUBLIC_DIR) : path.join(process.cwd(), "public");
const strategies = loadStrategies();
const taxonomy = loadTaxonomy();
const readJson = (rel) => JSON.parse(fs.readFileSync(path.join(PUBLIC, rel), "utf8"));

test("Omni variants inherit, replace or clear model base args", () => {
  for (const key of ["default", "alternate"]) {
    const recipe = {
      model: { model_id: "example/model", base_args: ["--max-model-len", "1024"] },
      variants: { [key]: {} },
    };
    const command = () => resolveOmniCommand(recipe, key, { extraArgs: ["--port", "9000"] }, taxonomy.hardware_profiles.h200, "h200").command;
    assert.match(command(), /--max-model-len 1024/);
    recipe.variants[key].base_args = ["--max-model-len", "2048"];
    assert.match(command(), /--max-model-len 2048/);
    recipe.variants[key].base_args = [];
    assert.doesNotMatch(command(), /--max-model-len/);
    assert.match(command(), /--port 9000/);
    assert.match(command(), /--omni/);
  }
});

test("installation exports use the higher model or variant minimum", () => {
  for (const [modelMin, variantMin, expected] of [
    ["0.40.0", "0.31.0", "0.40.0"], ["0.9.0", "0.10.0", "0.10.0"],
    [undefined, "0.31.0", "0.31.0"], ["0.31.0", undefined, "0.31.0"],
    ["0.31.0", "0.31.0", "0.31.0"],
  ]) {
    const recipe = {
      hf_id: "example/model",
      model: { model_id: "example/model", min_vllm_version: modelMin, nightly_required: true },
      variants: { default: { precision: "bf16", vram_minimum_gb: 1, min_vllm_version: variantMin } },
      compatible_strategies: ["single_node_tp"],
    };
    const resolved = resolveSelection(recipe, new URLSearchParams("hardware=h200"), strategies, taxonomy);
    const md = selectionMarkdown(recipe, resolved, {
      strategies, taxonomy, urls: { page: "https://example.org/example/model", json: "https://example.org/example/model.json" },
    });
    assert.ok(md.includes(`Requires vLLM >= ${expected}`));
    assert.ok(synthesizeInstall(recipe, recipe.variants.default).pip.note.startsWith(`vLLM ${expected} `));
  }
});

test("explicit empty feature selections stay off", () => {
  const recipe = {
    hf_id: "example/model",
    model: { model_id: "example/model" },
    variants: { default: { precision: "bf16", vram_minimum_gb: 1 } },
    compatible_strategies: ["single_node_tp"],
    features: { reasoning: { args: ["--reasoning-parser", "example"] } },
  };
  const resolve = (q) => resolveSelection(recipe, new URLSearchParams(q), strategies, taxonomy);
  assert.deepEqual(resolve("hardware=h200").selection.features, ["reasoning"]);
  for (const features of ["", "none"]) {
    const result = resolve(`hardware=h200&features=${features}`);
    assert.deepEqual(result.selection.features, []);
    assert.equal(result.rendering.argv.includes("--reasoning-parser"), false);
  }
  const md = selectionMarkdown(recipe, resolve("hardware=h200&features=none"), {
    strategies, taxonomy, urls: { page: "https://example.org/example/model", json: "https://example.org/example/model.json" },
  });
  assert.match(md, /^Configuration: .*features none/m);
  assert.ok(md.includes("`features=none`"));
});

test("docker-disabled recipes omit containers across deployment layouts", () => {
  const recipe = {
    hf_id: "example/model",
    model: { model_id: "example/model", install: { docker: false } },
    variants: { default: { precision: "bf16", vram_minimum_gb: 1 } },
    compatible_strategies: ["single_node_tp", "multi_node_tp", "pd_cluster"],
  };
  const amd = computeDockerMeta(recipe, recipe.variants.default, taxonomy.hardware_profiles.mi300x, "mi300x");
  assert.equal(amd.brandKey, "amd");
  assert.ok(amd.gpuFlags.includes("/dev/kfd"));
  assert.equal(amd.image, null);

  for (const query of [
    "hardware=h200&strategy=single_node_tp",
    "hardware=h200&strategy=multi_node_tp&nodes=2",
    "hardware=h200&strategy=single_node_tp&kv_offload=kv_store_distributed_mooncake",
    "hardware=h200&strategy=pd_cluster&prefill_nodes=1&decode_nodes=1",
  ]) {
    const result = resolveSelection(recipe, new URLSearchParams(query), strategies, taxonomy).rendering;
    assert.equal(result.docker_image, null);
    if (result.deploy_type === "multi_node") {
      assert.deepEqual(result.worker_docker_commands, [null]);
      assert.deepEqual(result.worker_docker_argvs, [null]);
    }
    const containers = result.deploy_type === "pd_cluster" ? [result.prefill, result.decode] : [result];
    for (const container of containers) {
      const commands = Object.keys(container).filter((key) => key.endsWith("docker_command"));
      assert.ok(commands.length > 0);
      for (const key of commands) {
        assert.equal(container[key], null);
        assert.equal(container[key.replace(/_command$/, "_argv")], null);
      }
    }
    assert.deepEqual(JSON.parse(JSON.stringify(result)), result);
  }

  const query = new URLSearchParams("hardware=gb300&strategy=pd_cluster&pd_router=smg");
  const { router } = resolveSelection(recipe, query, strategies, taxonomy).rendering;
  assert.match(router.command, /smg launch/);
  assert.ok(!Object.hasOwn(router, "dockerCommand"));
  assert.ok(!Object.hasOwn(router, "dockerInstall"));
});

// Build-only annotations the static files add around a rendering.
const strip = ({ strategy_spec, hardware_profile, alternatives, by_hardware, ...rest }) => rest;

// Static renderings the builder can't select, so no page query reaches them:
// a single-node layout where the model needs more than one node (the builder
// opens on 2 nodes and disables 1), a single-node command at node_count > 1,
// or a serving strategy `strategy_hardware` rules out on that GPU.
function builderReachable(recipe, r) {
  const hw = taxonomy.hardware_profiles?.[r.hardware];
  const variant = recipe.variants?.[r.variant] || {};
  const serving = r.deploy_type === "kv_store_lb" ? r.serving_strategy : r.strategy;
  if (r.node_count === 1 && needsMultiNode(recipe, strategies, hw, variant, r.hardware)) return false;
  if (r.node_count > 1 && strategies[serving]?.deploy_type === "single_node") return false;
  return isStrategySupportedOnHardware(recipe, serving, hw, r.hardware);
}

// The page query that selects a static rendering.
function queryFor(r) {
  const q = new URLSearchParams({ hardware: r.hardware, nodes: String(r.node_count) });
  if (r.variant !== "default") q.set("variant", r.variant);
  if (r.deploy_type === "kv_store_lb") {
    q.set("strategy", r.serving_strategy);
    q.set("kv_offload", r.strategy);
  } else {
    q.set("strategy", r.strategy);
  }
  if (r.deploy_type === "pd_cluster") {
    q.set("prefill_nodes", String(r.prefill?.nodes ?? 1));
    q.set("decode_nodes", String(r.decode?.nodes ?? 1));
  }
  return q;
}

test("promoted JSON describes the checkpoint at its own URL", () => {
  for (const { json, derived_from } of readJson("models.json")) {
    if (!derived_from) continue;
    const checkpoint = readJson(json.slice(1));
    const variant = checkpoint.variants.default;
    const parent = readJson(`${derived_from}.json`);
    if (variant.base_args) assert.deepEqual(checkpoint.model.base_args, variant.base_args, json);
    if (variant.min_vllm_version) {
      assert.ok(!versionBelow(checkpoint.model.min_vllm_version, variant.min_vllm_version), json);
    }
    if (variant.supported_frontends) {
      assert.equal(checkpoint.model.default_frontend, resolveFrontend(parent, undefined, taxonomy, variant), json);
      for (const id of Object.keys(taxonomy.frontends)) {
        assert.equal(isFrontendSupported(checkpoint, id), isFrontendSupported(parent, id, variant), json);
      }
    }
    if (variant.supported_features) {
      const expected = Object.keys(parent.features || {}).filter((f) => variant.supported_features.includes(f));
      assert.deepEqual(Object.keys(checkpoint.features || {}), expected, json);
    }
    assert.deepEqual(validateContextEntries(checkpoint), [], json);
  }
});

test("resolveSelection renders every static per-hardware rendering identically", () => {
  const mismatches = [];
  let checked = 0;
  let unreachable = 0;
  for (const recipe of getAllRecipes()) {
    const parent = path.join(PUBLIC, `${recipe.hf_id}.json`);
    if (!fs.existsSync(parent)) continue;
    const byHw = readJson(`${recipe.hf_id}.json`).recommended_command?.by_hardware || {};
    for (const hwPath of Object.values(byHw)) {
      const rec = readJson(hwPath.slice(1));
      const renderings = [rec, ...Object.values(rec.alternatives || {}).map((p) => readJson(p.slice(1)))];
      for (const expected of renderings) {
        // PD with co-located pools (pdNodes null) has no Nodes-row equivalent.
        if (expected.deploy_type === "pd_cluster") continue;
        if (!builderReachable(recipe, expected)) { unreachable++; continue; }
        const q = queryFor(expected);
        const got = resolveSelection(recipe, q, strategies, taxonomy).rendering;
        checked++;
        try {
          assert.deepEqual(strip(got || {}), strip(expected));
        } catch {
          mismatches.push(`/${recipe.hf_id}.json?${q}`);
        }
      }
    }
  }
  console.log(`  ${checked} renderings checked, ${unreachable} not selectable in the builder skipped`);
  assert.ok(checked > 1000, `only ${checked} renderings checked`);
  assert.deepEqual(mismatches, [], `${mismatches.length}/${checked} renderings differ:\n  ${mismatches.slice(0, 20).join("\n  ")}`);
});

test("the builder's multi-node start matches resolveSelection, so a shared link and its .md agree", () => {
  const mismatches = [];
  let checked = 0;
  for (const recipe of getAllRecipes()) {
    if (recipe.meta?.tasks?.includes("omni")) continue;
    // A recipe with no multi-node strategy stays on one node, whatever fits.
    const offered = effectiveCompatibleStrategies(recipe);
    if (!offered.some((s) => ["multi_node", "pd_cluster"].includes(strategies[s]?.deploy_type))) continue;
    for (const [hwId, hw] of Object.entries(taxonomy.hardware_profiles)) {
      // A preset's `select.nodes` moves the count on its own GPU, in both.
      if (Object.values(recipe.presets || {}).some((p) => p.hardware === hwId)) continue;
      if (hw.restricted && !recipe.meta?.hardware?.[hwId]) continue;
      if (recipe.meta?.hardware?.[hwId] === "unsupported" || !isHardwareScalable(hw)) continue;
      for (const [variantKey, variant] of Object.entries(recipe.variants || {})) {
        if (variantKey !== "default" || !needsMultiNode(recipe, strategies, hw, variant, hwId)) continue;
        for (const requested of [null, ...offered.filter((s) => strategies[s]?.deploy_type === "multi_node")]) {
          if (requested && !isStrategySupportedOnHardware(recipe, requested, hw, hwId)) continue;
          const q = new URLSearchParams({ hardware: hwId, ...(requested ? { strategy: requested } : {}) });
          const { selection } = resolveSelection(recipe, q, strategies, taxonomy);
          if (selection.hardware !== hwId) continue;
          checked++;
          const start = multiNodeFloor(recipe, strategies, hw, hwId, requested);
          if (selection.strategy !== "pd_cluster" && selection.nodes !== start) {
            mismatches.push(`/${recipe.hf_id}?${q}: builder ${start}, resolveSelection ${selection.nodes}`);
          }
        }
      }
    }
  }
  assert.ok(checked > 100, `only ${checked} selections checked`);
  assert.deepEqual(mismatches, [], `${mismatches.length}/${checked} differ:\n  ${mismatches.slice(0, 20).join("\n  ")}`);
});

test("resolveSelection renders every preset point identically from ?preset=&point=", () => {
  const mismatches = [];
  let checked = 0;
  // Point annotations the preset files add; PD's top-level node_count is
  // nominal (each pool carries its own `nodes`).
  const stripPoint = (r) => {
    const { label, concurrency, description, ...rest } = strip(r);
    if (rest.deploy_type === "pd_cluster") delete rest.node_count;
    return rest;
  };
  for (const recipe of getAllRecipes()) {
    for (const key of Object.keys(recipe.presets || {})) {
      const rel = `${recipe.hf_id}/presets/${key}.json`;
      if (!fs.existsSync(path.join(PUBLIC, rel))) continue;
      for (const [point, expected] of Object.entries(readJson(rel).points || {})) {
        const q = new URLSearchParams({ hardware: expected.hardware, strategy: expected.serving_strategy || expected.strategy, preset: key });
        if (point !== "default") q.set("point", point);
        if (expected.variant !== "default") q.set("variant", expected.variant);
        if (expected.deploy_type === "pd_cluster") {
          q.set("prefill_nodes", String(expected.prefill.nodes));
          q.set("decode_nodes", String(expected.decode.nodes));
        } else if (expected.node_count > 1) {
          q.set("nodes", String(expected.node_count));
        }
        const got = resolveSelection(recipe, q, strategies, taxonomy).rendering;
        checked++;
        try {
          assert.deepEqual(stripPoint(got || {}), stripPoint(expected));
        } catch {
          mismatches.push(`/${recipe.hf_id}.json?${q}`);
        }
      }
    }
  }
  assert.ok(checked > 50, `only ${checked} preset points checked`);
  assert.deepEqual(mismatches, [], `${mismatches.length}/${checked} preset points differ:\n  ${mismatches.slice(0, 20).join("\n  ")}`);
});


const localRecipe = {
  hf_id: "example/model",
  model: { model_id: "example/model", base_args: ["--trust-remote-code"] },
  compatible_strategies: ["single_node_tp"],
  features: {
    tool_calling: { args: ["--enable-auto-tool-choice"] },
    reasoning: { args: ["--reasoning-parser", "example"] },
    unsupported: { args: ["--unsupported"] },
  },
  variants: {
    local: {
      precision: "int4", vram_minimum_gb: 4, supported_hardware: ["local"],
      base_args: ["--host", "127.0.0.1"],
      supported_frontends: ["python"], supported_features: ["tool_calling", "reasoning"],
      install: { pip: { command: "install-example-runtime" }, docker: false },
    },
  },
  notes: [{ when: { hardware: ["local"], variants: ["local"] }, body: "Local memory guidance." }],
};
const localTaxonomy = {
  ...taxonomy,
  hardware_profiles: {
    local: { brand: "Example", generation: "example", gpu_count: 1, vram_gb: 16, scalable: false, requires_kv_offload_opt_in: true },
  },
};

test("variant configuration is preserved in JSON and Markdown selections", () => {
  const params = { hardware: "local", variant: "local", frontend: "rust" };
  for (const features of [[], ["tool_calling"], ["reasoning"], ["tool_calling", "reasoning"]]) {
    const q = new URLSearchParams({ ...params, features: features.join(",") });
    const resolved = resolveSelection(localRecipe, q, strategies, localTaxonomy);
    const r = resolved.rendering;
    assert.deepEqual(resolved.selection.features, features);
    assert.deepEqual(resolved.options.features, ["tool_calling", "reasoning"]);
    assert.equal(resolved.selection.frontend, "python");
    assert.equal(r.argv.includes("--enable-auto-tool-choice"), features.includes("tool_calling"));
    assert.equal(r.argv.includes("--reasoning-parser"), features.includes("reasoning"));
    assert.equal(r.argv.includes("--trust-remote-code"), false);
    assert.equal(r.argv.includes("--unsupported"), false);
  }

  const resolved = resolveSelection(localRecipe, new URLSearchParams(params), strategies, localTaxonomy);
  const { rendering } = resolved;
  assert.equal(rendering.install.pip.command, "install-example-runtime");
  assert.equal(rendering.install.docker, undefined);
  assert.equal(rendering.docker_image, null);
  assert.equal(rendering.docker_command, null);
  const md = selectionMarkdown(localRecipe, resolved, {
    strategies, taxonomy: localTaxonomy,
    urls: { page: "https://example.org/example/model", json: "https://example.org/example/model.json" },
  });
  assert.ok(md.includes("install-example-runtime"));
  assert.ok(md.includes("Local memory guidance."));
  assert.ok(!md.includes("Docker image:"));
});

test("variant switches restore newly available defaults and preserve shared choices", () => {
  const previous = { supported_features: ["tool_calling", "reasoning"] };
  const defaults = new Set(["tool_calling", "reasoning", "encoder_parallel"]);
  const next = restoreVariantDefaults(["reasoning"], previous, defaults);
  assert.deepEqual(next, ["reasoning", "encoder_parallel"]);
  assert.deepEqual(restoreVariantDefaults(next, previous, defaults), next);
  assert.deepEqual(restoreVariantDefaults([], {}, defaults), []);
  const textOnly = ["reasoning", "text_only"];
  assert.deepEqual(restoreVariantDefaults(textOnly, previous, defaults), textOnly);
});

test("variant defaults respect preset and point opt-outs", () => {
  const previous = { supported_features: ["reasoning"] };
  const defaults = ["reasoning", "encoder_parallel"];
  const recipe = { presets: { tuned: {
    hardware: "h200", default: true, select: { features: { encoder_parallel: true } },
    points: { off: { default: true, select: { features: { encoder_parallel: false } } }, on: {} },
  } } };
  for (const [hw, pick, point, expected] of [
    ["h200", null, null, []], ["h200", null, "on", ["encoder_parallel"]],
    ["h200", "baseline", null, ["encoder_parallel"]], ["mi300x", null, null, ["encoder_parallel"]],
  ]) {
    const select = presetSelect(recipe, resolvePresetKey(recipe, hw, "single_node_tp", pick, "default"), point);
    assert.deepEqual(restoreVariantDefaults([], previous, defaults, select?.features), expected);
  }
  assert.deepEqual(restoreVariantDefaults(["encoder_parallel"], previous, defaults, { encoder_parallel: false }), ["encoder_parallel"]);
});

test("variant fallbacks are reported in JSON and Markdown", () => {
  const resolve = (params) => resolveSelection(localRecipe,
    new URLSearchParams({ hardware: "local", variant: "local", ...params }), strategies, localTaxonomy);
  const result = resolve({ frontend: "rust", features: "reasoning,unsupported" });
  assert.deepEqual(result.selection.adjusted.map(({ param, requested, used }) => ({ param, requested, used })), [
    { param: "frontend", requested: "rust", used: "python" },
    { param: "features", requested: "reasoning,unsupported", used: "reasoning" },
  ]);
  const md = selectionMarkdown(localRecipe, result, {
    strategies, taxonomy: localTaxonomy,
    urls: { page: "https://example.org/example/model", json: "https://example.org/example/model.json" },
  });
  assert.ok(md.includes("requested `frontend=rust`"));
  assert.ok(md.includes("requested `features=reasoning,unsupported`"));
  for (const features of ["", "none", "reasoning"]) {
    assert.deepEqual(resolve({ frontend: "python", features }).selection.adjusted, []);
  }
});

test("Apple hardware disables Docker even without a recipe opt-out", () => {
  const variant = { install: { pip: { command: "install-mlx" } } };
  const meta = computeDockerMeta({}, variant, taxonomy.hardware_profiles.apple_16gb, "apple_16gb");
  assert.equal(meta.brandKey, "apple");
  assert.equal(meta.dockerEnabled, false);
  assert.equal(meta.image, null);
  assert.equal(meta.gpuFlags, "");
  assert.deepEqual(synthesizeInstall({}, variant, meta), { pip: { command: "install-mlx" } });
});

test("KV offload follows profile opt-in without depending on the backend name", () => {
  const q = new URLSearchParams({ hardware: "local", variant: "local", kv_offload: "simple" });
  const render = (t) => resolveSelection(localRecipe, q, strategies, t).rendering;
  assert.equal(render(localTaxonomy).argv.includes("--kv-transfer-config"), false);
  const allowed = structuredClone(localTaxonomy);
  allowed.kv_offload.simple.brands = ["Example"];
  assert.equal(render(allowed).argv.includes("--kv-transfer-config"), true);
});
