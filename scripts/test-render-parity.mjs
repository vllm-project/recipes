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
import { resolveSelection } from "../src/lib/recipe-render.js";
import { selectionMarkdown } from "../src/lib/recipe-markdown.js";
import { selectedVersion } from "../src/lib/platform-builds.js";
import { computeDockerMeta, effectiveCompatibleStrategies, isHardwareScalable, isStrategySupportedOnHardware, multiNodeFloor, needsMultiNode } from "../src/lib/command-synthesis.js";

const PUBLIC = process.env.RECIPES_PUBLIC_DIR ? path.resolve(process.env.RECIPES_PUBLIC_DIR) : path.join(process.cwd(), "public");
const strategies = loadStrategies();
const taxonomy = loadTaxonomy();
const readJson = (rel) => JSON.parse(fs.readFileSync(path.join(PUBLIC, rel), "utf8"));

test("platform build versions honor hardware overrides without changing commands", () => {
  const base = {
    hf_id: "example/model",
    meta: { hardware: { "810e": "supported", m890: "supported" } },
    model: { model_id: "example/model", min_vllm_version: "0.34.0" },
    variants: { default: { precision: "bf16", vram_minimum_gb: 1 } },
    compatible_strategies: ["single_node_tp"],
  };
  const withBuilds = structuredClone(base);
  withBuilds.model.platform_builds = {
    zhenwu: "0.32.0",
    h200: "0.33.0",
  };
  assert.equal(selectedVersion(base.model, "810e", taxonomy.hardware_profiles).label, "vLLM 0.34.0+");
  assert.equal(selectedVersion(withBuilds.model, "810e", taxonomy.hardware_profiles).label, "vLLM 0.32.0+");
  assert.equal(selectedVersion(withBuilds.model, "m890", taxonomy.hardware_profiles).label, "vLLM 0.32.0+");
  assert.equal(selectedVersion(withBuilds.model, "b300", taxonomy.hardware_profiles).label, "vLLM 0.34.0+");
  assert.equal(selectedVersion(withBuilds.model, "h200", taxonomy.hardware_profiles).label, "vLLM 0.33.0 · NVIDIA custom build");
  const overridden = structuredClone(withBuilds.model);
  overridden.platform_builds["810e"] = "0.33.0";
  assert.equal(selectedVersion(overridden, "810e", taxonomy.hardware_profiles).label, "vLLM 0.33.0+");
  assert.equal(selectedVersion(overridden, "m890", taxonomy.hardware_profiles).label, "vLLM 0.32.0+");
  for (const hw of ["810e", "m890"]) {
    const query = new URLSearchParams(`hardware=${hw}`);
    const original = resolveSelection(base, query, strategies, taxonomy);
    const result = resolveSelection(withBuilds, query, strategies, taxonomy);
    const { platform_builds, ...rendering } = result.rendering;
    assert.deepEqual(platform_builds, withBuilds.model.platform_builds);
    assert.deepEqual(rendering, original.rendering);
    const md = selectionMarkdown(withBuilds, result, {
      strategies, taxonomy, urls: { page: "https://example.org/example/model", json: "https://example.org/example/model.json" },
    });
    assert.match(md, /Platform build: vLLM 0\.32\.0\+/);
    assert.match(md, /Requires vLLM 0\.32\.0\+/);
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
