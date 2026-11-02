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
import { isStrategySupportedOnHardware, needsMultiNode } from "../src/lib/command-synthesis.js";

const PUBLIC = process.env.RECIPES_PUBLIC_DIR ? path.resolve(process.env.RECIPES_PUBLIC_DIR) : path.join(process.cwd(), "public");
const strategies = loadStrategies();
const taxonomy = loadTaxonomy();
const readJson = (rel) => JSON.parse(fs.readFileSync(path.join(PUBLIC, rel), "utf8"));

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
