// Invariants every published JSON file must hold, checked against the built
// API rather than against a rendering this file constructs. The point is to
// catch a build that writes self-contradictory output: a command serving a
// checkpoint the recipe excludes from that hardware, a preset whose declared
// `variant` disagrees with the checkpoint in its own command, a TP wider than
// the node can hold.
//
// `test:render` compares resolveSelection() against these files, which catches
// a *disagreement* between the two paths. This file checks each file on its
// own terms, so it fails even when both paths are wrong in the same way.
//
// Run after `node scripts/build-recipes-api.mjs` (reads public/).

import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { test } from "node:test";
import { getAllRecipes } from "../src/lib/recipes.js";
import { loadTaxonomy } from "../src/lib/taxonomy.js";
import { loadStrategies } from "../src/lib/strategies.js";
import { resolveSelection } from "../src/lib/recipe-render.js";
import { variantRunsOnHardware } from "../src/lib/command-synthesis.js";

const PUBLIC = process.env.RECIPES_PUBLIC_DIR ? path.resolve(process.env.RECIPES_PUBLIC_DIR) : path.join(process.cwd(), "public");
const taxonomy = loadTaxonomy();
const strategies = loadStrategies();
const profiles = taxonomy.hardware_profiles || {};

const recipes = getAllRecipes();

function jsonFiles(dir, out = []) {
  if (!fs.existsSync(dir)) return out;
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) jsonFiles(full, out);
    else if (entry.name.endsWith(".json")) out.push(full);
  }
  return out;
}

const readJson = (file) => JSON.parse(fs.readFileSync(file, "utf8"));

// Every rendering a recipe publishes: the per-hardware files, their strategy
// alternatives, and each preset's points. Omni recipes carry a different shape
// (`{ command }`, no argv) and are covered by their own checks.
function* publishedRenderings(recipe) {
  for (const file of jsonFiles(path.join(PUBLIC, recipe.hf_id))) {
    const doc = readJson(file);
    const rel = path.relative(PUBLIC, file);
    if (doc?.points) {
      for (const [point, rendering] of Object.entries(doc.points)) {
        yield { rel: `${rel}#${point}`, rendering, preset: doc };
      }
    } else if (doc?.argv || doc?.head_argv || doc?.prefill?.argv) {
      yield { rel, rendering: doc, preset: null };
    }
  }
}

const flagValue = (argv, flag) => {
  const at = argv.indexOf(flag);
  return at >= 0 ? argv[at + 1] : undefined;
};

// The argv a rendering launches with: `argv` for a single-node deployment,
// `head_argv` for multi-node, the prefill pool's for a PD cluster (both pools
// serve the same checkpoint, so either answers the question).
function launchArgv(rendering) {
  return rendering.argv || rendering.head_argv || rendering.prefill?.argv || [];
}

// The checkpoint a rendering serves. `vllm serve <id>` and Dynamo's
// `python -m dynamo.vllm --model <id>` are the two launch forms published here.
function servedCheckpoint(rendering) {
  const argv = launchArgv(rendering);
  const afterServe = argv[argv.indexOf("serve") + 1];
  if (argv.includes("serve") && afterServe) return afterServe;
  return flagValue(argv, "--model");
}

test("every published rendering runs the variant it declares, on the hardware it declares", () => {
  const wrong = [];
  let checked = 0;
  for (const recipe of recipes) {
    for (const { rel, rendering } of publishedRenderings(recipe)) {
      const { hardware, variant } = rendering;
      const profile = profiles[hardware];
      const cfg = recipe.variants?.[variant];
      if (!profile || !cfg) continue;
      checked++;
      if (!variantRunsOnHardware(profile, cfg, hardware)) {
        wrong.push(`${rel}: variant "${variant}" does not run on ${hardware} by the recipe's own rules`);
      }
    }
  }
  assert.ok(checked > 500, `only ${checked} renderings checked`);
  assert.deepEqual(wrong, [], `${wrong.length}/${checked} renderings contradict their recipe:\n  ${wrong.slice(0, 15).join("\n  ")}`);
});

test("a published rendering serves the checkpoint its variant names", () => {
  const wrong = [];
  let checked = 0;
  for (const recipe of recipes) {
    for (const { rel, rendering } of publishedRenderings(recipe)) {
      const cfg = recipe.variants?.[rendering.variant];
      const expected = cfg?.model_id || recipe.model?.model_id;
      const served = servedCheckpoint(rendering);
      if (!expected || !served) continue;
      checked++;
      if (served !== expected) wrong.push(`${rel}: declares variant "${rendering.variant}" (${expected}) but serves ${served}`);
    }
  }
  assert.ok(checked > 500, `only ${checked} renderings checked`);
  assert.deepEqual(wrong, [], `${wrong.length}/${checked} serve a different checkpoint than they declare:\n  ${wrong.slice(0, 15).join("\n  ")}`);
});

test("a multi-node rendering never asks for more ranks than its nodes hold", () => {
  const wrong = [];
  let checked = 0;
  for (const recipe of recipes) {
    for (const { rel, rendering } of publishedRenderings(recipe)) {
      if (rendering.deploy_type !== "multi_node") continue;
      const tp = parseInt(flagValue(launchArgv(rendering), "--tensor-parallel-size"), 10);
      const gpus = profiles[rendering.hardware]?.gpu_count;
      const nodes = rendering.node_count || 1;
      if (!Number.isFinite(tp) || !gpus) continue;
      checked++;
      if (tp > gpus * nodes) wrong.push(`${rel}: --tensor-parallel-size ${tp} exceeds ${gpus} GPU x ${nodes} node(s)`);
    }
  }
  assert.ok(checked > 100, `only ${checked} multi-node renderings checked`);
  assert.deepEqual(wrong, [], `${wrong.length}/${checked} over-subscribe their nodes:\n  ${wrong.slice(0, 15).join("\n  ")}`);
});

test("each preset file's declared variant matches the checkpoint in its own commands", () => {
  const wrong = [];
  let checked = 0;
  for (const recipe of recipes) {
    for (const { rel, rendering, preset } of publishedRenderings(recipe)) {
      if (!preset || !launchArgv(rendering).length) continue;
      const cfg = recipe.variants?.[rendering.variant];
      const expected = cfg?.model_id || recipe.model?.model_id;
      const served = servedCheckpoint(rendering);
      if (!expected || !served) continue;
      checked++;
      // The file's top-level `variant` is what a reader trusts; every point
      // below it must serve that same checkpoint.
      if (preset.variant !== rendering.variant) {
        wrong.push(`${rel}: file declares variant "${preset.variant}", point renders "${rendering.variant}"`);
      } else if (served !== expected) {
        wrong.push(`${rel}: declares variant "${rendering.variant}" (${expected}) but serves ${served}`);
      }
    }
  }
  assert.ok(checked > 100, `only ${checked} preset points checked`);
  assert.deepEqual(wrong, [], `${wrong.length}/${checked} preset points disagree with their file:\n  ${wrong.slice(0, 15).join("\n  ")}`);
});

test("a per-strategy file renders the deployment it names from the page query", () => {
  const wrong = [];
  let checked = 0;
  for (const recipe of recipes) {
    const base = path.join(PUBLIC, recipe.hf_id);
    if (!fs.existsSync(base)) continue;
    for (const file of jsonFiles(base)) {
      const match = path.relative(base, file).match(/^hw[/\\]([^/\\]+)[/\\]strategies[/\\](.+)\.json$/);
      if (!match) continue;
      const [, hardware, strategy] = match;
      const expected = readJson(file);
      if (!expected?.head_argv) continue;
      checked++;
      const query = new URLSearchParams({ hardware, strategy });
      if (expected.variant && expected.variant !== "default") query.set("variant", expected.variant);
      if (expected.node_count > 1) query.set("nodes", String(expected.node_count));
      const got = resolveSelection(recipe, query, strategies, taxonomy)?.rendering;
      if (!got) wrong.push(`${path.relative(PUBLIC, file)}: the page query renders nothing`);
      else if (got.deploy_type !== expected.deploy_type) {
        wrong.push(`${path.relative(PUBLIC, file)}: file says ${expected.deploy_type}, the query renders ${got.deploy_type}`);
      }
    }
  }
  assert.ok(checked > 500, `only ${checked} per-strategy files checked`);
  assert.deepEqual(wrong, [], `${wrong.length}/${checked} per-strategy files are unreachable or mis-typed:\n  ${wrong.slice(0, 15).join("\n  ")}`);
});
