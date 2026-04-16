/**
 * Build-time script: reads YAML recipe/strategy/taxonomy files,
 * writes static JSON files for the API.
 *
 * API URLs (friendly, no /api/ prefix):
 *   /models.json                    — all recipes index
 *   /models/deepseek-v3.2.json      — single recipe
 *   /strategies.json                — all strategies
 *   /strategies/single_node_tp.json — single strategy
 *   /taxonomy.json                  — controlled vocabulary
 *   /quickstart/8x-h100.json       — hardware quickstart
 *
 * Run: node scripts/build-recipes-api.mjs
 */

import fs from "fs";
import path from "path";
import yaml from "js-yaml";

const ROOT = process.cwd();
const PUBLIC = path.join(ROOT, "public");

function readYaml(filePath) {
  return yaml.load(fs.readFileSync(filePath, "utf8"));
}

function writeJson(relPath, data) {
  const fullPath = path.join(PUBLIC, relPath);
  fs.mkdirSync(path.dirname(fullPath), { recursive: true });
  fs.writeFileSync(fullPath, JSON.stringify(data, null, 2));
}

// Normalize dates (js-yaml parses YYYY-MM-DD as Date objects)
function normalizeDates(obj) {
  if (obj instanceof Date) return obj.toISOString().split("T")[0];
  if (Array.isArray(obj)) return obj.map(normalizeDates);
  if (obj && typeof obj === "object") {
    return Object.fromEntries(Object.entries(obj).map(([k, v]) => [k, normalizeDates(v)]));
  }
  return obj;
}

// ── Taxonomy ──
const taxonomy = normalizeDates(readYaml(path.join(ROOT, "taxonomy.yaml")));
writeJson("taxonomy.json", taxonomy);

// ── Strategies ──
const strategiesDir = path.join(ROOT, "strategies");
const strategies = {};
for (const file of fs.readdirSync(strategiesDir).filter((f) => f.endsWith(".yaml"))) {
  const s = normalizeDates(readYaml(path.join(strategiesDir, file)));
  strategies[s.name] = s;
  writeJson(`strategies/${s.name}.json`, s);
}
writeJson("strategies.json", strategies);

// ── Recipes ──
const modelsDir = path.join(ROOT, "models");
const recipes = [];
for (const file of fs.readdirSync(modelsDir).filter((f) => f.endsWith(".yaml"))) {
  const r = normalizeDates(readYaml(path.join(modelsDir, file)));
  recipes.push(r);
  // /models/deepseek-v3.2.json
  writeJson(`models/${r.meta.slug}.json`, r);
}

// /models.json — index (no guide field, compact)
const index = recipes.map((r) => ({
  title: r.meta.title,
  slug: r.meta.slug,
  provider: r.meta.provider,
  description: r.meta.description,
  date_updated: r.meta.date_updated,
  difficulty: r.meta.difficulty,
  min_vllm_version: r.model.min_vllm_version,
  architecture: r.model.architecture,
  parameter_count: r.model.parameter_count,
  active_parameters: r.model.active_parameters,
  context_length: r.model.context_length,
  tasks: r.meta.tasks,
  performance_headline: r.meta.performance_headline,
  variants: Object.fromEntries(
    Object.entries(r.variants || {}).map(([k, v]) => [
      k,
      { precision: v.precision, vram_minimum_gb: v.vram_minimum_gb },
    ])
  ),
  compatible_strategies: r.compatible_strategies,
  features: Object.keys(r.features || {}),
  opt_in_features: r.opt_in_features || [],
  url: `/models/${r.meta.slug}`,
  json: `/models/${r.meta.slug}.json`,
}));
writeJson("models.json", index);

// ── Quickstart ──
const quickstartDir = path.join(ROOT, "quickstart");
if (fs.existsSync(quickstartDir)) {
  for (const file of fs.readdirSync(quickstartDir).filter((f) => f.endsWith(".yaml"))) {
    const q = normalizeDates(readYaml(path.join(quickstartDir, file)));
    writeJson(`quickstart/${q.hardware_profile}.json`, q);
  }
}

console.log(
  `✓ JSON API: ${recipes.length} models, ${Object.keys(strategies).length} strategies`
);
console.log(`  /models.json`);
console.log(`  /models/{slug}.json`);
console.log(`  /strategies.json`);
console.log(`  /taxonomy.json`);
