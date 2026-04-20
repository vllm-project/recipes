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

// Recursively find all .yaml files under a directory
function findYamlFiles(dir) {
  const results = [];
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      results.push(...findYamlFiles(full));
    } else if (entry.name.endsWith(".yaml") || entry.name.endsWith(".yml")) {
      results.push(full);
    }
  }
  return results;
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

// ── Recipes ── (walks models/<hf_org>/<hf_repo>.yaml)
const modelsDir = path.join(ROOT, "models");
const recipes = [];
for (const file of findYamlFiles(modelsDir)) {
  const r = normalizeDates(readYaml(file));
  // Derive HF identity from path
  const rel = path.relative(modelsDir, file);
  const parts = rel.split(path.sep);
  if (parts.length >= 2) {
    r.hf_org = parts[0];
    r.hf_repo = parts[parts.length - 1].replace(/\.(yaml|yml)$/, "");
    r.hf_id = `${r.hf_org}/${r.hf_repo}`;
  }
  recipes.push(r);
  // JSON at /<org>/<repo>.json — mirrors HF URL scheme
  writeJson(`${r.hf_org}/${r.hf_repo}.json`, r);
}

// /models.json — slim discovery index (~5 KB). For agents that want to
// enumerate recipes and follow links; per-recipe data lives at
// `/<hf_org>/<hf_repo>.json` (the `json` pointer below).
const index = recipes.map((r) => ({
  hf_id: r.hf_id,
  title: r.meta.title,
  provider: r.meta.provider,
  url: `/${r.hf_id}`,
  json: `/${r.hf_id}.json`,
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
console.log(`  /{hf_org}/{hf_repo}.json  (e.g. /moonshotai/Kimi-K2.5.json)`);
console.log(`  /strategies.json`);
console.log(`  /taxonomy.json`);
