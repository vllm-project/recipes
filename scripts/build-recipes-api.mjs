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

// Resolve the default NVIDIA Docker image for a recipe, picking the base CUDA
// tag from a `{cu129, cu130}` map if present. Mirrors `computeDockerMeta` in
// CommandBuilder.jsx but only for the NVIDIA path (the canonical default).
function resolveDockerImage(recipe, baseCuda) {
  const DEFAULT_IMAGE = "vllm/vllm-openai:latest";
  const override = recipe.model?.docker_image;
  if (!override) return DEFAULT_IMAGE;
  if (typeof override === "string") return override;
  if (typeof override !== "object") return DEFAULT_IMAGE;

  const isCudaMap = (v) => v && typeof v === "object" && ("cu129" in v || "cu130" in v);
  const isBrandKeyed = "nvidia" in override || "amd" in override || "tpu" in override;

  const pickFromCudaMap = (m) => m[baseCuda] || m.cu129 || m.cu130 || DEFAULT_IMAGE;

  if (isBrandKeyed) {
    const nv = override.nvidia;
    if (typeof nv === "string") return nv;
    if (isCudaMap(nv)) return pickFromCudaMap(nv);
    return DEFAULT_IMAGE;
  }
  if (isCudaMap(override)) return pickFromCudaMap(override);
  return DEFAULT_IMAGE;
}

// Synthesize the canonical NVIDIA install commands (pip + docker) for a recipe
// so JSON consumers see the same one-liners the site shows. Mirrors the logic
// in InstallBlock; defaults to NVIDIA + the model's base CUDA tag (cu130 for
// vLLM ≥0.20.0, cu129 for older). Per-recipe overrides at `model.install`
// follow the same semantics as the UI:
//   pip: false              → omit the `pip` key
//   pip: { command?, note?} → use overrides; missing fields fall back to defaults
//   (same for docker)
// `extras` carries the recipe's `dependencies` array verbatim.
function synthesizeInstall(recipe) {
  const installCfg = recipe.model?.install || {};
  const pipCfg = installCfg.pip;
  const dockerCfg = installCfg.docker;

  const v = recipe.model?.min_vllm_version || "";
  const [maj, min] = v.split(".").map((n) => parseInt(n, 10) || 0);
  const is020Plus = maj > 0 || min >= 20;
  const nightlyRequired = recipe.model?.nightly_required === true;
  const baseCuda = nightlyRequired || is020Plus ? "cu130" : "cu129";

  const out = {};

  if (pipCfg !== false) {
    const defaultPipCmd = nightlyRequired
      ? `uv venv\nsource .venv/bin/activate\nuv pip install -U vllm --pre \\\n  --extra-index-url https://wheels.vllm.ai/nightly/${baseCuda} \\\n  --extra-index-url https://download.pytorch.org/whl/${baseCuda} \\\n  --index-strategy unsafe-best-match`
      : `uv venv\nsource .venv/bin/activate\nuv pip install -U vllm --torch-backend auto`;
    const defaultPipNote = nightlyRequired
      ? `vLLM ${v} isn't released yet — nightly required.`
      : undefined;
    const command = (pipCfg && pipCfg.command) || defaultPipCmd;
    const note = (pipCfg && pipCfg.note) || defaultPipNote;
    out.pip = note ? { command, note } : { command };
  }

  if (dockerCfg !== false) {
    const defaultDockerCmd = `docker pull ${resolveDockerImage(recipe, baseCuda)}`;
    const command = (dockerCfg && dockerCfg.command) || defaultDockerCmd;
    const note = dockerCfg && dockerCfg.note;
    out.docker = note ? { command, note } : { command };
  }

  if (Array.isArray(recipe.dependencies) && recipe.dependencies.length) {
    out.extras = recipe.dependencies;
  }

  return Object.keys(out).length ? out : undefined;
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
  const install = synthesizeInstall(r);
  if (install) r.install = install;
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
