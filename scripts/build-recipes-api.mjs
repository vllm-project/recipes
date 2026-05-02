/**
 * Build-time script: reads YAML recipe/strategy/taxonomy files,
 * writes static JSON files for the API.
 *
 * API URLs (friendly, no /api/ prefix):
 *   /models.json                    — all recipes index
 *   /{hf_id}.json                   — single recipe (e.g. /deepseek-ai/DeepSeek-V3.2.json)
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
import {
  pickDefaultHardware,
  recommendStrategy,
  fitsSingleNode,
  resolveCommand,
} from "../src/lib/command-synthesis.js";

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

// Mirror CommandBuilder.jsx: features default to (all) − (opt_in_features) −
// (hardware_opt_in_features[hw]). On TP/TEP strategies, spec_decoding is
// auto-enabled when the recipe declares it (latency-oriented default).
function defaultFeaturesFor(recipe, hwId, strategyName) {
  const optIn = new Set(recipe.opt_in_features || []);
  for (const f of recipe.hardware_opt_in_features?.[hwId] || []) optIn.add(f);
  const features = Object.keys(recipe.features || {}).filter((f) => !optIn.has(f));
  const isLatency =
    strategyName === "single_node_tp" ||
    strategyName === "multi_node_tp" ||
    strategyName === "single_node_tep" ||
    strategyName === "multi_node_tep";
  if (isLatency && (recipe.features || {}).spec_decoding && !features.includes("spec_decoding")) {
    features.push("spec_decoding");
  }
  return features;
}

// Render one (recipe × variant × strategy × hardware × nodeCount) into the
// public JSON shape — handles all three deploy_types (single_node, multi_node,
// pd_cluster) and snake_cases the keys.
function renderCommand(recipe, strategy, hwId, nodeCount, features, strategies, taxonomy) {
  let result;
  try {
    result = resolveCommand(
      recipe, "default", strategy, hwId, features, strategies, taxonomy, [], nodeCount, null
    );
  } catch (e) {
    console.warn(`  ⚠ command synthesis failed for ${recipe.hf_id} / ${strategy}: ${e.message}`);
    return null;
  }
  if (!result) return null;

  const base = {
    hardware: hwId,
    strategy,
    variant: "default",
    node_count: nodeCount,
    features,
    deploy_type: result.deployType,
    env: result.env || {},
  };
  if (result.deployType === "multi_node") {
    return {
      ...base,
      head_command: result.headCommand,
      worker_command: result.workerCommand,
      head_argv: result.headArgv,
      worker_argv: result.workerArgv,
    };
  }
  if (result.deployType === "pd_cluster") {
    return {
      ...base,
      prefill: result.prefill,
      decode: result.decode,
      router: result.router,
      router_config: result.routerConfig,
    };
  }
  return { ...base, command: result.command, argv: result.argv };
}

// Build the canonical "recommended" rendering for an agent: default variant,
// preferred hardware, recipe's default strategy (or recommendStrategy fallback),
// and default feature set. Also returns an `alternatives` map keyed by strategy
// id covering every other entry in `compatible_strategies` (same hardware +
// variant), and inlines the strategy/hardware spec used by the recommended
// choice so agents don't need to fetch /strategies/<id>.json or /taxonomy.json.
// Returns null for recipes that don't run via `vllm serve` (omni recipes).
function buildRecommendedCommand(recipe, strategies, taxonomy) {
  const variant = recipe.variants?.default;
  if (!variant) return null;
  if ((recipe.meta?.tasks || []).includes("omni")) return null;

  const hwId = pickDefaultHardware(taxonomy.hardware_profiles || {}, variant, recipe);
  const hwProfile = taxonomy.hardware_profiles?.[hwId] || {};
  const compatible = recipe.compatible_strategies || [];
  const supportsMultiNode = compatible.some((s) => s.startsWith("multi_node_"));
  const recommendedNodeCount = !fitsSingleNode(hwProfile, variant) && supportsMultiNode ? 2 : 1;
  const recommendedStrategy = recommendStrategy(recipe, hwProfile, recommendedNodeCount);
  const recommendedFeatures = defaultFeaturesFor(recipe, hwId, recommendedStrategy);

  const recommended = renderCommand(
    recipe, recommendedStrategy, hwId, recommendedNodeCount, recommendedFeatures, strategies, taxonomy
  );
  if (!recommended) return null;

  // Inline the strategy + hw spec the recommendation depends on. Saves the
  // agent two follow-up fetches and keeps the deploy decision self-contained.
  recommended.strategy_spec = strategies[recommendedStrategy];
  recommended.hardware_profile = hwProfile;

  // Per-strategy alternatives. Each uses the same hardware + variant; nodeCount
  // is 2 for multi_node_* (the canonical scale-out example), 1 otherwise.
  // pd_cluster falls through resolveCommand's legacy 1-node fallback.
  const commands = {};
  for (const s of compatible) {
    if (s === recommendedStrategy) continue;
    const nc = s.startsWith("multi_node_") ? 2 : 1;
    const feats = defaultFeaturesFor(recipe, hwId, s);
    const rendered = renderCommand(recipe, s, hwId, nc, feats, strategies, taxonomy);
    if (rendered) commands[s] = rendered;
  }
  if (Object.keys(commands).length) recommended.alternatives = commands;

  return recommended;
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
  // Derive HF identity from path. Only `hf_id` is exposed in the public JSON;
  // `org` and `repo` are trivially `hf_id.split("/")` for consumers.
  const rel = path.relative(modelsDir, file);
  const parts = rel.split(path.sep);
  let hfOrg = "";
  let hfRepo = "";
  if (parts.length >= 2) {
    hfOrg = parts[0];
    hfRepo = parts[parts.length - 1].replace(/\.(yaml|yml)$/, "");
    r.hf_id = `${hfOrg}/${hfRepo}`;
  }
  // Replace the raw `model.install` config with the synthesized commands so
  // JSON consumers see the rendered one-liners (pip + docker) plus any
  // `extras` from `dependencies`. The raw toggle config is internal-only.
  const install = synthesizeInstall(r);
  if (r.model) {
    const { install: _drop, ...rest } = r.model;
    r.model = install ? { ...rest, install } : rest;
  } else if (install) {
    r.model = { install };
  }
  // Pre-render the canonical deploy command so agents don't have to reimplement
  // command-synthesis. Mirrors the website's default selections.
  const recommended = buildRecommendedCommand(r, strategies, taxonomy);
  if (recommended) r.recommended_command = recommended;
  recipes.push(r);
  // JSON at /<org>/<repo>.json — mirrors HF URL scheme
  writeJson(`${hfOrg}/${hfRepo}.json`, r);
}

// /models.json — slim discovery index (~5 KB). For agents that want to
// enumerate recipes and follow links; per-recipe data lives at
// `/<hf_id>.json` (the `json` pointer below).
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

const rcCount = recipes.filter((r) => r.recommended_command).length;
console.log(
  `✓ JSON API: ${recipes.length} models (${rcCount} with recommended_command), ${Object.keys(strategies).length} strategies`
);
console.log(`  /models.json`);
console.log(`  /{hf_id}.json             (e.g. /moonshotai/Kimi-K2.5.json)`);
console.log(`  /strategies.json`);
console.log(`  /taxonomy.json`);
