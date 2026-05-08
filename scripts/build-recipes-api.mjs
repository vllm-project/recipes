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
  computeDockerMeta,
  buildDockerRun,
  buildDockerArgv,
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
// in InstallBlock; defaults to NVIDIA + cu130 (today's upstream baseline).
// Per-recipe overrides at `model.install` follow the same semantics as the UI:
//   pip: false              → omit the `pip` key
//   pip: { command?, note?} → use overrides; missing fields fall back to defaults
//   (same for docker)
// `extras` carries the recipe's `dependencies` array verbatim.
function synthesizeInstall(recipe) {
  const installCfg = recipe.model?.install || {};
  const pipCfg = installCfg.pip;
  const dockerCfg = installCfg.docker;

  const v = recipe.model?.min_vllm_version || "";
  const nightlyRequired = recipe.model?.nightly_required === true;
  const baseCuda = "cu130";

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
// (hardware_opt_in_features[hw]). spec_decoding is treated as a normal opt-in
// (off by default); agents that want it must add it explicitly.
function defaultFeaturesFor(recipe, hwId) {
  const optIn = new Set(recipe.opt_in_features || []);
  for (const f of recipe.hardware_opt_in_features?.[hwId] || []) optIn.add(f);
  return Object.keys(recipe.features || {}).filter((f) => !optIn.has(f));
}

// Wrap a rendered (command, argv) pair in `docker run`. Returns
// { docker_command, docker_argv } so each form has its docker counterpart.
function dockerize(command, argv, env, dockerMeta, port = 8000) {
  return {
    docker_command: buildDockerRun({
      command, env, image: dockerMeta.image, gpuFlags: dockerMeta.gpuFlags, port,
    }),
    docker_argv: buildDockerArgv({ argv, env, meta: dockerMeta, port }),
  };
}

// Render one (recipe × variant × strategy × hardware × nodeCount) into the
// public JSON shape — handles all three deploy_types (single_node, multi_node,
// pd_cluster), snake_cases the keys, and attaches docker_run wrappers
// alongside the pip-mode command/argv. The `features` argument controls
// command synthesis but is intentionally NOT echoed back in the output —
// agents read the actual args from `command`/`argv`.
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

  const variant = recipe.variants?.default || {};
  const hwProfile = taxonomy.hardware_profiles?.[hwId] || {};
  const dockerMeta = computeDockerMeta(recipe, variant, hwProfile);
  const env = result.env || {};

  const base = {
    hardware: hwId,
    strategy,
    variant: "default",
    node_count: nodeCount,
    deploy_type: result.deployType,
    env,
    docker_image: dockerMeta.image,
  };
  if (result.deployType === "multi_node") {
    const head = dockerize(result.headCommand, result.headArgv, env, dockerMeta);
    const worker = dockerize(result.workerCommand, result.workerArgv, env, dockerMeta);
    return {
      ...base,
      head_command: result.headCommand,
      worker_command: result.workerCommand,
      head_argv: result.headArgv,
      worker_argv: result.workerArgv,
      head_docker_command: head.docker_command,
      worker_docker_command: worker.docker_command,
      head_docker_argv: head.docker_argv,
      worker_docker_argv: worker.docker_argv,
    };
  }
  if (result.deployType === "pd_cluster") {
    // Prefill exposes :8001, decode exposes :8002 (matches the router endpoints
    // emitted by command-synthesis).
    const prefill = { ...result.prefill };
    const decode = { ...result.decode };
    if (prefill.command && prefill.argv) {
      Object.assign(prefill, dockerize(prefill.command, prefill.argv, prefill.env, dockerMeta, 8001));
    }
    if (decode.command && decode.argv) {
      Object.assign(decode, dockerize(decode.command, decode.argv, decode.env, dockerMeta, 8002));
    }
    return {
      ...base,
      prefill,
      decode,
      router: result.router,
      router_config: result.routerConfig,
    };
  }
  return {
    ...base,
    command: result.command,
    argv: result.argv,
    ...dockerize(result.command, result.argv, env, dockerMeta),
  };
}

// Build the canonical "recommended" rendering for an agent: default variant,
// preferred hardware, recipe's default strategy (or recommendStrategy fallback),
// and YAML-default feature set. Inlines the strategy/hardware spec used by the
// recommended choice so agents don't need to fetch /strategies/<id>.json or
// /taxonomy.json. Returns:
//   { recommended, alternatives: { <strategy>: <rendered> } }
// where the caller writes each alternative to its own file and replaces it
// with a path link in the recipe JSON. Null for omni recipes.
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
  const recommendedFeatures = defaultFeaturesFor(recipe, hwId);

  const recommended = renderCommand(
    recipe, recommendedStrategy, hwId, recommendedNodeCount, recommendedFeatures, strategies, taxonomy
  );
  if (!recommended) return null;

  recommended.strategy_spec = strategies[recommendedStrategy];
  recommended.hardware_profile = hwProfile;

  const alternatives = {};
  for (const s of compatible) {
    if (s === recommendedStrategy) continue;
    const nc = s.startsWith("multi_node_") ? 2 : 1;
    const feats = defaultFeaturesFor(recipe, hwId);
    const rendered = renderCommand(recipe, s, hwId, nc, feats, strategies, taxonomy);
    if (rendered) alternatives[s] = rendered;
  }
  return { recommended, alternatives };
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
  // `dependencies` is the YAML-authored source — drop it from the public JSON
  // since `model.install.extras` already carries the same data verbatim.
  const install = synthesizeInstall(r);
  if (r.model) {
    const { install: _drop, ...rest } = r.model;
    r.model = install ? { ...rest, install } : rest;
  } else if (install) {
    r.model = { install };
  }
  delete r.dependencies;
  // Pre-render the canonical deploy command so agents don't have to reimplement
  // command-synthesis. Mirrors the website's default selections.
  const built = buildRecommendedCommand(r, strategies, taxonomy);
  if (built) {
    const { recommended, alternatives } = built;
    // Write each alternative to its own file under /<hf_id>/strategies/<s>.json
    // so the recipe JSON stays slim. The recipe links to them by path.
    const altLinks = {};
    for (const [s, rendered] of Object.entries(alternatives)) {
      const altPath = `${hfOrg}/${hfRepo}/strategies/${s}.json`;
      writeJson(altPath, rendered);
      altLinks[s] = `/${altPath}`;
    }
    if (Object.keys(altLinks).length) recommended.alternatives = altLinks;
    r.recommended_command = recommended;
  }
  // strategy_overrides, compatible_strategies, default_strategy are synthesis
  // inputs whose effects are already baked into recommended_command + the
  // per-strategy alternative files (whose keys are the compatible_strategies
  // set; the recommended one comes from default_strategy). Drop AFTER
  // synthesis runs — buildRecommendedCommand reads them. The YAML on GitHub
  // is the source of truth for anyone re-synthesizing.
  delete r.strategy_overrides;
  delete r.compatible_strategies;
  delete r.default_strategy;
  // Build the output object with `recommended_command` near the top — agents
  // hit it before scrolling past the YAML body. Order: identity → headline →
  // details → guide.
  const HEAD_KEYS = ["hf_id", "meta", "recommended_command"];
  const out = {};
  for (const k of HEAD_KEYS) if (k in r) out[k] = r[k];
  for (const [k, v] of Object.entries(r)) if (!HEAD_KEYS.includes(k)) out[k] = v;
  recipes.push(out);
  // JSON at /<org>/<repo>.json — mirrors HF URL scheme
  writeJson(`${hfOrg}/${hfRepo}.json`, out);
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
const altCount = recipes.reduce(
  (n, r) => n + Object.keys(r.recommended_command?.alternatives || {}).length, 0
);
console.log(
  `✓ JSON API: ${recipes.length} models (${rcCount} with recommended_command, ${altCount} alternative renderings), ${Object.keys(strategies).length} strategies`
);
console.log(`  /models.json`);
console.log(`  /{hf_id}.json                       (e.g. /moonshotai/Kimi-K2.5.json)`);
console.log(`  /{hf_id}/strategies/{strategy}.json (alternative renderings)`);
console.log(`  /strategies.json`);
console.log(`  /taxonomy.json`);
