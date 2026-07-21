/**
 * Build-time script: reads YAML recipe/strategy/taxonomy files,
 * writes static JSON files for the API.
 *
 * API URLs (friendly, no /api/ prefix):
 *   /models.json                                — all recipes index
 *   /{hf_id}.json                               — single recipe (default-hw rendering)
 *   /{hf_id}/strategies/{strategy}.json         — default-hw alternative (per strategy)
 *   /{hf_id}/hw/{hw}.json                       — per-hardware rendering
 *   /{hf_id}/hw/{hw}/strategies/{strategy}.json — per-(hw, strategy) alternative
 *   /strategies.json                            — all strategies
 *   /strategies/{strategy}.json                 — single strategy spec
 *   /taxonomy.json                              — controlled vocabulary
 *   /quickstart/8x-h100.json                    — hardware quickstart
 *
 * Run: node scripts/build-recipes-api.mjs
 */

import fs from "fs";
import path from "path";
import yaml from "js-yaml";
import {
  pickDefaultHardware,
  listCompatibleHardware,
  recommendStrategy,
  fitsSingleNode,
  isHardwareScalable,
  isKvStoreBrandSupported,
  pickFittingVariant,
  pdFitsSingleNode,
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
  const isBrandKeyed = "nvidia" in override || "amd" in override || "tpu" in override || "intel" in override;

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

// Pick the per-role node count for a `pd_cluster` rendering.
//   null   → co-located on a single node (variant fits the 2× PD VRAM budget)
//   {prefill,decode} → one full node per role × N, derived from
//                       ceil(variant.vram_minimum_gb / hwProfile.vram_gb)
//   "skip" → would need >4 nodes per role; don't emit a fantasy cluster
function pickPdNodes(hwProfile, variant) {
  if (pdFitsSingleNode(hwProfile, variant)) return null;
  const nodeVram = typeof hwProfile?.vram_gb === "number" ? hwProfile.vram_gb : 0;
  const modelVram = variant?.vram_minimum_gb || 0;
  if (nodeVram <= 0 || modelVram <= 0) return null;
  const nodesPerRole = Math.ceil(modelVram / nodeVram);
  if (nodesPerRole > 4) return "skip";
  return { prefill: nodesPerRole, decode: nodesPerRole };
}

// Render one (recipe × variant × strategy × hardware × nodeCount) into the
// public JSON shape — handles all three deploy_types (single_node, multi_node,
// pd_cluster), snake_cases the keys, and attaches docker_run wrappers
// alongside the pip-mode command/argv. The `features` argument controls
// command synthesis but is intentionally NOT echoed back in the output —
// agents read the actual args from `command`/`argv`.
function renderCommand(recipe, variantKey, strategy, hwId, nodeCount, features, strategies, taxonomy, pdNodes = null, kvOffload = null) {
  let result;
  try {
    result = resolveCommand(
      recipe, variantKey, strategy, hwId, features, strategies, taxonomy, [], nodeCount, pdNodes,
      {}, kvOffload
    );
  } catch (e) {
    console.warn(`  ⚠ command synthesis failed for ${recipe.hf_id} / ${variantKey} / ${strategy}: ${e.message}`);
    return null;
  }
  if (!result) return null;

  const variant = recipe.variants?.[variantKey] || recipe.variants?.default || {};
  const hwProfile = taxonomy.hardware_profiles?.[hwId] || {};
  const dockerMeta = computeDockerMeta(recipe, variant, hwProfile, hwId);
  const env = result.env || {};

  const base = {
    hardware: hwId,
    strategy,
    variant: variantKey,
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
  if (result.deployType === "kv_store_lb") {
    // Instance env lives on result.vllm (not result.env). `instances` is how
    // many engines sit behind the router; `node_count` is nodes PER INSTANCE
    // (worker command present when an instance spans >1 node). Mooncake
    // COMPOSES with a serving strategy: `strategy` names the kv deployment
    // (the alternatives key) and `serving_strategy` the parallelism layout
    // each instance actually runs.
    const vllmDocker = result.vllm?.command && result.vllm?.argv
      ? dockerize(result.vllm.command, result.vllm.argv, result.vllm.env || {}, dockerMeta)
      : {};
    // Worker gets the same docker wrap as the head — mirrors the multi_node
    // branch's head/worker docker pair.
    const vllmWorkerDocker = result.vllm?.workerCommand && result.vllm?.workerArgv
      ? dockerize(result.vllm.workerCommand, result.vllm.workerArgv, result.vllm.env || {}, dockerMeta)
      : {};
    return {
      ...base,
      strategy: kvOffload || base.strategy,
      serving_strategy: result.servingStrategy || base.strategy,
      // The kv deployment's own vLLM floor (MooncakeStoreConnector ships in
      // 0.21.0+) — consumers should take max(recipe, variant, this).
      kv_min_vllm_version: (kvOffload && strategies[kvOffload]?.min_vllm_version) || null,
      env: result.vllm?.env || {},
      node_count: result.nodeCount,
      instances: result.instances,
      vllm_command: result.vllm?.command,
      vllm_argv: result.vllm?.argv,
      vllm_worker_command: result.vllm?.workerCommand || null,
      vllm_worker_argv: result.vllm?.workerArgv || null,
      vllm_worker_docker_command: vllmWorkerDocker.docker_command ?? null,
      vllm_worker_docker_argv: vllmWorkerDocker.docker_argv ?? null,
      // Extra pip dep for the instances (MooncakeStoreConnector imports the
      // mooncake package) — same field the UI's "Requires:" hint consumes.
      vllm_install: result.vllm?.install || null,
      vllm_docker_command: vllmDocker.docker_command,
      vllm_docker_argv: vllmDocker.docker_argv,
      master: result.master,
      store: result.store || null,
      router: result.router,
      mooncake_config: result.mooncakeConfig,
      mooncake_config_note: result.mooncakeConfigNote || "",
    };
  }
  return {
    ...base,
    command: result.command,
    argv: result.argv,
    ...dockerize(result.command, result.argv, env, dockerMeta),
    // Companion processes (feature `companion:` or the active kv_offload
    // option's, e.g. LMCache's `lmcache server`) — run alongside `vllm serve`
    // on the same node, in their own terminal.
    ...(result.companions ? { companions: result.companions } : {}),
  };
}

// Build the canonical "recommended" rendering plus per-strategy alternatives
// for one variant of a recipe ON A SPECIFIC HARDWARE. Inlines the strategy/
// hardware spec used by the recommended choice so agents don't need to fetch
// /strategies/<id>.json or /taxonomy.json. Returns
// { recommended, alternatives: { <strategy>: <rendered> } } where the caller
// writes each alternative to its own file and replaces it with a path link.
// Null for omni recipes or unknown variants.
function buildVariantRendering(recipe, variantKey, hwId, strategies, taxonomy) {
  let variant = recipe.variants?.[variantKey];
  if (!variant) return null;
  if ((recipe.meta?.tasks || []).includes("omni")) return null;

  const hwProfile = taxonomy.hardware_profiles?.[hwId] || {};
  const scalable = isHardwareScalable(hwProfile);
  // Non-scalable hardware (single-GPU workstation, e.g. DGX Station) can't
  // shard an oversized variant — substitute the largest variant that fits, and
  // skip multi-node strategies. Mirrors the command builder's UI behavior.
  if (!scalable && !fitsSingleNode(hwProfile, variant)) {
    const fitting = pickFittingVariant(recipe, hwProfile, hwId);
    if (!fitting) return null;
    variantKey = fitting;
    variant = recipe.variants[fitting];
  }
  // kv_store_lb strategies apply to every non-omni recipe by default — no
  // compatible_strategies opt-in (mirrors the UI's KV Offload row; omni
  // recipes already returned null above). Deduped in case a recipe still
  // names them explicitly.
  const kvStoreIds = Object.keys(strategies).filter(
    (s) => strategies[s].deploy_type === "kv_store_lb"
  );
  const compatible = [...new Set([
    ...(recipe.compatible_strategies || []),
    ...kvStoreIds,
  ])].filter((s) => {
    // Mirrors the UI's KV Offload gating: Mooncake renders on scalable
    // NVIDIA/AMD hardware (the transfer engine has no CPU/TPU build) unless
    // the recipe opts that GPU out (`unsupported`) under
    // kv_cache_strategy_hardware.
    if (strategies[s]?.deploy_type === "kv_store_lb") {
      return scalable && isKvStoreBrandSupported(hwProfile)
        && recipe.kv_cache_strategy_hardware?.[s]?.[hwId] !== "unsupported";
    }
    // Serving-strategy per-GPU opt-out (mirrors kv_cache_strategy_hardware),
    // fail-open: a recipe marks a (strategy, GPU) pair `unsupported` under
    // `strategy_hardware` when that layout isn't usable on that GPU.
    if (recipe.strategy_hardware?.[s]?.[hwId] === "unsupported") return false;
    return scalable || (!s.startsWith("multi_node_") && s !== "pd_cluster");
  });
  const supportsMultiNode = scalable && compatible.some((s) => s.startsWith("multi_node_"));
  const recommendedNodeCount = !fitsSingleNode(hwProfile, variant) && supportsMultiNode ? 2 : 1;
  let recommendedStrategy = recommendStrategy(recipe, hwProfile, recommendedNodeCount);
  // Never recommend a strategy the recipe opted this GPU out of via
  // strategy_hardware — fall back to the first compatible serving strategy.
  if (recipe.strategy_hardware?.[recommendedStrategy]?.[hwId] === "unsupported") {
    recommendedStrategy = compatible.find(
      (s) => strategies[s]?.deploy_type !== "kv_store_lb" && s !== "pd_cluster"
    ) || compatible.find((s) => strategies[s]?.deploy_type !== "kv_store_lb")
      || recommendedStrategy;
  }
  const recommendedFeatures = defaultFeaturesFor(recipe, hwId);

  // PD's node-count is independent of nodeCount — it lives in pdNodes per role.
  const recommendedPdNodes = recommendedStrategy === "pd_cluster"
    ? pickPdNodes(hwProfile, variant)
    : null;
  if (recommendedPdNodes === "skip") return null;

  const recommended = renderCommand(
    recipe, variantKey, recommendedStrategy, hwId, recommendedNodeCount,
    recommendedFeatures, strategies, taxonomy, recommendedPdNodes
  );
  if (!recommended) return null;

  recommended.strategy_spec = strategies[recommendedStrategy];
  recommended.hardware_profile = hwProfile;

  // Serving strategy for a Mooncake composition at a given nodes-per-instance
  // count: the recommendation for that node count, skipping PD (PD × Mooncake
  // is a MultiConnector composition the interactive builder renders; the
  // API's pd_cluster entry stays the plain Nixl deployment) and falling back
  // to the first compatible serving strategy that fits.
  const kvServingFor = (nc) => {
    const ok = (id) => {
      const d = strategies[id]?.deploy_type;
      if (!strategies[id] || d === "pd_cluster" || d === "kv_store_lb") return false;
      return nc > 1 ? d === "multi_node" : d !== "multi_node";
    };
    const rec = recommendStrategy(recipe, hwProfile, nc);
    if (ok(rec)) return rec;
    return (recipe.compatible_strategies || []).find(ok) || null;
  };

  const alternatives = {};
  for (const s of compatible) {
    if (s === recommendedStrategy) continue;
    let nc;
    let pdNodes = null;
    let servingStrategy = s;
    let kvOffload = null;
    if (s === "pd_cluster") {
      pdNodes = pickPdNodes(hwProfile, variant);
      if (pdNodes === "skip") continue;
      nc = 1;
    } else if (strategies[s]?.deploy_type === "kv_store_lb") {
      // nc = nodes PER INSTANCE (the instance count defaults inside
      // resolveCommand). Single-node instances unless the variant needs
      // multi-node sharding to fit at all. Mooncake composes with a serving
      // strategy — the kv id rides in via kvOffload, never as the strategy.
      nc = fitsSingleNode(hwProfile, variant) ? 1 : 2;
      servingStrategy = kvServingFor(nc);
      if (!servingStrategy && nc === 2) {
        // No multi_node_* strategy to shard with — fall back to single-node
        // instances rather than dropping the kv rendering: VRAM is a display
        // hint in this repo, never a hard gate.
        nc = 1;
        servingStrategy = kvServingFor(1);
      }
      if (!servingStrategy) continue;
      kvOffload = s;
    } else {
      nc = s.startsWith("multi_node_") ? 2 : 1;
    }
    const feats = defaultFeaturesFor(recipe, hwId);
    const rendered = renderCommand(recipe, variantKey, servingStrategy, hwId, nc, feats, strategies, taxonomy, pdNodes, kvOffload);
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

// ── Platforms ── (third-party self-host targets; flat catalog)
const platformsFile = path.join(ROOT, "platforms.yaml");
let platformsCount = 0;
if (fs.existsSync(platformsFile)) {
  const platforms = normalizeDates(readYaml(platformsFile));
  writeJson("platforms.json", platforms);
  platformsCount = Array.isArray(platforms?.platforms) ? platforms.platforms.length : 0;
}

// ── Strategies ──
const strategiesDir = path.join(ROOT, "strategies");
const strategies = {};
for (const file of fs.readdirSync(strategiesDir).filter((f) => f.endsWith(".yaml"))) {
  const s = normalizeDates(readYaml(path.join(strategiesDir, file)));
  strategies[s.name] = s;
  writeJson(`strategies/${s.name}.json`, s);
}
writeJson("strategies.json", strategies);
const servingStrategyCount = Object.keys(strategies).length;

// ── KV-store deployments ── (kv_store/*.yaml, deploy_type: kv_store_lb)
// They power the KV Offload row, not the Strategy row, so they publish under
// /kv_store/ and stay out of /strategies.json — but merge into the same
// in-memory map since renderCommand looks every deployment spec up by id.
const kvStoreDir = path.join(ROOT, "kv_store");
const kvStoreDeployments = {};
for (const file of fs.readdirSync(kvStoreDir).filter((f) => f.endsWith(".yaml"))) {
  const s = normalizeDates(readYaml(path.join(kvStoreDir, file)));
  kvStoreDeployments[s.name] = s;
  strategies[s.name] = s;
  writeJson(`kv_store/${s.name}.json`, s);
}
writeJson("kv_store.json", kvStoreDeployments);

// ── Recipes ── (walks models/<hf_org>/<hf_repo>.yaml)
const modelsDir = path.join(ROOT, "models");

// Pre-scan: collect every recipe's hf_id so variant-promotion can detect
// collisions with standalone YAMLs (e.g. inclusionAI/Ring-1T-FP8.yaml exists
// as its own recipe, so a sibling recipe must not also promote a "Ring-1T-FP8"
// variant).
const allRecipeHfIds = new Set();
for (const file of findYamlFiles(modelsDir)) {
  const rel = path.relative(modelsDir, file);
  const parts = rel.split(path.sep);
  if (parts.length >= 2) {
    const org = parts[0];
    const repo = parts[parts.length - 1].replace(/\.(yaml|yml)$/, "");
    allRecipeHfIds.add(`${org}/${repo}`);
  }
}

// Write the public recipe JSON (parent or promoted) with HEAD_KEYS ordering.
function writeRecipeJson(hfId, payload) {
  const HEAD_KEYS = ["hf_id", "meta", "recommended_command"];
  const ordered = {};
  for (const k of HEAD_KEYS) if (k in payload) ordered[k] = payload[k];
  for (const [k, v] of Object.entries(payload)) if (!HEAD_KEYS.includes(k)) ordered[k] = v;
  writeJson(`${hfId}.json`, ordered);
  return ordered;
}

// Render one variant's full hardware × strategy matrix and write all the
// per-resource JSON files. Returns the parent's `recommended_command` object
// (the default-hardware rendering with `alternatives` path map + `by_hardware`
// path map) or null if the variant can't be rendered on its default hardware.
//
// `altBaseHfId` is the base directory for written files — the parent recipe's
// hf_id, or a promoted variant's hf_id (e.g. zai-org/GLM-5.1-FP8). Output:
//
//   /<base>.json                          (parent — written by caller)
//   /<base>/strategies/<s>.json           (default-hw alternatives — legacy path)
//   /<base>/hw/<hw>.json                  (per-hw recommended + alternatives index)
//   /<base>/hw/<hw>/strategies/<s>.json   (per-(hw, strategy) — non-default hw only)
//
// The default-hw entry in `by_hardware` re-uses the legacy `/strategies/`
// paths to avoid duplicating identical files; non-default hw entries each get
// their own `/hw/<hw>/strategies/` subtree.
function renderAndWriteVariant(recipe, variantKey, altBaseHfId, strategies, taxonomy) {
  const variant = recipe.variants?.[variantKey];
  if (!variant) return null;
  if ((recipe.meta?.tasks || []).includes("omni")) return null;

  // Hardware/strategy compatibility can shrink between recipe revisions.
  // Clear this variant's generated subtree so removed targets do not survive
  // as stale, directly-addressable JSON files from an earlier build.
  fs.rmSync(path.join(PUBLIC, altBaseHfId), { recursive: true, force: true });

  const hwProfiles = taxonomy.hardware_profiles || {};
  const defaultHw = pickDefaultHardware(hwProfiles, variant, recipe);
  // Mirror the UI rule: `restricted` profiles (TPU, etc.) only surface for
  // recipes that explicitly opt in via `meta.hardware.<id>`. Otherwise we'd
  // emit speculative TPU commands for every NVIDIA recipe.
  const declaredHw = recipe.meta?.hardware || {};
  const compatibleHw = listCompatibleHardware(hwProfiles, variant, recipe)
    .filter((id) => !hwProfiles[id]?.restricted || id in declaredHw);
  // Ensure default is first; de-dupe.
  const hwIds = Array.from(new Set([defaultHw, ...compatibleHw]));

  // Render every (variant × hardware) combo up front.
  const renderedByHw = {};
  for (const hwId of hwIds) {
    const built = buildVariantRendering(recipe, variantKey, hwId, strategies, taxonomy);
    if (built) renderedByHw[hwId] = built;
  }
  if (!renderedByHw[defaultHw]) return null;

  const byHardware = {};
  for (const [hwId, { recommended, alternatives }] of Object.entries(renderedByHw)) {
    const isDefault = hwId === defaultHw;
    const stratDir = isDefault
      ? `${altBaseHfId}/strategies`
      : `${altBaseHfId}/hw/${hwId}/strategies`;
    const altLinks = {};
    for (const [s, rendered] of Object.entries(alternatives)) {
      const altPath = `${stratDir}/${s}.json`;
      writeJson(altPath, rendered);
      altLinks[s] = `/${altPath}`;
    }
    if (Object.keys(altLinks).length) recommended.alternatives = altLinks;

    const hwPath = `${altBaseHfId}/hw/${hwId}.json`;
    writeJson(hwPath, recommended);
    byHardware[hwId] = `/${hwPath}`;
  }

  // Parent's recommended_command = default-hw rendering + by_hardware index.
  // Shallow clone so the on-disk per-hw default file stays clean of the index.
  const defaultRecommended = { ...renderedByHw[defaultHw].recommended, by_hardware: byHardware };
  return defaultRecommended;
}

const recipes = [];
const promotedIndexEntries = [];
let promotedCount = 0;
let collisionCount = 0;

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
  // Pre-render the canonical deploy command (default variant) so agents don't
  // have to reimplement command-synthesis. Mirrors the website's default
  // selections.
  const parentHfId = `${hfOrg}/${hfRepo}`;
  const defaultRecommended = renderAndWriteVariant(r, "default", parentHfId, strategies, taxonomy);
  if (defaultRecommended) r.recommended_command = defaultRecommended;

  // Promote each non-default variant whose `model_id` points at a distinct HF
  // repo to its own top-level JSON endpoint, mirroring the HF URL convention.
  // Renderings for promoted variants are gathered here and written after the
  // parent JSON so we can store `json:` pointers in parent.variants.<v>.
  const promotedRenderings = [];  // { variantKey, variantHfId, recommended }
  for (const [variantKey, variantCfg] of Object.entries(r.variants || {})) {
    if (variantKey === "default") continue;
    const variantModelId = variantCfg?.model_id;
    if (!variantModelId || typeof variantModelId !== "string" || !variantModelId.includes("/")) continue;
    if (allRecipeHfIds.has(variantModelId)) {
      // A standalone recipe at models/<variantModelId>.yaml exists — let it win.
      console.warn(`  ⚠ skipping variant promotion: ${variantModelId} (variant of ${parentHfId}) collides with standalone recipe`);
      collisionCount++;
      continue;
    }
    const variantRecommended = renderAndWriteVariant(r, variantKey, variantModelId, strategies, taxonomy);
    if (!variantRecommended) continue;
    promotedRenderings.push({ variantKey, variantModelId, recommended: variantRecommended });
  }

  // Annotate parent variants with `json:` pointers so consumers can navigate.
  if (r.variants) {
    for (const { variantKey, variantModelId } of promotedRenderings) {
      r.variants[variantKey] = { ...r.variants[variantKey], json: `/${variantModelId}.json` };
    }
  }

  // strategy_overrides, compatible_strategies, default_strategy are synthesis
  // inputs whose effects are already baked into recommended_command + the
  // per-strategy alternative files (whose keys are the compatible_strategies
  // set; the recommended one comes from default_strategy). Drop AFTER all
  // renderings (parent + promoted) have run — buildVariantRendering reads
  // them. The YAML on GitHub is the source of truth for anyone re-synthesizing.
  delete r.strategy_overrides;
  delete r.compatible_strategies;
  delete r.default_strategy;
  // JSON at /<org>/<repo>.json — mirrors HF URL scheme.
  const out = writeRecipeJson(parentHfId, r);
  recipes.push(out);

  // Now write the promoted-variant JSONs. Each is a self-contained recipe
  // shape with hf_id / model.model_id rewritten and `variants.default` set to
  // the variant's config (model_id stripped — it's at top-level now). The
  // sibling-variant info is lost on purpose; consumers can follow
  // `meta.derived_from` back to the parent for the full family view.
  for (const { variantKey, variantModelId, recommended } of promotedRenderings) {
    const variantCfg = r.variants?.[variantKey] || {};
    const { model_id: _vMid, json: _vJson, ...variantCore } = variantCfg;
    const promoted = {
      ...r,  // share meta/features/guide/etc. with the parent
      hf_id: variantModelId,
      meta: { ...r.meta, derived_from: parentHfId, variant: variantKey },
      model: { ...r.model, model_id: variantModelId },
      variants: { default: variantCore },
      recommended_command: recommended,
    };
    writeRecipeJson(variantModelId, promoted);
    promotedIndexEntries.push({
      hf_id: variantModelId,
      title: promoted.meta?.title,
      provider: promoted.meta?.provider,
      derived_from: parentHfId,
    });
    promotedCount++;
  }
}

// /models.json — slim discovery index (~5 KB). For agents that want to
// enumerate recipes and follow links; per-recipe data lives at
// `/<hf_id>.json` (the `json` pointer below). Promoted variants (e.g.
// zai-org/GLM-5.1-FP8) appear alongside parents so consumers can discover
// them directly.
const index = [
  ...recipes.map((r) => ({
    hf_id: r.hf_id,
    title: r.meta.title,
    provider: r.meta.provider,
    url: `/${r.hf_id}`,
    json: `/${r.hf_id}.json`,
  })),
  ...promotedIndexEntries.map((e) => ({
    hf_id: e.hf_id,
    title: e.title,
    provider: e.provider,
    url: `/${e.derived_from}`,  // site route — variant lives under the parent page
    json: `/${e.hf_id}.json`,
    derived_from: e.derived_from,
  })),
];
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
const hwIndexedCount = recipes.reduce(
  (n, r) => n + Object.keys(r.recommended_command?.by_hardware || {}).length, 0
);
console.log(
  `✓ JSON API: ${recipes.length} models (${rcCount} with recommended_command, ${altCount} default-hw alternatives, ${hwIndexedCount} per-hw renderings), ${promotedCount} promoted variants, ${servingStrategyCount} strategies, ${Object.keys(kvStoreDeployments).length} kv-store deployments, ${platformsCount} platforms` +
  (collisionCount ? ` (${collisionCount} variant collision${collisionCount > 1 ? "s" : ""} skipped)` : "")
);
console.log(`  /models.json`);
console.log(`  /{hf_id}.json                                (e.g. /moonshotai/Kimi-K2.5.json)`);
console.log(`  /{hf_id}/strategies/{strategy}.json          (default-hw alternatives)`);
console.log(`  /{hf_id}/hw/{hw}.json                        (per-hardware rendering)`);
console.log(`  /{hf_id}/hw/{hw}/strategies/{strategy}.json  (per-(hw, strategy) alternatives)`);
console.log(`  /{variant_hf_id}.json                        (promoted variants, e.g. /zai-org/GLM-5.1-FP8.json)`);
console.log(`  /strategies.json`);
console.log(`  /kv_store.json`);
console.log(`  /kv_store/{id}.json                          (per-deployment spec)`);
console.log(`  /taxonomy.json`);
console.log(`  /platforms.json`);
