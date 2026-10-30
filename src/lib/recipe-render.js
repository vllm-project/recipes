// Render a recipe selection into the public JSON API shape. Shared by the
// build (scripts/build-recipes-api.mjs writes the static API from it) and the
// on-demand `/<hf_id>.md` / `.json?<query>` routes, so a rendering means the
// same thing wherever it is read.

import { BASELINE, effectivePreset, presetPoints, presetSelect, resolvePresetKey, withPreset } from "./recipe-presets.js";
import { ADVANCED_OPTIONS } from "./advanced-options.js";
import { SELECTION_PARAMS } from "./selection-params.js";

export { SELECTION_PARAMS };
import { resolveOmniTasks, resolveOmniTaskForHardware } from "./omni-tasks.js";
import {
  MAX_NODES,
  pickDefaultHardware,
  listCompatibleHardware,
  isPrecisionCompatible,
  isHardwareSupported,
  isVariantHardwareSupported,
  needsMultiNode,
  multiNodeUnavailable,
  pdPoolModes,
  resolveModeKey,
  resolveFrontend,
  resolveOmniCommand,
  isKvOffloadAllowedForStrategy,
  isKvOffloadSupportedForRecipe,
  isKvOffloadBrandSupported,
  recommendStrategy,
  fitsSingleNode,
  variantVramMinimumGb,
  isHardwareScalable,
  isKvStoreBrandSupported,
  strategyAllowsKvOffload,
  pickFittingVariant,
  pdFitsSingleNode,
  pdRoleNodes,
  pdRoleParallelism,
  resolveCommand,
  computeDockerMeta,
  buildDockerRun,
  buildDockerArgv,
  nodesForStrategy,
  isStrategyReachable,
  isStrategySupportedOnHardware,
  effectiveCompatibleStrategies,
} from "./command-synthesis.js";

// Resolve the default NVIDIA Docker image for a recipe, picking the base CUDA
// tag from a `{cu129, cu130}` map if present. Mirrors `computeDockerMeta` in
// CommandBuilder.jsx but only for the NVIDIA path (the canonical default).
export function resolveDockerImage(recipe, baseCuda) {
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
export function synthesizeInstall(recipe) {
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
// (hardware_opt_in_features[hw]). A variant can force an otherwise opt-in
// feature on by declaring `default_modes.<feature>` — e.g. a fused DSpark
// checkpoint should render with its DSpark config in the recommended command.
export function defaultFeaturesFor(recipe, hwId, variantKey = "default") {
  const optIn = new Set(recipe.opt_in_features || []);
  for (const f of recipe.hardware_opt_in_features?.[hwId] || []) optIn.add(f);
  const forced = new Set(Object.keys(recipe.variants?.[variantKey]?.default_modes || {}));
  return Object.keys(recipe.features || {}).filter((f) => forced.has(f) || !optIn.has(f));
}

// Wrap a rendered (command, argv) pair in `docker run`. Returns
// { docker_command, docker_argv } so each form has its docker counterpart.
// `docker` is a PD KV connector's container needs ({ host_network, flags }).
export function dockerize(command, argv, env, dockerMeta, port = null, docker = null) {
  const hostNetwork = !!docker?.host_network;
  const extraFlags = docker?.flags || [];
  return {
    docker_command: buildDockerRun({
      command, env, image: dockerMeta.image, gpuFlags: dockerMeta.gpuFlags, port,
      isNpu: dockerMeta.isNpu, cmdPrefix: dockerMeta.cmdPrefix, hostNetwork, extraFlags,
    }),
    docker_argv: buildDockerArgv({ argv, env, meta: dockerMeta, port, hostNetwork, extraFlags }),
  };
}

// Pick the per-role node count for a `pd_cluster` rendering.
//   null   → co-located on a single node (variant fits the 2× PD VRAM budget)
//   {prefill,decode} → one full node per role × N: ceil(variant.vram_minimum_gb /
//                       hwProfile.vram_gb), raised to the recipe's per-role `nodes`
//   "skip" → would need >4 nodes per role; don't emit a fantasy cluster
// Each role is then grown to clear its parallelism's `strategy_min_gpus` floor,
// matching what the builder renders.
export function pickPdNodes(hwProfile, variant, baseRecipe, strategies, hwId, presetPick = undefined, pointPick = undefined, variantKey = undefined) {
  // The pd_cluster preset this rendering bakes in may own the pools' topology.
  const presetKey = resolvePresetKey(baseRecipe, hwId, "pd_cluster", presetPick, variantKey);
  const recipe = withPreset(baseRecipe, presetKey, hwId, "pd_cluster", { variantKey, point: pointPick });
  // A preset that sizes a pool explicitly gets dedicated pools even where two
  // copies would fit one node, as the builder renders it.
  const eff = presetKey !== BASELINE && baseRecipe.presets?.[presetKey] ? effectivePreset(baseRecipe.presets[presetKey], pointPick) : null;
  const presetSizesPools = ["prefill", "decode"].some((r) => eff?.roles?.[r]?.nodes != null);
  if (!presetSizesPools && pdFitsSingleNode(hwProfile, variant, hwId)) return null;
  const nodeVram = typeof hwProfile?.vram_gb === "number" ? hwProfile.vram_gb : 0;
  const modelVram = variantVramMinimumGb(variant, hwId);
  if (nodeVram <= 0 || modelVram <= 0) return null;
  const nodesPerRole = Math.ceil(modelVram / nodeVram);
  if (nodesPerRole > 4) return "skip";
  const floored = (role) => {
    const par = pdRoleParallelism(recipe, strategies?.pd_cluster, role, hwProfile, hwId);
    const nodes = Math.max(nodesPerRole, pdRoleNodes(recipe, role, hwProfile, hwId) ?? 0);
    return Math.max(nodes, nodesForStrategy(recipe, par, hwProfile.gpu_count, hwId));
  };
  return { prefill: floored("prefill"), decode: floored("decode") };
}

// Render one (recipe × variant × strategy × hardware × nodeCount) into the
// public JSON shape — handles all three deploy_types (single_node, multi_node,
// pd_cluster), snake_cases the keys, and attaches docker_run wrappers
// alongside the pip-mode command/argv. The `features` argument controls
// command synthesis but is intentionally NOT echoed back in the output —
// agents read the actual args from `command`/`argv`.
//
// `opts` carries the rest of a builder selection (resolveSelection reads it
// from a page URL): { modes, advancedArgs, kvInstances, frontend, pdRouter,
// pdTransport, dcp, pdConnector, linked }. The preset's `select` fills these
// in, except for the params in `linked` — the ones the URL set explicitly,
// which stay as linked, as the builder treats them on first render.
export function renderCommand(recipe, variantKey, strategy, hwId, nodeCount, features, strategies, taxonomy, pdNodes = null, kvOffload = null, presetPick = undefined, pointPick = undefined, strict = false, opts = {}) {
  // The preset in effect (the default here unless one is picked) implies
  // builder choices — frontend, PD router, features, DCP, KV offload — which
  // the rendering applies the way the builder does.
  const sel = presetSelect(recipe, resolvePresetKey(recipe, hwId, strategy, presetPick, variantKey), pointPick) || {};
  const linked = opts.linked || new Set();
  const pick = (param, selValue, optValue) => (linked.has(param) ? optValue : (selValue ?? optValue));
  let feats = [...features];
  const modes = { ...(opts.modes || {}) };
  if (!linked.has("features")) {
    for (const [f, v] of Object.entries(sel.features || {})) {
      if (!recipe.features?.[f]) continue;
      if (v === false) feats = feats.filter((x) => x !== f);
      else {
        if (!feats.includes(f)) feats.push(f);
        if (typeof v === "string" && !linked.has("fmode")) modes[f] = v;
      }
    }
  }
  const selKv = !linked.has("kv_offload") && sel.kv_offload && sel.kv_offload !== "off" ? sel.kv_offload : null;
  // A multi-node preset's node count, as the builder applies it.
  if (Number.isInteger(sel.nodes) && !linked.has("nodes") && strategies[strategy]?.deploy_type === "multi_node") nodeCount = sel.nodes;
  let result;
  try {
    result = resolveCommand(
      recipe, variantKey, strategy, hwId, feats, strategies, taxonomy, opts.advancedArgs || [], nodeCount, pdNodes,
      modes, kvOffload ?? selKv, opts.kvInstances ?? null, pick("frontend", sel.frontend, opts.frontend),
      pick("pd_router", sel.pd_router, opts.pdRouter) ?? null, opts.pdTransport ?? "http",
      pick("dcp", sel.dcp, opts.dcp) ?? null, presetPick, pointPick,
      pick("pd_connector", sel.pd_connector, opts.pdConnector) ?? null
    );
  } catch (e) {
    // A preset is authored config with nothing to fall back to: fail the build
    // (and the source repo's PR check) rather than drop its rendering.
    if (strict) throw new Error(`Invalid preset ${presetPick}${pointPick ? ` at point ${pointPick}` : ""} in ${recipe.hf_id} (${hwId} / ${strategy}): ${e.message}`);
    console.warn(`  ⚠ command synthesis failed for ${recipe.hf_id} / ${variantKey} / ${strategy}: ${e.message}`);
    return null;
  }
  if (!result) return null;
  // A preset that names its KV transfer connector must get it.
  if (strict && sel.pd_connector && result.kvConnector?.unavailableReason) {
    throw new Error(`Invalid preset ${presetPick}${pointPick ? ` at point ${pointPick}` : ""} in ${recipe.hf_id} (${hwId} / ${strategy}): ${result.kvConnector.unavailableReason}`);
  }

  const baseVariant = recipe.variants?.[variantKey] || recipe.variants?.default || {};
  const hwProfile = taxonomy.hardware_profiles?.[hwId] || {};
  // A preset's engine image pins this GPU's docker command.
  const pinned = result.preset?.image?.vllm;
  const variant = pinned
    ? { ...baseVariant, hardware_overrides: { ...(baseVariant.hardware_overrides || {}), [hwId]: { ...(baseVariant.hardware_overrides?.[hwId] || {}), docker_image: pinned } } }
    : baseVariant;
  const dockerMeta = computeDockerMeta(recipe, variant, hwProfile, hwId);
  const env = result.env || {};

  const base = {
    hardware: hwId,
    strategy,
    variant: variantKey,
    node_count: nodeCount,
    deploy_type: result.deployType,
    // The tuned preset baked into this rendering (its default for this
    // hardware + strategy); absent when the baseline renders.
    ...(result.preset && result.preset.selected !== "baseline" ? { preset: result.preset.selected } : {}),
    ...(result.preset?.point?.selected && result.preset.selected !== "baseline" ? { preset_point: result.preset.point.selected } : {}),
    env,
    docker_image: dockerMeta.image,
  };
  if (result.deployType === "multi_node") {
    const head = dockerize(result.headCommand, result.headArgv, env, dockerMeta);
    const worker = dockerize(result.workerCommand, result.workerArgv, env, dockerMeta);
    // `worker_*` (singular) is rank 1; the plural arrays hold EVERY follower
    // rank (1..node_count-1) — a 4-node cluster needs three different worker
    // commands, so an agent must not replay rank 1 on every node.
    const workerCommands = result.workerCommands || [result.workerCommand];
    const workerArgvs = result.workerArgvs || [result.workerArgv];
    const workerDockers = workerCommands.map((cmd, i) =>
      dockerize(cmd, workerArgvs[i], env, dockerMeta)
    );
    return {
      ...base,
      head_command: result.headCommand,
      worker_command: result.workerCommand,
      worker_commands: workerCommands,
      head_argv: result.headArgv,
      worker_argv: result.workerArgv,
      worker_argvs: workerArgvs,
      head_docker_command: head.docker_command,
      worker_docker_command: worker.docker_command,
      worker_docker_commands: workerDockers.map((w) => w.docker_command),
      head_docker_argv: head.docker_argv,
      worker_docker_argv: worker.docker_argv,
      worker_docker_argvs: workerDockers.map((w) => w.docker_argv),
    };
  }
  if (result.deployType === "pd_cluster") {
    // Prefill exposes :8001, decode exposes :8002 (matches the router endpoints
    // emitted by command-synthesis).
    const prefill = { ...result.prefill };
    const decode = { ...result.decode };
    const docker = result.kvConnector?.docker;
    if (prefill.command && prefill.argv) {
      Object.assign(prefill, dockerize(prefill.command, prefill.argv, prefill.env, dockerMeta, 8001, docker));
    }
    if (decode.command && decode.argv) {
      Object.assign(decode, dockerize(decode.command, decode.argv, decode.env, dockerMeta, 8002, docker));
    }
    return {
      ...base,
      pd_connector: result.kvConnector?.selected ?? null,
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
    // Per-rank follower commands for instances spanning >1 node (same
    // singular-is-rank-1 / plural-is-every-rank convention as multi_node).
    const vllmWorkerCommands = result.vllm?.workerCommands || null;
    const vllmWorkerArgvs = result.vllm?.workerArgvs || null;
    const vllmWorkerDockers = vllmWorkerCommands
      ? vllmWorkerCommands.map((cmd, i) =>
          dockerize(cmd, vllmWorkerArgvs?.[i], result.vllm.env || {}, dockerMeta))
      : null;
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
      vllm_worker_commands: vllmWorkerCommands,
      vllm_worker_argv: result.vllm?.workerArgv || null,
      vllm_worker_argvs: vllmWorkerArgvs,
      vllm_worker_docker_command: vllmWorkerDocker.docker_command ?? null,
      vllm_worker_docker_argv: vllmWorkerDocker.docker_argv ?? null,
      vllm_worker_docker_commands: vllmWorkerDockers?.map((w) => w.docker_command ?? null) ?? null,
      vllm_worker_docker_argvs: vllmWorkerDockers?.map((w) => w.docker_argv ?? null) ?? null,
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
export function buildVariantRendering(recipe, variantKey, hwId, strategies, taxonomy) {
  let variant = recipe.variants?.[variantKey];
  if (!variant) return null;
  if ((recipe.meta?.tasks || []).includes("omni")) return null;

  const hwProfile = taxonomy.hardware_profiles?.[hwId] || {};
  const scalable = isHardwareScalable(hwProfile);
  // Non-scalable hardware (single-GPU workstation, e.g. DGX Station) can't
  // shard an oversized variant — substitute the largest variant that fits, and
  // skip multi-node strategies. Mirrors the command builder's UI behavior.
  if (!scalable && !fitsSingleNode(hwProfile, variant, hwId)) {
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
  // Nodes to render a strategy at: its usual count, grown to clear any
  // `strategy_min_gpus` floor. Same helper the builder's Nodes pills use.
  const nodesFor = (s) => Math.max(
    strategies[s]?.deploy_type === "multi_node" ? 2 : 1,
    nodesForStrategy(recipe, s, hwProfile.gpu_count, hwId),
  );
  const compatible = [...new Set([
    ...effectiveCompatibleStrategies(recipe),
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
    // Serving-strategy per-GPU gate: compatible_strategies membership is the
    // default (listed = supported everywhere, unlisted = nowhere), and
    // `strategy_hardware` overrides per hardware in either direction — maps
    // take the variant-`tp` keyspace (exact GPU id > generation > brand >
    // `default`).
    if (!isStrategySupportedOnHardware(recipe, s, hwProfile, hwId)) return false;
    // Same GPU floor the builder applies (`strategy_min_gpus`).
    if (!isStrategyReachable(recipe, s, strategies[s]?.deploy_type, hwProfile.gpu_count, hwId)) return false;
    return scalable || (!s.startsWith("multi_node_") && s !== "pd_cluster");
  });
  const supportsMultiNode = scalable && compatible.some((s) => s.startsWith("multi_node_"));
  const baseNodeCount = !fitsSingleNode(hwProfile, variant, hwId) && supportsMultiNode ? 2 : 1;
  let recommendedStrategy = recommendStrategy(recipe, hwProfile, baseNodeCount);
  // Never recommend a strategy this GPU can't actually run — opted out via
  // strategy_hardware, or unreachable under its `strategy_min_gpus` floor
  // (e.g. a single-node layout on a GPU whose floor spans several nodes).
  // Both are already excluded from `compatible`, so membership is the test.
  if (!compatible.includes(recommendedStrategy)) {
    recommendedStrategy = compatible.find(
      (s) => strategies[s]?.deploy_type !== "kv_store_lb" && s !== "pd_cluster"
    ) || compatible.find((s) => strategies[s]?.deploy_type !== "kv_store_lb")
      || recommendedStrategy;
  }
  // Grow to whatever the recommended strategy's floor demands, so the headline
  // rendering never sits below its own bar.
  const recommendedNodeCount = Math.max(baseNodeCount, nodesFor(recommendedStrategy));
  const recommendedFeatures = defaultFeaturesFor(recipe, hwId, variantKey);

  // PD's node-count is independent of nodeCount — it lives in pdNodes per role.
  const recommendedPdNodes = recommendedStrategy === "pd_cluster"
    ? pickPdNodes(hwProfile, variant, recipe, strategies, hwId, undefined, undefined, variantKey)
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
      if (!strategyAllowsKvOffload(strategies[id])) return false;
      return nc > 1 ? d === "multi_node" : d !== "multi_node";
    };
    const rec = recommendStrategy(recipe, hwProfile, nc);
    if (ok(rec)) return rec;
    return effectiveCompatibleStrategies(recipe).find(ok) || null;
  };

  const alternatives = {};
  for (const s of compatible) {
    if (s === recommendedStrategy) continue;
    let nc;
    let pdNodes = null;
    let servingStrategy = s;
    let kvOffload = null;
    if (s === "pd_cluster") {
      pdNodes = pickPdNodes(hwProfile, variant, recipe, strategies, hwId, undefined, undefined, variantKey);
      if (pdNodes === "skip") continue;
      nc = 1;
    } else if (strategies[s]?.deploy_type === "kv_store_lb") {
      // nc = nodes PER INSTANCE (the instance count defaults inside
      // resolveCommand). Single-node instances unless the variant needs
      // multi-node sharding to fit at all. Mooncake composes with a serving
      // strategy — the kv id rides in via kvOffload, never as the strategy.
      nc = fitsSingleNode(hwProfile, variant, hwId) ? 1 : 2;
      servingStrategy = kvServingFor(nc);
      if (!servingStrategy && nc === 2) {
        // No multi_node_* strategy to shard with — fall back to single-node
        // instances rather than dropping the kv rendering: VRAM is a display
        // hint in this repo, never a hard gate.
        nc = 1;
        servingStrategy = kvServingFor(1);
      }
      if (!servingStrategy) continue;
      // Each instance still owes its serving strategy's GPU floor (K3 on H100:
      // 4 nodes per instance, not the plain doesn't-fit 2) — growing a
      // multi-node instance keeps that same strategy valid.
      if (strategies[servingStrategy]?.deploy_type === "multi_node") {
        nc = Math.max(nc, nodesForStrategy(recipe, servingStrategy, hwProfile.gpu_count, hwId));
      }
      kvOffload = s;
    } else {
      nc = nodesFor(s);
    }
    const feats = defaultFeaturesFor(recipe, hwId, variantKey);
    const rendered = renderCommand(recipe, variantKey, servingStrategy, hwId, nc, feats, strategies, taxonomy, pdNodes, kvOffload);
    if (rendered) alternatives[s] = rendered;
  }
  return { recommended, alternatives };
}


const parseIntParam = (v) => {
  const n = parseInt(v ?? "", 10);
  return Number.isFinite(n) ? n : null;
};

// Resolve a page URL's query into the selection the command builder opens on
// and render it. Mirrors the builder's initial state — URL param > recipe
// default — without the viewer's stored preferences, which a shared link must
// not depend on. `query` is a URLSearchParams.
//
// Returns { omni, selection, rendering, hwProfile, options } — `rendering` is
// the JSON API shape (renderCommand), or { command, env } for omni recipes;
// null when the selection can't be rendered.
export function resolveSelection(recipe, query, strategies, taxonomy) {
  const get = (k) => query.get(k);
  const linked = new Set(SELECTION_PARAMS.filter((k) => get(k) != null));
  const profiles = taxonomy.hardware_profiles || {};

  let variantKey = recipe.variants?.[get("variant")] ? get("variant") : "default";
  let variant = recipe.variants?.[variantKey] || {};
  const defaultHw = pickDefaultHardware(profiles, variant, recipe);
  const reqHw = get("hardware");
  const reqProfile = profiles[reqHw];
  const hwAllowed = reqProfile
    && (!reqProfile.restricted || reqHw in (recipe.meta?.hardware || {}))
    && isPrecisionCompatible(reqProfile, variant)
    && isHardwareSupported(recipe, reqHw)
    && isVariantHardwareSupported(variant, reqHw, reqProfile);
  const hwId = hwAllowed ? reqHw : defaultHw;
  const hwProfile = profiles[hwId] || {};
  const scalable = isHardwareScalable(hwProfile);
  if (!get("variant") && !scalable && !fitsSingleNode(hwProfile, variant, hwId)) {
    const fitting = pickFittingVariant(recipe, hwProfile, hwId);
    if (fitting) { variantKey = fitting; variant = recipe.variants[fitting]; }
  }

  const declaredHw = recipe.meta?.hardware || {};
  const options = {
    hardware: listCompatibleHardware(profiles, variant, recipe).filter((id) => !profiles[id]?.restricted || id in declaredHw),
    variants: Object.keys(recipe.variants || {}),
    strategies: effectiveCompatibleStrategies(recipe),
    features: Object.keys(recipe.features || {}),
    presets: Object.entries(recipe.presets || {}).filter(([, p]) => p.hardware === hwId).map(([k]) => k),
  };

  const featureModeKeys = Object.keys(recipe.features || {}).filter((k) => recipe.features[k]?.modes);
  const modes = {};
  for (const k of featureModeKeys) modes[k] = resolveModeKey(recipe.features[k], k, variant, variantKey, hwProfile, hwId, undefined);
  for (const pair of (get("fmode") || "").split(",")) {
    const [k, m] = pair.split(":");
    if (k && m && recipe.features?.[k]?.modes?.[m]) modes[k] = m;
  }
  const features = get("features")
    ? get("features").split(",").filter((f) => recipe.features?.[f])
    : defaultFeaturesFor(recipe, hwId, variantKey);
  const frontend = resolveFrontend(recipe, get("frontend") || undefined, taxonomy);

  if ((recipe.meta?.tasks || []).includes("omni")) {
    const tasks = resolveOmniTasks(recipe);
    const task = tasks.find((t) => t.key === get("task")) || tasks.find((t) => t.id === get("task")) || tasks[0] || null;
    const { command, env } = resolveOmniCommand(recipe, variantKey, resolveOmniTaskForHardware(task, hwId), hwProfile, hwId);
    return {
      omni: true,
      selection: { hardware: hwId, variant: variantKey, task: task?.key ?? null },
      rendering: { hardware: hwId, variant: variantKey, task: task?.key ?? null, env, command },
      hwProfile,
      options: { ...options, strategies: [], tasks: tasks.map((t) => t.key) },
    };
  }

  // Nodes (per instance under Mooncake), as the builder's initial count.
  const perNode = hwProfile.gpu_count || 8;
  const supportsMultiNode = options.strategies.some((s) => s.startsWith("multi_node_") || s === "pd_cluster");
  let nodeCount = 1;
  if (supportsMultiNode && scalable) {
    const multi = needsMultiNode(recipe, strategies, hwProfile, variant, hwId);
    const n = parseIntParam(get("nodes"));
    if (n != null) {
      if (n === 1 && multi) nodeCount = 2;
      else if (n > 1 && multiNodeUnavailable(recipe, strategies, hwProfile, variant, hwId)) nodeCount = 1;
      else nodeCount = n >= 1 && n <= MAX_NODES ? n : 1;
    } else nodeCount = multi ? 2 : 1;
  }

  // Serving strategy: the linked one when this node count can run it here,
  // else the recommendation (same fallback as the Strategy row).
  const offered = options.strategies.filter((s) => {
    const d = strategies[s]?.deploy_type;
    if (!d || d === "kv_store_lb") return false;
    if (nodeCount === 1 && d === "multi_node") return false;
    if (nodeCount > 1 && d === "single_node") return false;
    return isStrategyReachable(recipe, s, d, perNode, hwId)
      && isStrategySupportedOnHardware(recipe, s, hwProfile, hwId);
  });
  const recommended = recommendStrategy(recipe, hwProfile, nodeCount);
  const reqStrategy = get("strategy");
  const strategy = offered.includes(reqStrategy) ? reqStrategy
    : offered.includes(recommended) ? recommended
    : (offered[0] || recommended);
  if (scalable && strategy !== "pd_cluster") {
    const needed = nodesForStrategy(recipe, strategy, perNode, hwId);
    const presetNodes = Object.values(recipe.presets || {})
      .filter((p) => p.hardware === hwId)
      .flatMap((p) => [p.select?.nodes, ...presetPoints(p).map((pt) => pt.select?.nodes)])
      .filter((n) => Number.isInteger(n) && n > 2 && n <= MAX_NODES);
    const maxNeeded = Math.max(0, ...options.strategies.map((s) => nodesForStrategy(recipe, s, perNode, hwId)));
    const nodeOptions = new Set([1, 2, ...(maxNeeded > 2 && maxNeeded <= MAX_NODES ? [maxNeeded] : []), ...presetNodes]);
    if (needed > nodeCount && needed <= MAX_NODES) nodeCount = needed;
    else if (nodeCount > 2 && !nodeOptions.has(nodeCount)) nodeCount = Math.min(MAX_NODES, Math.max(2, needed));
  }

  // KV offload: a taxonomy option (Simple, LMCache) or a Mooncake kv_store id.
  const kvReq = get("kv_offload") || "";
  const kvOpt = taxonomy.kv_offload?.[kvReq];
  const kvAllowedByStrategy = strategyAllowsKvOffload(strategies[strategy]);
  let kvOffload = "";
  if (hwProfile.generation !== "xpu" && kvReq) {
    if (kvOpt) {
      if (isKvOffloadAllowedForStrategy(kvOpt, strategy, strategies[strategy])
        && isKvOffloadSupportedForRecipe(kvOpt, kvReq, recipe)
        && isKvOffloadBrandSupported(kvOpt, hwProfile)) kvOffload = kvReq;
    } else if (strategies[kvReq]?.deploy_type === "kv_store_lb" && scalable && kvAllowedByStrategy
      && isKvStoreBrandSupported(hwProfile) && recipe.kv_cache_strategy_hardware?.[kvReq]?.[hwId] !== "unsupported") {
      kvOffload = kvReq;
    }
  }
  const kvInstances = parseIntParam(get("kv_instances"));

  const presetPick = get("preset") || undefined;
  const pointPick = get("point") || undefined;

  // PD pools: per-role nodes and parallelism, defaults read off the recipe
  // with the pd_cluster preset in effect mounted.
  let pdNodes = null;
  if (strategy === "pd_cluster") {
    const pdRecipe = withPreset(recipe, resolvePresetKey(recipe, hwId, "pd_cluster", presetPick ?? null, variantKey), hwId, "pd_cluster", { variantKey, point: pointPick ?? null });
    const nodeVram = hwProfile.vram_gb || 0;
    const modelVram = variantVramMinimumGb(variant, hwId);
    const minNodes = modelVram > 0 && nodeVram > 0 ? Math.max(1, Math.ceil(modelVram / nodeVram)) : 1;
    const poolModes = pdPoolModes(recipe);
    const available = poolModes.filter((m) => isStrategyReachable(recipe, m, "multi_node", perNode, hwId));
    const role = (r) => {
      const defPar = pdRoleParallelism(pdRecipe, strategies.pd_cluster, r, hwProfile, hwId);
      const reqPar = get(`${r}_mode`);
      const par0 = reqPar && poolModes.includes(reqPar) ? reqPar : (poolModes.includes(defPar) ? defPar : poolModes[0]);
      const parallelism = available.includes(par0) ? par0 : (available[0] || "tp");
      const reqNodes = parseIntParam(get(`${r}_nodes`));
      const nodes0 = reqNodes > 0 ? reqNodes : (pdRoleNodes(pdRecipe, r, hwProfile, hwId) ?? minNodes);
      const rank = parseIntParam(get(`${r}_rank`));
      return { nodes: Math.max(nodes0, nodesForStrategy(recipe, parallelism, perNode, hwId)), rank: rank >= 0 ? rank : 0, parallelism };
    };
    pdNodes = { prefill: role("prefill"), decode: role("decode") };
  }

  const advancedById = Object.fromEntries([...ADVANCED_OPTIONS, ...(recipe.advanced_options || [])].map((o) => [o.id, o]));
  const advanced = (get("advanced") || "").split(",").filter((id) => advancedById[id]);
  const pdRouter = ["dynamo", "smg"].includes(get("pd_router")) ? get("pd_router") : "vllm";
  const pdConnector = strategies.pd_cluster?.connectors?.[get("pd_connector")] ? get("pd_connector") : null;
  const dcp = parseIntParam(get("dcp"));

  // Requested values the selection couldn't honor, so a reader of the
  // rendering knows it isn't what they asked for.
  const adjusted = [];
  const note = (param, requested, used, reason) => {
    if (requested != null && requested !== "" && String(requested) !== String(used)) adjusted.push({ param, requested, used, reason });
  };
  note("variant", get("variant"), variantKey, "not a variant of this recipe, or doesn't fit this hardware");
  const runsOnReqHw = reqProfile && Object.keys(recipe.variants || {}).find((k) =>
    isPrecisionCompatible(reqProfile, recipe.variants[k]) && isVariantHardwareSupported(recipe.variants[k], reqHw, reqProfile));
  note("hardware", reqHw, hwId, !reqProfile ? "an unknown hardware id"
    : `not available for variant \`${variantKey}\`${runsOnReqHw && runsOnReqHw !== variantKey && isHardwareSupported(recipe, reqHw) ? ` (variant \`${runsOnReqHw}\` runs on it)` : ""}`);
  note("strategy", reqStrategy, strategy, strategies[reqStrategy]?.deploy_type === "multi_node" && nodeCount === 1
    ? "multi-node — add `nodes=2` (or more)"
    : "not offered for this hardware and node count");
  if (strategies[strategy]?.deploy_type === "multi_node") note("nodes", get("nodes"), nodeCount, "not a node count this hardware offers");
  note("kv_offload", kvReq || null, kvOffload || "off", "not available for this hardware and strategy");

  const rendering = renderCommand(
    recipe, variantKey, strategy, hwId, nodeCount, features, strategies, taxonomy, pdNodes,
    kvOffload || null, presetPick, pointPick, false,
    {
      modes,
      advancedArgs: advanced.flatMap((id) => advancedById[id].args || []),
      kvInstances: { count: kvInstances >= 1 && kvInstances <= 16 ? kvInstances : undefined, current: Math.max(0, parseIntParam(get("kv_instance")) ?? 0) },
      frontend,
      pdRouter,
      pdTransport: recipe.model?.smg_grpc && get("pd_transport") === "grpc" ? "grpc" : "http",
      dcp: dcp >= 1 ? dcp : null,
      pdConnector,
      linked,
    },
  );
  return {
    omni: false,
    selection: {
      hardware: hwId, variant: variantKey, strategy, nodes: nodeCount, features, modes,
      advanced, frontend, kv_offload: kvOffload || null, pd: pdNodes,
      preset: rendering?.preset ?? null, point: rendering?.preset_point ?? null,
      adjusted,
    },
    rendering,
    hwProfile,
    options,
  };
}
