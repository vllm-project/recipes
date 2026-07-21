/**
 * Core command synthesis: Recipe + Variant + Strategy + Hardware + Features → vllm serve command.
 * Pure functions, no I/O.
 */

// NVL4-only env vars: NCCL symmetric memory + cross-node NVLink kernels +
// UCX device selection. Tuned for GB200/GB300 NVL4 trays (NVL72 rack with
// NVLink between nodes); on plain HGX nodes they're either inert or harmful,
// so they're filtered out unless the user picks a GB NVL4 profile.
const NVL4_ONLY_ENV_KEYS = new Set([
  "VLLM_USE_NCCL_SYMM_MEM",
  "NCCL_CUMEM_ENABLE",
  "NCCL_MNNVL_ENABLE",
  "NCCL_NVLS_ENABLE",
  "UCX_NET_DEVICES",
]);
const NVL4_HW_IDS = new Set(["gb200", "gb300"]);

/**
 * Normalize gpu_generation to a single string for hardware_overrides lookup.
 */
function normalizeGeneration(gen) {
  if (Array.isArray(gen)) {
    // ada/blackwell consumer cards → use first
    return gen[0];
  }
  return gen;
}

/**
 * Auto-fit TP for single_node_tp: binary — TP=1 when the weights fit on a
 * single GPU, otherwise fan out to the full node (TP=gpu_count). Skips
 * intermediate sizes (TP=2/4) intentionally: on a multi-GPU node users
 * generally want either single-GPU serving (lowest overhead) or full-node
 * sharding (max throughput); half-node TP leaves GPUs idle without the
 * latency wins that would justify it.
 */
function autoFitTp(vramMinGb, perGpuVram, gpuCount) {
  if (!vramMinGb || !perGpuVram || perGpuVram <= 0) return gpuCount;
  return vramMinGb <= perGpuVram ? 1 : gpuCount;
}

/**
 * Single-node TP size for a recipe/variant/hardware triple. Exported so
 * the UI ("using N of M GPUs" hint) uses the same rule as the generated
 * command. See the precedence note in resolveCommand.
 */
export function resolveSingleNodeTp(recipe, variant, hwProfile, strategyName = "single_node_tp") {
  const gpuCount = typeof hwProfile?.gpu_count === "number" ? hwProfile.gpu_count : 1;
  if (strategyName !== "single_node_tp") return gpuCount;
  // Variant-level override beats recipe-level. Used when a non-default variant
  // (typically an FP8-block-quantized sibling) needs a smaller TP than the
  // bf16 default — e.g. moe_intermediate_size=1536 demands TP ≤ 4 under FP8
  // block_n=128, while bf16 happily runs at TP=8.
  const variantTp = variant?.tp;
  if (typeof variantTp === "number" && variantTp > 0) {
    return Math.min(variantTp, gpuCount);
  }
  const declaredTp = recipe?.strategy_overrides?.[strategyName]?.tp;
  if (typeof declaredTp === "number" && declaredTp > 0) {
    return Math.min(declaredTp, gpuCount);
  }
  const perGpuVram = hwProfile?.vram_gb && gpuCount ? hwProfile.vram_gb / gpuCount : 0;
  const vramMinGb = variant?.vram_minimum_gb || 0;
  return autoFitTp(vramMinGb, perGpuVram, gpuCount);
}

/**
 * Given a recipe and hardware profile, recommend the default strategy.
 *
 * Tensor Parallel is the default for every model — it's the most widely
 * tested, works for both dense and MoE. TEP / DEP / PD-cluster are
 * advanced strategies that users can opt into explicitly.
 */
export function recommendStrategy(recipe, _hwProfile, nodeCount = 1) {
  const compatible = recipe.compatible_strategies || [];
  // Recipe-level override — useful when the global TP-first preference is wrong
  // for a model (e.g. MoE recipes where TEP/DEP is the intended default and TP
  // is offered only as a latency-oriented alternative).
  const explicit = recipe.default_strategy;
  if (explicit && compatible.includes(explicit)) {
    if (nodeCount > 1 && explicit.startsWith("single_node_")) {
      // Single-node default at >1 node: prefer the multi-node sibling so a
      // recipe whose single-node default is single_node_tep doesn't fall back
      // to the global multi-node preference order (which puts dep before tep).
      const sibling = explicit.replace(/^single_node_/, "multi_node_");
      if (compatible.includes(sibling)) return sibling;
    } else {
      return explicit;
    }
  }
  if (nodeCount > 1) {
    if (compatible.includes("multi_node_tp")) return "multi_node_tp";
    if (compatible.includes("multi_node_dep")) return "multi_node_dep";
    if (compatible.includes("multi_node_tep")) return "multi_node_tep";
  }
  if (compatible.includes("single_node_tp")) return "single_node_tp";
  return compatible[0] || "single_node_tp";
}

/**
 * PD-cluster pool parallelism modes offered for a recipe, derived from its
 * `compatible_strategies[]`. Each strategy id encodes a parallelism family in
 * its suffix — `*_tp` / `*_tp_pp` → TP, `*_tep` → TEP, `*_dep` → DEP — and the
 * pd_cluster pools reuse that same vocabulary. So a recipe listing
 * single_node_tp + multi_node_tep + multi_node_dep offers TP / TEP / DEP per
 * pool, while a dense model that only lists `*_tp` strategies offers TP alone.
 *
 * Returns an ordered subset of ["tp", "tep", "dep"]; never empty (falls back to
 * ["tp"]). EP modes (tep/dep) are inherently MoE-only, which compatible_strategies
 * already encodes — dense recipes don't list them.
 */
export function pdPoolModes(recipe) {
  const compat = recipe?.compatible_strategies || [];
  const modes = new Set();
  for (const s of compat) {
    if (s === "pd_cluster") continue;
    if (/(?:^|_)dep$/.test(s)) modes.add("dep");
    else if (/(?:^|_)tep$/.test(s)) modes.add("tep");
    else if (/(?:^|_)tp(?:_pp)?$/.test(s)) modes.add("tp");
  }
  if (modes.size === 0) modes.add("tp");
  return ["tp", "tep", "dep"].filter((m) => modes.has(m));
}

/**
 * Precision → allowed hardware constraint.
 * NVFP4 is NVIDIA Blackwell-only (sm_100+). FP4 generic is also Blackwell-only
 * in practice. AWQ/GPTQ/INT quants run on most NVIDIA+AMD hardware.
 */
const PRECISION_HARDWARE_CONSTRAINTS = {
  nvfp4: { brand: "NVIDIA", generation: "blackwell" },
  fp4: { brand: "NVIDIA", generation: "blackwell" },
};

function matchesConstraint(profile, constraint) {
  if (!constraint) return true;
  if (constraint.brand && profile.brand !== constraint.brand) return false;
  if (constraint.generation) {
    const profileGen = profile.generation || profile.gpu_generation;
    const gens = Array.isArray(profileGen) ? profileGen : [profileGen];
    if (!gens.includes(constraint.generation)) return false;
  }
  return true;
}

/**
 * Check whether a hardware profile is compatible with a variant based on
 * precision constraints (e.g., NVFP4 requires Blackwell). Does NOT check VRAM.
 */
export function isPrecisionCompatible(profile, variant) {
  const constraint = PRECISION_HARDWARE_CONSTRAINTS[variant?.precision];
  return matchesConstraint(profile, constraint);
}

/**
 * Recipe-level hardware opt-out: author marked `meta.hardware.<id>: unsupported`
 * because the model is known not to run on that GPU. Absence = silent default
 * (assumed to work); `verified` = positively tested (separate signal).
 */
export function isHardwareSupported(recipe, hwId) {
  return recipe?.meta?.hardware?.[hwId] !== "unsupported";
}

/**
 * Variant-level hardware allowlist. Missing/empty means the variant inherits
 * the recipe's normal hardware compatibility; otherwise only listed profile
 * ids may render or be selected.
 */
export function isVariantHardwareSupported(variant, hwId) {
  const supported = variant?.supported_hardware;
  return !Array.isArray(supported) || supported.length === 0 || supported.includes(hwId);
}

/**
 * A feature with a `modes` map is single-select ("pick one of N") rather than a
 * boolean toggle — e.g. `spec_decoding` offering MTP / DFlash / DSpark. This
 * returns which mode is active by default when the feature is turned on:
 * `default_mode` if it names a real mode, else the first declared mode.
 * Returns undefined for plain boolean features (no `modes`).
 */
export function defaultModeFor(feature) {
  if (!feature?.modes || typeof feature.modes !== "object") return undefined;
  const keys = Object.keys(feature.modes);
  if (feature.default_mode && keys.includes(feature.default_mode)) return feature.default_mode;
  return keys[0];
}

/**
 * Whether a single mode may run on a given hardware profile. A mode can gate
 * itself with a tri-state `hardware` map keyed by generation (hopper/blackwell/
 * amd) or gpu-id; a value of `unsupported` disables it (e.g. a DFlash draft
 * checkpoint that only ships for Blackwell). Absence = runs everywhere.
 */
export function isModeSupported(mode, hwProfile, hwId) {
  const hw = mode?.hardware;
  if (!hw || typeof hw !== "object") return true;
  const gen = normalizeGeneration(hwProfile?.generation || hwProfile?.gpu_generation);
  if (gen && hw[gen] === "unsupported") return false;
  if (hwId && hw[hwId] === "unsupported") return false;
  return true;
}

/**
 * Whether a mode is available on a given variant. A mode may restrict itself to
 * specific checkpoints with `variants: [<variant_key>, ...]` — e.g. the dspark
 * method only exists on the fused DSpark checkpoint, so FP8/NVFP4 offer MTP
 * only. Absence of `variants` = available on every variant.
 */
export function isModeAllowedForVariant(mode, variantKey) {
  const allow = mode?.variants;
  return !Array.isArray(allow) || allow.length === 0 || allow.includes(variantKey);
}

/**
 * Whether a feature is available under a given strategy. A feature may restrict
 * itself with `strategies: [<strategy_id>, ...]` when its args collide with a
 * strategy's own flags or its setup doesn't generalize — e.g. an offloading
 * feature whose --kv-transfer-config would clobber pd_cluster's NixlConnector
 * config (features emit last, so last-wins dedupe would win), or whose
 * companion server is node-local. Absence of `strategies` = every strategy.
 */
export function isFeatureAllowedForStrategy(feature, strategyName) {
  const allow = feature?.strategies;
  return !Array.isArray(allow) || allow.length === 0 || allow.includes(strategyName);
}

/**
 * Whether a composing KV-offload option (taxonomy.kv_offload.*) may run under
 * a strategy. Two layers: pd_cluster / kv_store_lb are excluded for EVERY
 * option (both own --kv-transfer-config; last-wins dedupe would corrupt it),
 * and an option may further restrict itself with a `strategies` allowlist
 * (e.g. LMCache's node-local MP server rules out multi-node engines).
 */
export function isKvOffloadAllowedForStrategy(option, strategyName, strategy) {
  if (!option) return false;
  if (strategy?.deploy_type === "pd_cluster" || strategy?.deploy_type === "kv_store_lb") return false;
  const allow = option.strategies;
  return !Array.isArray(allow) || allow.length === 0 || allow.includes(strategyName);
}

/**
 * The effective mode key for a (feature, variant, hardware, user-selection)
 * tuple — the single source of truth shared by the command emitter and the UI.
 * Only considers modes allowed on this variant + hardware, then prefers, in
 * order: the user's explicit pick, the variant's `default_modes[featureKey]`,
 * the feature's `default_mode`, else the first allowed mode. Returns undefined
 * when the feature has no modes or none are allowed here.
 */
export function resolveModeKey(feature, featureKey, variantObj, variantKey, hwProfile, hwId, selectedMode) {
  if (!feature?.modes || typeof feature.modes !== "object") return undefined;
  const allowed = Object.keys(feature.modes).filter(
    (k) => isModeAllowedForVariant(feature.modes[k], variantKey)
      && isModeSupported(feature.modes[k], hwProfile, hwId)
  );
  if (allowed.length === 0) return undefined;
  for (const c of [selectedMode, variantObj?.default_modes?.[featureKey], feature.default_mode]) {
    if (c && allowed.includes(c)) return c;
  }
  return allowed[0];
}

/**
 * List hardware profiles compatible with a variant by precision constraint
 * only. VRAM is NOT a blocking constraint — users can scale out via multi-node
 * TP/DP, so any profile that satisfies the precision requirement is valid.
 */
export function listCompatibleHardware(hwProfiles, variant, recipe) {
  return Object.entries(hwProfiles)
    .filter(([id, p]) =>
      isPrecisionCompatible(p, variant)
      && isHardwareSupported(recipe, id)
      && isVariantHardwareSupported(variant, id)
    )
    .map(([id]) => id);
}

/**
 * Single-node fit check: strategies bound to one node (TP, TEP, DEP) shard
 * weights across that node's GPUs and can't scale VRAM further. Returns false
 * when the variant's declared `vram_minimum_gb` exceeds the node's `vram_gb`.
 * Missing size info → treat as fit (don't block on incomplete metadata).
 */
export function fitsSingleNode(hwProfile, variant) {
  const nodeVram = typeof hwProfile?.vram_gb === "number" ? hwProfile.vram_gb : 0;
  const modelVram = variant?.vram_minimum_gb || 0;
  if (modelVram <= 0 || nodeVram <= 0) return true;
  return modelVram <= nodeVram;
}

/**
 * Whether a hardware profile can scale VRAM by adding nodes. Defaults to true;
 * single-GPU desktop workstations (e.g. DGX Station) set `scalable: false` in
 * the taxonomy because they can't be clustered into a multi-node deployment.
 * On non-scalable hardware VRAM becomes a hard constraint — a variant that
 * doesn't fit one box has nowhere to grow.
 *
 * NB: the pre-existing `multi_node: false` on every profile is unrelated dead
 * metadata (it's false everywhere) — don't conflate it with scalability.
 */
export function isHardwareScalable(hwProfile) {
  return hwProfile?.scalable !== false;
}

/**
 * Mooncake's transfer engine ships CUDA and ROCm builds only, so the KV-store
 * deployments are limited to NVIDIA/AMD GPUs — CPU (Intel Xeon) and TPU
 * (Google) backends have no wheel and no RDMA GPU-transfer path. Shared by
 * the UI's Mooncake pill gate and the JSON API's kv-rendering filter.
 */
export function isKvStoreBrandSupported(hwProfile) {
  return hwProfile?.brand === "NVIDIA" || hwProfile?.brand === "AMD";
}

/**
 * A variant is runnable on a hardware profile when it's precision-compatible
 * and either the hardware can scale out (multi-node supplies more VRAM) or the
 * weights already fit single-node. Used to disable variant pills on
 * non-scalable hardware where the variant has nowhere to shard.
 */
export function variantRunsOnHardware(hwProfile, variant, hwId = null) {
  if (!isPrecisionCompatible(hwProfile, variant)) return false;
  if (hwId && !isVariantHardwareSupported(variant, hwId)) return false;
  if (isHardwareScalable(hwProfile)) return true;
  return fitsSingleNode(hwProfile, variant);
}

/**
 * For non-scalable hardware: pick the best variant that actually runs on it —
 * precision-compatible and single-node-fitting, preferring the
 * largest-footprint (highest fidelity) among those that fit. Returns a variant
 * key, or null when nothing fits. Used to auto-fall off an oversized variant
 * (e.g. BF16 → FP8) when the user selects a single-GPU workstation.
 */
export function pickFittingVariant(recipe, hwProfile, hwId = null) {
  const fitting = Object.entries(recipe.variants || {}).filter(
    ([, v]) => variantRunsOnHardware(hwProfile, v, hwId)
  );
  if (!fitting.length) return null;
  fitting.sort((a, b) => (b[1].vram_minimum_gb || 0) - (a[1].vram_minimum_gb || 0));
  return fitting[0][0];
}

/**
 * Single-node PD splits the node 50/50 between prefill and decode. Each half
 * holds a full model across its TP group, so the node must fit 2× the model's
 * VRAM. Also requires at least 2 GPUs to split.
 */
export function pdFitsSingleNode(hwProfile, variant) {
  if (!hwProfile || !variant) return false;
  const gpuCount = typeof hwProfile.gpu_count === "number" ? hwProfile.gpu_count : 0;
  if (gpuCount < 2) return false;
  const nodeVram = typeof hwProfile.vram_gb === "number" ? hwProfile.vram_gb : 0;
  const modelVram = variant.vram_minimum_gb || 0;
  return nodeVram >= 2 * modelVram;
}

/**
 * Given a variant, pick the preferred default hardware:
 * - If variant requires Blackwell (e.g., NVFP4), prefer B200 then GB200
 * - Otherwise H200 is the canonical default
 * `recipe` is optional; when provided, hardware marked `unsupported` is excluded.
 */
export function pickDefaultHardware(hwProfiles, variant, recipe) {
  const constraint = PRECISION_HARDWARE_CONSTRAINTS[variant?.precision];
  const compatible = Object.entries(hwProfiles).filter(
    ([id, p]) =>
      matchesConstraint(p, constraint)
      && isHardwareSupported(recipe, id)
      && isVariantHardwareSupported(variant, id)
  );

  if (constraint?.generation === "blackwell") {
    if (compatible.some(([id]) => id === "b200")) return "b200";
    if (compatible.some(([id]) => id === "gb200")) return "gb200";
  }

  if (compatible.some(([id]) => id === "h200")) return "h200";
  // Fallback: NVIDIA-first, then alphabetical
  const brandOrder = { NVIDIA: 0, AMD: 1 };
  compatible.sort((a, b) => {
    const ba = brandOrder[a[1].brand] ?? 9;
    const bb = brandOrder[b[1].brand] ?? 9;
    if (ba !== bb) return ba - bb;
    return a[0].localeCompare(b[0]);
  });
  return compatible[0]?.[0] || "h200";
}

// Resolve the Docker image / GPU flags / brand key for the active hardware.
// Shared by the install block (docker pull line) and the main command blocks
// (docker run wrapping). Precedence: variant.docker_image → model.docker_image
// → DEFAULT_IMAGE[brand].
//
// `docker_image` shapes:
//   "vllm/vllm-openai:x"               (NVIDIA-only)
//   { nvidia, amd, tpu }               (brand-keyed; each value is a string)
//   { cu129, cu130 }                   (NVIDIA CUDA-keyed — explicit paired tags,
//                                        auto-suffix is skipped in favor of these)
//   { nvidia: { cu129, cu130 }, amd, tpu }  (mixed: NVIDIA value may be a CUDA map)
// Exact variant+hardware overrides may also set
//   variants.<key>.hardware_overrides.<hw_id>.docker_image
//
// When a CUDA map is in play, `cudaMap` is returned so the caller can pick by
// the user's `dockerCudaVariant` toggle instead of appending `-cu129`/`-cu130`.
export function computeDockerMeta(recipe, variant, hwProfile, hwId = null) {
  // When `model.nightly_required: true` and no explicit `docker_image` pin,
  // swap the brand defaults to nightly tags so the Install block matches the
  // nightly pip wheel that's also being rendered. vLLM publishes `:nightly`
  // for all three brand repos. NVIDIA also publishes `cu129-nightly` /
  // `cu130-nightly` — the CUDA suffix logic in the caller handles those.
  const nightlyRequired = recipe.model?.nightly_required === true;
  const DEFAULT_IMAGE = nightlyRequired
    ? {
        nvidia: "vllm/vllm-openai:nightly",
        amd: "vllm/vllm-openai-rocm:nightly",
        tpu: "vllm/vllm-tpu:nightly",
      }
    : {
        nvidia: "vllm/vllm-openai:latest",
        amd: "vllm/vllm-openai-rocm:latest",
        tpu: "vllm/vllm-tpu:latest",
        intel: "vllm/vllm-openai-cpu:latest-x86_64",
      };
  const isAmd = hwProfile?.brand === "AMD";
  const isTpu = hwProfile?.generation === "tpu";
  const isIntel = hwProfile?.generation === "cpu" ||hwProfile?.brand === "Intel";
  const brandKey = isTpu ? "tpu" : isAmd ? "amd" : isIntel ? "intel" : "nvidia";
  // Exact variant+hardware image overrides win over variant-wide and
  // model-wide images (for example, an MI355X-only ROCm nightly).
  const exactHardwareOverride = hwId
    ? variant?.hardware_overrides?.[hwId]?.docker_image
    : null;

  const isCudaMap = (v) =>
    v && typeof v === "object" && ("cu129" in v || "cu130" in v);

  let pinned = typeof exactHardwareOverride === "string" ? exactHardwareOverride : null;
  let cudaMap = null;

  function applyOverride(override) {
    if (!override || pinned || cudaMap) return;
    if (typeof override === "string") {
      if (brandKey === "nvidia") pinned = override;
      return;
    }
    if (typeof override === "object") {
      const isBrandKeyed = "nvidia" in override || "amd" in override || "tpu" in override || "intel" in override;
      if (isBrandKeyed) {
        const brandValue = override[brandKey];
        if (typeof brandValue === "string") pinned = brandValue;
        else if (brandKey === "nvidia" && isCudaMap(brandValue)) cudaMap = brandValue;
      } else if (brandKey === "nvidia" && isCudaMap(override)) {
        cudaMap = override;
      }
    }
  }

  applyOverride(variant?.docker_image);
  // A partial variant override falls back to the model image for brands it
  // does not cover instead of skipping directly to the global default.
  applyOverride(recipe.model?.docker_image);

  const image = pinned || DEFAULT_IMAGE[brandKey];
  const gpuFlags = isTpu
    ? "--privileged --network host \\\n  -v /dev/shm:/dev/shm"
    : isAmd
      ? "--device=/dev/kfd --device=/dev/dri \\\n  --security-opt seccomp=unconfined --group-add video"
    : isIntel
      ? "--shm-size=16g"	
      : "--gpus all";
  return { image, gpuFlags, brandKey, isAmd, isTpu, isIntel, pinned, cudaMap, nightlyRequired };
}

// argv form of the brand-specific GPU flags from computeDockerMeta. Mirrors
// the shell-string above token-for-token so docker_command and docker_argv
// stay consistent.
function dockerGpuArgv(meta) {
  if (meta.isTpu) return ["--privileged", "--network", "host", "-v", "/dev/shm:/dev/shm"];
  if (meta.isAmd) {
    return [
      "--device=/dev/kfd", "--device=/dev/dri",
      "--security-opt", "seccomp=unconfined",
      "--group-add", "video",
    ];
  }
  if (meta.isIntel) {
    return ["--shm-size", "16g"];
  }	
  return ["--gpus", "all"];
}

// Wrap a `vllm serve MODEL <args>` command in `docker run`. The vllm/vllm-openai
// image's entrypoint is `vllm serve`, so we pass MODEL and the trailing args as
// CMD. Env vars become `-e KEY=VAL` inside the container.
export function buildDockerRun({ command, env, image, gpuFlags, port = 8000 }) {
  const envFlags = Object.entries(env || {})
    .map(([k, v]) => `-e ${k}=${v}`)
    .join(" \\\n  ");
  const modelId = command.match(/^vllm serve (\S+)/)?.[1] || "MODEL";
  const serveBody = command.replace(/^vllm serve \S+\s*\\?\n?\s*/, "");
  return `docker run ${gpuFlags} \\
  --privileged --ipc=host -p ${port}:${port} \\
  -v ~/.cache/huggingface:/root/.cache/huggingface \\${envFlags ? `\n  ${envFlags} \\` : ""}
  ${image} ${modelId}${serveBody ? ` \\\n  ${serveBody}` : ""}`;
}

// argv companion to buildDockerRun. `argv` here is the inner command's argv —
// `["vllm", "serve", "<model>", ...flags]` from formatArgv. Returns the full
// docker-run argv ready to spawn without a shell.
export function buildDockerArgv({ argv, env, meta, port = 8000 }) {
  const envFlags = [];
  for (const [k, v] of Object.entries(env || {})) {
    envFlags.push("-e", `${k}=${v}`);
  }
  // `vllm serve <model> <...flags>` → CMD becomes `<model> <...flags>` since
  // the image's entrypoint is already `vllm serve`.
  const cmdArgs = argv[0] === "vllm" && argv[1] === "serve" ? argv.slice(2) : argv;
  return [
    "docker", "run",
    ...dockerGpuArgv(meta),
    "--privileged", "--ipc=host",
    "-p", `${port}:${port}`,
    "-v", "~/.cache/huggingface:/root/.cache/huggingface",
    ...envFlags,
    meta.image,
    ...cmdArgs,
  ];
}

// Dedupe `--flag value` pairs by keeping only the LAST occurrence of each
// flag. Matches shell "last wins" semantics and makes recipe/variant/
// hardware overrides transparently shadow strategy defaults — the strategy
// YAML sets a baseline, anything the recipe author writes later overrides
// it without leaving stale pairs in the rendered command.
function dedupeArgs(args) {
  // Parse into units so (flag, value) stay together.
  const units = [];
  for (let i = 0; i < args.length; i++) {
    const cur = args[i];
    if (typeof cur === "string" && cur.startsWith("-")) {
      const next = args[i + 1];
      if (next !== undefined && !(typeof next === "string" && next.startsWith("-"))) {
        units.push({ flag: cur, value: next });
        i++;
      } else {
        units.push({ flag: cur });
      }
    } else {
      units.push({ positional: cur });
    }
  }
  // Last-wins: walk backward, mark first sighting of each flag as keep.
  const seen = new Set();
  const keep = new Array(units.length).fill(false);
  for (let i = units.length - 1; i >= 0; i--) {
    const u = units[i];
    if (u.positional !== undefined) {
      keep[i] = true;
    } else if (!seen.has(u.flag)) {
      seen.add(u.flag);
      keep[i] = true;
    }
  }
  const out = [];
  for (let i = 0; i < units.length; i++) {
    if (!keep[i]) continue;
    const u = units[i];
    if (u.positional !== undefined) out.push(u.positional);
    else {
      out.push(u.flag);
      if (u.value !== undefined) out.push(u.value);
    }
  }
  return out;
}

// Wrap values containing shell-special chars in single quotes so the rendered
// command is paste-safe. Without this, JSON values like
// `{"cudagraph_mode":"FULL_AND_PIECEWISE"}` trigger brace expansion and get
// their double quotes stripped by bash.
function shellQuote(s) {
  if (typeof s !== "string" || s.length === 0) return s;
  // Bare $VAR references must stay unquoted so bash expands them at runtime.
  if (/^\$[A-Z_][A-Z0-9_]*$/.test(s)) return s;
  if (/^[A-Za-z0-9_./=:@,+%-]+$/.test(s)) return s;
  return `'${s.replace(/'/g, "'\\''")}'`;
}

/**
 * Resolve the `vllm serve --omni` command for a vllm-omni recipe.
 *
 * Much simpler than the regular `resolveCommand`: no strategy logic, no
 * multi-node, no router. Just a single online-serving process whose model id
 * (and optionally extra args) can be swapped per omni task — e.g. Wan2.2 picks
 * a different checkpoint for T2V vs I2V vs TI2V.
 *
 * `task` is the resolved entry from `resolveOmniTasks(recipe)`:
 *   { id, modelId?, extraArgs?, ... }
 *
 * Outliers:
 *   - `recipe.omni.serve_binary: "vllm-omni serve"` swaps the binary (today's
 *     only user is stable-audio-open, whose handler doesn't ship in `vllm`).
 *   - `recipe.omni.port` overrides the rendered `--port` flag (default 8000).
 */
export function resolveOmniCommand(recipe, variantKey, task, hwProfile) {
  const variant = recipe.variants?.[variantKey] || recipe.variants?.default || {};
  const modelId = task?.modelId || variant.model_id || recipe.model?.model_id || "unknown";
  const gen = normalizeGeneration(hwProfile?.generation || hwProfile?.gpu_generation);
  const isNvidia = hwProfile?.brand === "NVIDIA";

  const env = {};
  Object.assign(env, recipe.model?.base_env || {});
  if (variantKey !== "default" && variant.extra_env) Object.assign(env, variant.extra_env);
  const ho = recipe.hardware_overrides?.[gen]
    || (isNvidia ? recipe.hardware_overrides?.nvidia : null);
  if (ho?.extra_env) Object.assign(env, ho.extra_env);

  const args = [];
  if (recipe.model?.base_args) args.push(...recipe.model.base_args);
  if (variantKey !== "default" && variant.extra_args) args.push(...variant.extra_args);
  if (task?.extraArgs?.length) args.push(...task.extraArgs);
  if (ho?.extra_args) args.push(...ho.extra_args);
  // --omni is the toggle that puts vllm into omni-handler mode. Always emit it
  // last — dedupeArgs's last-wins rule keeps it idempotent if the recipe also
  // declares it in base_args.
  args.push("--omni");

  const serveBinary = recipe.omni?.serve_binary || "vllm serve";

  const filtered = dedupeArgs(args.filter(Boolean));
  const lines = [];
  for (let i = 0; i < filtered.length; i++) {
    const cur = filtered[i];
    const next = filtered[i + 1];
    if (cur.startsWith("-") && next !== undefined && !next.startsWith("-")) {
      lines.push(`${cur} ${shellQuote(next)}`);
      i++;
    } else {
      lines.push(cur);
    }
  }
  const command = lines.length === 0
    ? `${serveBinary} ${modelId}`
    : `${serveBinary} ${modelId} \\\n  ${lines.join(" \\\n  ")}`;

  return { command, env, modelId };
}

/**
 * Resolve a complete vllm serve command from recipe + user selections.
 *
 * Returns: { command, env, deployType } for single_node/multi_node,
 *          { prefillCommand, decodeCommand, routerConfig, env, deployType } for pd_cluster.
 */
export function resolveCommand(recipe, variantKey, strategyName, hwProfileId, enabledFeatures, strategies, taxonomy, advancedArgs = [], nodeCount = 1, pdNodes = null, featureModes = {}, kvOffload = null, kvInstances = null) {
  const variant = recipe.variants?.[variantKey] || recipe.variants?.default || {};
  const strategy = strategies[strategyName] || {};
  const hwProfile = taxonomy.hardware_profiles?.[hwProfileId] || {};
  const gen = normalizeGeneration(hwProfile.generation || hwProfile.gpu_generation);
  const gpuCount = typeof hwProfile.gpu_count === "number" ? hwProfile.gpu_count : 1;
  const totalGpus = gpuCount * Math.max(1, nodeCount);

  // single_node_tp TP size precedence (see resolveSingleNodeTp):
  //   1. `strategy_overrides.single_node_tp.tp` (explicit recipe override)
  //   2. auto-fit from `variant.vram_minimum_gb` / per-GPU VRAM (pow-2).
  //   3. gpuCount (legacy fan-out when the recipe has no VRAM hint).
  //
  // TEP/DEP require full TP by topology; multi-node is explicit scale-out.
  const singleNodeTp = resolveSingleNodeTp(recipe, variant, hwProfile, strategyName);

  // The served checkpoint is owned by the variant axis (variant.model_id > base).
  // Spec-decoding modes are single-select and contribute only args (the
  // --speculative-config), so exactly one is ever emitted. A fused-spec
  // checkpoint (e.g. DeepSeek-V4-Pro-DSpark) is a variant that steers the spec
  // mode via `default_modes`; the mode never overrides model_id.
  const modelId = variant.model_id || recipe.model?.model_id || "unknown";
  const variantHardwareOverride = variant?.hardware_overrides?.[hwProfileId];

  // Mooncake composition: `kvOffload` may name a kv_store deployment
  // (deploy_type: kv_store_lb). It does NOT replace the serving strategy —
  // the KV layer is orthogonal to parallelism — instead the serving strategy
  // builds each instance's command as usual, the kv store's connector args +
  // env are appended (last-wins), and the result is wrapped in the
  // router/master/(store) deployment shell. Under pd_cluster the per-role
  // MultiConnector path composes instead (see buildArgs step 8).
  const kvStoreStrat = kvOffload && strategies?.[kvOffload]?.deploy_type === "kv_store_lb"
    ? strategies[kvOffload]
    : null;
  const kvComposing = !!kvStoreStrat && strategy.deploy_type !== "pd_cluster";

  // kv composition only: instance count + which instance's command is being
  // rendered (0-based — affects the multi-node --master-addr naming below).
  // `kvInstances` accepts a bare count or { count, current }.
  const kvInstObj = (kvInstances && typeof kvInstances === "object")
    ? kvInstances
    : { count: kvInstances };
  const kvInstanceCount = (typeof kvInstObj.count === "number" && kvInstObj.count >= 1)
    ? Math.floor(kvInstObj.count)
    : (kvStoreStrat?.default_instances || 2);
  const kvCurrentInstance = Math.max(0, Math.min(kvInstanceCount - 1, kvInstObj.current ?? 0));
  // A kv_store `vllm.install` may be brand-keyed ({ nvidia, amd }) — Mooncake
  // ships a CUDA wheel and a separate non-CUDA build for ROCm. Other brands
  // (Intel CPU, Google TPU) have no wheel at all → null, never a silent
  // fall-through to the CUDA build (they're gated out upstream by
  // isKvStoreBrandSupported, this is defense in depth).
  const resolveBrandInstall = (raw) => {
    if (raw && typeof raw === "object") {
      const key = hwProfile?.brand === "AMD" ? "amd"
        : hwProfile?.brand === "NVIDIA" ? "nvidia"
        : null;
      return key ? (raw[key] || null) : null;
    }
    return raw || null;
  };

  // Helper to merge args
  function buildArgs(roleOverride, nodeRole) {
    const args = [];

    // Order:
    // 1. base_args         (trust-remote-code, required flags)
    // 2. variant           (quantization flags)
    // 3. strategy cluster  (parallelism — strategy.vllm_args + -tp/-dp grouped together)
    // 4. strategy_override (recipe's per-strategy tweaks)
    // 5. hardware_override (per-generation tweaks)
    // 6. advanced          (user-picked perf flags)
    // 7. features          (tool_calling, reasoning, mtp — last for readability)

    // 1. base_args
    if (recipe.model?.base_args) args.push(...recipe.model.base_args);

    // 2. Variant extra args
    if (variantKey !== "default" && variant.extra_args) args.push(...variant.extra_args);
    if (variantKey !== "default" && variantHardwareOverride?.extra_args) {
      args.push(...variantHardwareOverride.extra_args);
    }

    // 3. Strategy args + parallel size (grouped together so -tp/-dp sits next to -ep etc.)
    if (strategy.deploy_type !== "pd_cluster") {
      if (strategy.vllm_args) args.push(...strategy.vllm_args);
    } else if (roleOverride && strategy[roleOverride]?.vllm_args) {
      args.push(...strategy[roleOverride].vllm_args);
    }
    const parallelFlag = strategy.parallel_flag || "--tensor-parallel-size";
    const isMulti = strategy.deploy_type === "multi_node" && nodeCount > 1;
    const isPdMulti = strategy.deploy_type === "pd_cluster" && nodeCount > 1;
    // Multi-node rendezvous host. When Mooncake composes with a multi-node
    // strategy, each instance's head is the same host the router lists as
    // $VLLM_INSTANCE_N — reuse that name so one Endpoints value feeds both.
    // With a single instance there is no router (and no INSTANCE naming), so
    // it falls back to the plain multi-node convention: $HEAD_IP.
    const mpMasterAddr = kvComposing && kvInstanceCount > 1
      ? `$VLLM_INSTANCE_${kvCurrentInstance + 1}`
      : "$HEAD_IP";

    if (isMulti && parallelFlag === "--data-parallel-size") {
    // dp-local may be overridden by the recipe (hybrid TP+DP: TP within a node,
    // DP across nodes → dp_local < gpuCount).
    // The example worker is node 1, so its DP start rank = 1 × dp_local, not a
    // full node's worth of GPUs.
    const soDep = recipe.strategy_overrides?.[strategyName];
    const soDepHo = soDep?.hardware_overrides?.[gen]
    || (hwProfile?.brand === "NVIDIA" ? soDep?.hardware_overrides?.nvidia : null);
    const depOv = [
    ...(soDep?.vllm_args || []),
    ...(soDep?.extra_args || []),
    ...(soDepHo?.extra_args || []),
      ];
    const i = depOv.lastIndexOf("--data-parallel-size-local");
    const dpLocal = i >= 0 ? Number(depOv[i + 1]) : gpuCount;
    args.push("--data-parallel-size", String(totalGpus));
    args.push("--data-parallel-size-local", String(gpuCount));
    args.push("--data-parallel-address", mpMasterAddr);
    if (nodeRole === "worker") {
    args.push("--data-parallel-start-rank", String(dpLocal));
    }
} else if (isMulti && strategy.parallelism === "tp_pp") {
      // TP inside each node, PP across nodes. Cross-node traffic flows through
      // the PP stage boundaries only — much less bandwidth than pure TP across
      // nodes. Suited for very large models on commodity inter-node links.
      args.push("--tensor-parallel-size", String(gpuCount));
      args.push("--pipeline-parallel-size", String(nodeCount));
      args.push("--nnodes", String(nodeCount));
      args.push("--node-rank", nodeRole === "worker" ? "1" : "0");
      args.push("--master-addr", mpMasterAddr);
      if (nodeRole === "worker") args.push("--headless");
    } else if (isMulti) {
      // Multi-node TP/TEP via vLLM multiprocessing (mp) backend:
      // TP spans all GPUs in the cluster; every node runs the same command,
      // varying only --node-rank and (for rank > 0) --headless.
      args.push("--tensor-parallel-size", String(totalGpus));
      args.push("--nnodes", String(nodeCount));
      args.push("--node-rank", nodeRole === "worker" ? "1" : "0");
      args.push("--master-addr", mpMasterAddr);
      if (nodeRole === "worker") args.push("--headless");
    } else if (strategy.deploy_type === "pd_cluster") {
      // PD splits inference across separate prefill and decode pools.
      //
      // New model (per-role nodes + parallelism):
      //   pdNodes = { prefill: {nodes, rank, parallelism?}, decode: {…} }
      //   parallelism precedence: pdNodes (UI pill) → strategy_overrides →
      //     strategy YAML → "tp". Modes: "tp" | "tep" | "dep".
      //   role config (strategy_overrides.pd_cluster.<role>):
      //     parallelism: "tp" | "tep" | "dep"  (default: "tp")
      //     tp:          <int>                  (default: 1 for dep, poolGpus for tp)
      //     parallel_flag: "--…"                (last-resort override)
      //
      // Legacy fallback (pdNodes null):
      //   - nodeCount===1 → both roles on one node, TP=gpuCount/2 each (50/50)
      //   - nodeCount===2 → one full node per role, TP=gpuCount each
      const roleKey = roleOverride;                           // "prefill" | "decode"
      const roleCfg = strategy[roleKey] || {};
      const soRoleCfg = recipe.strategy_overrides?.[strategyName]?.[roleKey] || {};
      // pdNodes accepts two shapes per role — a bare integer (just nodes) or
      // { nodes, rank } when the UI wants to surface a specific DP rank.
      const pdRoleRaw = pdNodes ? pdNodes[roleKey] : undefined;
      const pdRole =
        typeof pdRoleRaw === "number" ? { nodes: pdRoleRaw }
        : (pdRoleRaw && typeof pdRoleRaw === "object") ? pdRoleRaw
        : {};
      const legacyNodes = isPdMulti ? 1 : 0;                  // 0 = co-located half-node
      const rolePoolNodes =
        typeof pdRole.nodes === "number" ? pdRole.nodes : legacyNodes;
      const poolGpus = rolePoolNodes === 0
        ? Math.floor(gpuCount / 2)
        : rolePoolNodes * gpuCount;
      const parallelism = pdRole.parallelism || soRoleCfg.parallelism || roleCfg.parallelism || "tp";

      if (parallelism === "dep") {
        // Data-parallel + expert-parallel pool (Kimi-K2.5 GB200 pattern).
        // One `vllm serve` per NODE. vLLM spawns `--data-parallel-size-local`
        // ranks internally per node; the leader node starts at rank 0 and
        // each follower offsets by `--data-parallel-start-rank = nodeIndex × dp_local`.
        //
        // Example (nodes=4, gpus_per_node=4, tp=1):
        //   dp_local = 4, dp_total = 16
        //   Node 0 → start_rank 0; Node 1 → 4; Node 2 → 8; Node 3 → 12
        const nodesInPool = Math.max(1, rolePoolNodes);
        // `tp` is only rendered when the recipe sets it explicitly (vLLM's
        // own default is 1, so emitting "--tensor-parallel-size 1" would be
        // pure noise). Internal math still uses 1 as the effective value.
        const tpExplicit = soRoleCfg.tp ?? roleCfg.tp;
        const roleTp = tpExplicit ?? 1;
        const dpLocal = Math.max(1, Math.floor(gpuCount / roleTp));
        const dpSize = nodesInPool * dpLocal;
        const nodeIdx = Math.max(0, Math.min(nodesInPool - 1, pdRole.rank ?? 0));
        if (tpExplicit !== undefined) {
          args.push("--tensor-parallel-size", String(roleTp));
        }
        args.push("--data-parallel-size", String(dpSize));
        args.push("--data-parallel-size-local", String(dpLocal));
        // Always emit start-rank (even 0 for node 0) so every node's command
        // has the same shape; only the value differs per node.
        args.push("--data-parallel-start-rank", String(nodeIdx * dpLocal));
        // DP leader is always node 0 of the pool — rendered as NODE_1 (1-indexed,
        // same naming as the router's --prefill/--decode endpoints) so the user
        // only fills one IP per node, not separate LEADER/HOST/HEAD aliases.
        args.push("--data-parallel-address", `$${roleKey.toUpperCase()}_NODE_1`);
        args.push("--data-parallel-rpc-port", `$${roleKey.toUpperCase()}_DP_RPC_PORT`);
        // Multi-node DEP needs hybrid load balancing — without this flag the
        // router distributes requests round-robin across DP ranks, which
        // defeats local batching inside each node.
        if (nodesInPool > 1) {
          args.push("--data-parallel-hybrid-lb");
        }
        args.push("--enable-expert-parallel");
      } else {
        // TP / TEP pool. A single engine spanning the pool's nodes via vLLM's
        // mp multi-node backend: every node runs the same command, varying only
        // --node-rank; rank > 0 adds --headless (no HTTP server). The UI's node
        // selector picks which rank to render via pdRole.rank, mirroring the
        // DEP pool's per-node command rendering. TEP = the same TP layout plus
        // expert parallel (mirrors single_node_tep / multi_node_tep strategies).
        const roleParallelFlag =
          soRoleCfg.parallel_flag ||
          roleCfg.parallel_flag ||
          strategy.parallel_flag ||
          "--tensor-parallel-size";
        args.push(roleParallelFlag, String(Math.max(1, poolGpus)));
        if (parallelism === "tep") {
          args.push("--enable-expert-parallel");
          // Cross-node TEP perf tweak from multi_node_tep; single-node TEP omits it.
          if (rolePoolNodes > 1) args.push("-cc.pass_config.fuse_allreduce_rms=False");
        }
        if (rolePoolNodes > 1) {
          const nodeIdx = Math.max(0, Math.min(rolePoolNodes - 1, pdRole.rank ?? 0));
          args.push("--nnodes", String(rolePoolNodes));
          args.push("--node-rank", String(nodeIdx));
          // TP master = node 0 of pool = NODE_1 (same naming as router endpoints).
          args.push("--master-addr", `$${roleKey.toUpperCase()}_NODE_1`);
          // Followers are GPU workers only — no API server / NIXL side channel.
          if (nodeIdx > 0) args.push("--headless");
        }
      }
    } else {
      // Single-node TP / TEP / DEP. `singleNodeTp` equals `gpuCount` for
      // everything except single_node_tp with a recipe-declared
      // `strategy_overrides.single_node_tp.tp`. Always emit the flag — the
      // command builder's contract is "what you declare is what you see", so
      // hiding TP=1 because it matches vLLM's default would break the
      // declarative mapping and leave the user guessing what's active.
      args.push(parallelFlag, String(singleNodeTp));
    }

    // 4. Strategy overrides from recipe
    //    Accept both `extra_args` (recipe convention) and `vllm_args` (strategy
    //    YAML convention) — some recipes mirror the strategy's role schema.
    const so = recipe.strategy_overrides?.[strategyName];
    if (so) {
      if (roleOverride) {
        const roleOv = so[roleOverride];
        if (roleOv?.extra_args) args.push(...roleOv.extra_args);
        if (roleOv?.vllm_args)  args.push(...roleOv.vllm_args);
      } else {
        if (so.extra_args) args.push(...so.extra_args);
        if (so.vllm_args)  args.push(...so.vllm_args);
      }
    }

    // 5. Hardware overrides
    //    Precedence: generation-specific (hopper/blackwell/amd) > brand-wide (nvidia).
    //    `nvidia:` lets a recipe apply the same overrides to every NVIDIA GPU
    //    without duplicating hopper and blackwell blocks.
    //
    //    A strategy may further override hardware overrides via
    //    `strategy_overrides.<strategy>.hardware_overrides.<gen>` — when set,
    //    it REPLACES the recipe-level hardware override for that gen on that
    //    strategy. Use this to drop a recipe-wide hw flag (e.g. an MoE kernel
    //    backend) for a specific strategy without duplicating the rest.
    const isNvidia = hwProfile?.brand === "NVIDIA";
    const strategyHo = so?.hardware_overrides?.[gen]
      || (isNvidia ? so?.hardware_overrides?.nvidia : null);
    const ho = strategyHo
      || recipe.hardware_overrides?.[gen]
      || (isNvidia ? recipe.hardware_overrides?.nvidia : null);
    if (ho?.extra_args) args.push(...ho.extra_args);

    // 6. Advanced tuning args (from UI's Advanced panel)
    if (advancedArgs && advancedArgs.length) args.push(...advancedArgs);

    // 7. Features last — tool_calling, reasoning, mtp, etc.
    //    A feature can declare per-generation overrides under
    //    `hardware_overrides.<gen>.args`; when present they REPLACE the
    //    feature's default args (not merged), so a recipe can ship different
    //    spec-decoding configs for hopper vs blackwell without dedupe gymnastics.
    for (const f of enabledFeatures || []) {
      const feat = recipe.features?.[f];
      if (!feat) continue;
      if (!isFeatureAllowedForStrategy(feat, strategyName)) continue;
      // A companion-backed feature can't render its helper inside the
      // Mooncake deployment shell (companion tabs are a single-node-block
      // concept), so its args are skipped too — args and companion always
      // gate together, never a command that references an unstarted process.
      if (kvComposing && feat.companion?.command) continue;
      // Single-select sub-modes: the feature declares `modes` and the user
      // picks one (spec_decoding → MTP / DFlash / DSpark). Emit only the
      // effective mode's args — resolveModeKey honours variant + hardware
      // availability and falls back when the selection isn't valid here.
      if (feat.modes && typeof feat.modes === "object") {
        const modeKey = resolveModeKey(feat, f, variant, variantKey, hwProfile, hwProfileId, featureModes?.[f]);
        const mode = modeKey ? feat.modes[modeKey] : null;
        if (mode) {
          const modeHo = mode.hardware_overrides?.[gen]
            || (isNvidia ? mode.hardware_overrides?.nvidia : null);
          const modeArgs = modeHo?.args ?? mode.args;
          if (modeArgs) args.push(...modeArgs);
        }
        continue;
      }
      const featHo = feat.hardware_overrides?.[gen]
        || (isNvidia ? feat.hardware_overrides?.nvidia : null);
      const featArgs = featHo?.args ?? feat.args;
      if (featArgs) args.push(...featArgs);
    }

    // 8. Composing KV offload (taxonomy.kv_offload.<key>: Simple, LMCache) —
    //    the option's --kv-transfer-config is appended last so it wins the
    //    last-wins dedupe over any earlier occurrence. Gating (pd/kv_store
    //    exclusion + per-option strategy allowlist) lives in
    //    isKvOffloadAllowedForStrategy, shared with the UI pills.
    const kvOpt = taxonomy?.kv_offload?.[kvOffload];
    if (kvOpt && isKvOffloadAllowedForStrategy(kvOpt, strategyName, strategy)) {
      args.push(...(kvOpt.args || []));
    }
    // Mooncake composes the same way on any non-PD serving strategy: the
    // MooncakeStoreConnector config appends after the strategy/feature args
    // so its --kv-transfer-config wins the last-wins dedupe. Parallelism
    // stays whatever the strategy emitted above — the KV layer is orthogonal.
    if (kvComposing) {
      args.push(...(kvStoreStrat.vllm?.vllm_args || []));
    }
    // Mooncake × pd_cluster: each role's plain NixlConnector config is
    // swapped (last-wins) for the kv_store YAML's `pd.<role>.args`
    // MultiConnector — Nixl prefill↔decode path + shared MooncakeStoreConnector.
    if (strategy.deploy_type === "pd_cluster" && roleOverride
        && kvStoreStrat?.pd?.[roleOverride]?.args) {
      args.push(...kvStoreStrat.pd[roleOverride].args);
    }

    return args;
  }

  // Helper to merge env
  function buildEnv(roleOverride) {
    const env = {};

    // Base env
    Object.assign(env, recipe.model?.base_env || {});

    // Variant env
    if (variantKey !== "default" && variant.extra_env) Object.assign(env, variant.extra_env);
    if (variantKey !== "default" && variantHardwareOverride?.extra_env) {
      Object.assign(env, variantHardwareOverride.extra_env);
    }

    // Strategy env
    if (strategy.deploy_type !== "pd_cluster") {
      Object.assign(env, strategy.env || {});
    } else if (roleOverride && strategy[roleOverride]?.env) {
      Object.assign(env, strategy[roleOverride].env);
    }
    // Mooncake composition: instances (any serving strategy) and PD roles
    // read the shared config via MOONCAKE_CONFIG_PATH (+ PYTHONHASHSEED for
    // consistent prefix hashing across processes).
    if (kvComposing) {
      Object.assign(env, kvStoreStrat.vllm?.env || {});
    } else if (strategy.deploy_type === "pd_cluster" && roleOverride
        && kvStoreStrat?.pd) {
      Object.assign(env, kvStoreStrat.vllm?.env || {});
    }

    // PD: pin GPUs per role via CUDA_VISIBLE_DEVICES.
    // With per-role `nodes` (new model):
    //   nodes >= 1 → this role owns whole node(s), list all local GPUs
    //   nodes === 0 (legacy co-located half) → split 0..N/2-1 / N/2..N-1
    if (strategy.deploy_type === "pd_cluster" && roleOverride) {
      const raw = pdNodes ? pdNodes[roleOverride] : undefined;
      const pdRoleObj =
        typeof raw === "number" ? { nodes: raw }
        : (raw && typeof raw === "object") ? raw
        : {};
      const legacyNodes = nodeCount > 1 ? 1 : 0;
      const rolePoolNodes =
        typeof pdRoleObj.nodes === "number" ? pdRoleObj.nodes : legacyNodes;
      if (rolePoolNodes >= 1) {
        // Dedicated node(s) — each node owns all its local GPUs. Same value
        // is correct whether the pool is DEP (vllm spawns local ranks) or TP.
        env.CUDA_VISIBLE_DEVICES = Array.from({ length: gpuCount }, (_, i) => i).join(",");
        // Cross-node NCCL/GLOO for this role only kicks in when the role
        // spans 2+ nodes. Single-node roles use NIXL for cross-role transfer
        // and don't need socket-iface hints.
        if (rolePoolNodes >= 2) {
          env.GLOO_SOCKET_IFNAME = "$IFACE_NAME";
          env.NCCL_SOCKET_IFNAME = "$IFACE_NAME";
        }
      } else {
        // Co-located demo: first half for prefill, second half for decode.
        const half = Math.floor(gpuCount / 2);
        if (half > 0) {
          const ids = roleOverride === "prefill"
            ? Array.from({ length: half }, (_, i) => i)
            : Array.from({ length: gpuCount - half }, (_, i) => half + i);
          env.CUDA_VISIBLE_DEVICES = ids.join(",");
        }
      }
    }

    // Strategy overrides env
    const so = recipe.strategy_overrides?.[strategyName];
    if (so) {
      if (so.env) Object.assign(env, so.env);
      if (roleOverride && so[roleOverride]?.env) Object.assign(env, so[roleOverride].env);
      if (!roleOverride && so.extra_env) Object.assign(env, so.extra_env);
    }

    // Hardware overrides env — same precedence as args block: generation key
    // first, then brand-wide `nvidia:` for NVIDIA GPUs. Per-strategy nested
    // hardware_overrides REPLACES the recipe-level for that gen on that
    // strategy (mirrors the args-block behavior).
    const envIsNvidia = hwProfile?.brand === "NVIDIA";
    const envStrategyHo = so?.hardware_overrides?.[gen]
      || (envIsNvidia ? so?.hardware_overrides?.nvidia : null);
    const envHo = envStrategyHo
      || recipe.hardware_overrides?.[gen]
      || (envIsNvidia ? recipe.hardware_overrides?.nvidia : null);
    if (envHo?.extra_env) Object.assign(env, envHo.extra_env);

    // NVL4-only env vars are meaningful only on GB200/GB300 trays. Drop them
    // for any other hardware regardless of where they came from (strategy YAML
    // or recipe-level pd_cluster override).
    if (strategy.deploy_type === "pd_cluster" && !NVL4_HW_IDS.has(hwProfileId)) {
      for (const key of NVL4_ONLY_ENV_KEYS) delete env[key];
    }

    // Per-rank node rewrite: strategy YAML carries `$PREFILL_NODE_1` /
    // `$DECODE_NODE_1` as the rank-0 default for per-node bind hosts (NIXL
    // side channel etc). Only DEP pools open a per-node side channel (one
    // `vllm serve` per node), so for a non-zero DEP rank point those values at
    // NODE_{rank+1} to match that physical node. TP followers run --headless
    // with no side channel, so their host stays pinned to the head (NODE_1).
    if (strategy.deploy_type === "pd_cluster" && roleOverride) {
      const raw = pdNodes ? pdNodes[roleOverride] : undefined;
      const pdRoleObj =
        typeof raw === "number" ? { nodes: raw }
        : (raw && typeof raw === "object") ? raw
        : {};
      const rank = pdRoleObj.rank || 0;
      const rwSoRoleCfg = recipe.strategy_overrides?.[strategyName]?.[roleOverride] || {};
      const rwRoleCfg = strategy[roleOverride] || {};
      const rwParallelism = pdRoleObj.parallelism || rwSoRoleCfg.parallelism || rwRoleCfg.parallelism || "tp";
      if (rank > 0 && rwParallelism === "dep") {
        const oldVar = `$${roleOverride.toUpperCase()}_NODE_1`;
        const newVar = `$${roleOverride.toUpperCase()}_NODE_${rank + 1}`;
        for (const k of Object.keys(env)) {
          if (env[k] === oldVar) env[k] = newVar;
        }
      }
    }

    return env;
  }

  function formatCommand(args) {
    const filtered = dedupeArgs(args.filter(Boolean));
    if (filtered.length === 0) return `vllm serve ${modelId}`;
    // Pair each --flag with its immediate value on the same line so the output
    // reads like the human-written command in the recipe guide, not
    // --flag\n value\n --flag\n value\n ...
    const lines = [];
    for (let i = 0; i < filtered.length; i++) {
      const cur = filtered[i];
      const next = filtered[i + 1];
      if (cur.startsWith("-") && next !== undefined && !next.startsWith("-")) {
        lines.push(`${cur} ${shellQuote(next)}`);
        i++;
      } else {
        lines.push(cur);
      }
    }
    return `vllm serve ${modelId} \\\n  ${lines.join(" \\\n  ")}`;
  }

  // Companion to formatCommand: returns the deduped flat argv (no shell
  // quoting, no line continuations) so JSON consumers can spawn vllm without
  // going through a shell. ["vllm", "serve", "<model>", ...flags].
  function formatArgv(args) {
    const filtered = dedupeArgs(args.filter(Boolean));
    return ["vllm", "serve", modelId, ...filtered];
  }

  const deployType = strategy.deploy_type || "single_node";

  if (deployType === "pd_cluster") {
    // Each pool (prefill / decode) exposes one HTTP endpoint per node, so the
    // router needs `--prefill` / `--decode` repeated to match the pool size.
    // `--intra-node-data-parallel-size` should match dp_local for DEP pools
    // (one local rank per GPU) so routing picks the right DP rank.
    const policy = strategy.router?.policy || "round_robin";

    // Resolve per-role node counts + parallelism mode for the return payload
    // so the UI can render "1 node" vs "4 nodes — duplicate for ranks 1..N-1"
    // notices without rederiving the logic.
    const roleMeta = (role) => {
      const soRoleCfg = recipe.strategy_overrides?.[strategyName]?.[role] || {};
      const roleCfg = strategy[role] || {};
      const legacyNodes = nodeCount > 1 ? 1 : 0;
      const raw = pdNodes ? pdNodes[role] : undefined;
      const pdRole =
        typeof raw === "number" ? { nodes: raw }
        : (raw && typeof raw === "object") ? raw
        : {};
      const nodes = typeof pdRole.nodes === "number" ? pdRole.nodes : legacyNodes;
      const parallelism = pdRole.parallelism || soRoleCfg.parallelism || roleCfg.parallelism || "tp";
      const poolGpus = nodes === 0 ? Math.floor(gpuCount / 2) : nodes * gpuCount;
      const tp = parallelism === "dep"
        ? (soRoleCfg.tp || roleCfg.tp || 1)
        : poolGpus;
      const dpLocal = parallelism === "dep" ? Math.max(1, Math.floor(gpuCount / tp)) : 0;
      const dpSize = parallelism === "dep" ? Math.max(1, nodes) * dpLocal : 1;
      // `currentNode` = which node of the DEP pool this command is for
      // (0..nodes-1). Stored in pdRole.rank for backward-compat with the
      // earlier rank-selector UI; now semantically a node index.
      const nodesInPool = Math.max(1, nodes);
      const currentNode = Math.max(0, Math.min(nodesInPool - 1, pdRole.rank ?? 0));
      const startRank = currentNode * dpLocal;
      return { nodes, parallelism, tp, dpLocal, dpSize, poolGpus, currentNode, startRank };
    };

    const pMeta = roleMeta("prefill");
    const dMeta = roleMeta("decode");
    // Router endpoints: one per node (each node binds its own HTTP --port).
    // Shell-variable form ($PREFILL_NODE_N / $DECODE_NODE_N) so the same name
    // the prefill/decode commands consume is also what the router lists —
    // user fills one set of values in the Endpoints panel, applied everywhere.
    // Endpoint count per pool depends on the parallelism mode:
    //   DEP → one `vllm serve` per node, each binds its own HTTP port, so list
    //         one endpoint per node (NODE_1..NODE_n).
    //   TP  → a single engine spanning n nodes; only the head node (NODE_1)
    //         serves HTTP (followers are --headless), so list exactly one.
    const epCount = (meta) => (meta.parallelism === "dep" ? Math.max(1, meta.nodes || 1) : 1);
    const prefillEndpoints = Array.from(
      { length: epCount(pMeta) },
      (_, i) => `    --prefill http://$PREFILL_NODE_${i + 1}:8001 \\`,
    );
    const decodeEndpoints = Array.from(
      { length: epCount(dMeta) },
      (_, i) => `    --decode http://$DECODE_NODE_${i + 1}:8002 \\`,
    );
    // intra-node-data-parallel-size = max dp_local across the two pools for
    // DEP setups; 1 for pure-TP PD.
    const intraDp = Math.max(pMeta.dpLocal || 0, dMeta.dpLocal || 0, 1);
    // Router --host / --port use the same $ROUTER_HOST / $ROUTER_PORT vars
    // that curl / bench target, so filling them once in the Endpoints panel
    // makes the router and clients agree. Unfilled, shell leaves the $VAR
    // literal — vllm-router won't accept that, so the user must either fill
    // the panel or `export ROUTER_HOST=…; export ROUTER_PORT=…` first.
    const routerLines = [
      `vllm-router --policy ${policy} \\`,
      `    --vllm-pd-disaggregation \\`,
      ...prefillEndpoints,
      ...decodeEndpoints,
      `    --host $ROUTER_HOST \\`,
      `    --port $ROUTER_PORT \\`,
      `    --intra-node-data-parallel-size ${intraDp}`,
    ];
    const routerCommand = routerLines.join("\n");

    const prefillArgs = buildArgs("prefill", null);
    const decodeArgs = buildArgs("decode", null);
    // Mooncake composed into PD: surface master / store / config so the PD
    // block can render their tabs and the per-node config heredoc.
    const pdMooncake = (kvStoreStrat?.pd)
      ? {
          master: {
            command: kvStoreStrat.mooncake_master?.command || "mooncake_master",
            description: kvStoreStrat.mooncake_master?.description || "",
          },
          store: kvStoreStrat.mooncake_store_config ? {
            command: kvStoreStrat.mooncake_store_config.command,
            description: kvStoreStrat.mooncake_store_config?.description || "",
            config: kvStoreStrat.mooncake_store_config?.config || null,
            note: kvStoreStrat.mooncake_store_config?.note || "",
          } : null,
          config: kvStoreStrat.mooncake_vllm_config?.template || {},
          configNote: kvStoreStrat.mooncake_vllm_config?.note || "",
          install: resolveBrandInstall(kvStoreStrat.vllm?.install),
        }
      : null;
    return {
      deployType,
      nodeCount,
      ...(pdMooncake ? { mooncake: pdMooncake } : {}),
      prefill: {
        command: formatCommand(prefillArgs),
        argv: formatArgv(prefillArgs),
        env: buildEnv("prefill"),
        ...pMeta,
      },
      decode: {
        command: formatCommand(decodeArgs),
        argv: formatArgv(decodeArgs),
        env: buildEnv("decode"),
        ...dMeta,
      },
      router: {
        command: routerCommand,
        install: "uv pip install vllm-router",
      },
      routerConfig: strategy.router || { policy: "round_robin" },
    };
  }

  if (kvComposing) {
    // Mooncake deployment shell around the serving strategy: vllm-router
    // (cache-aware, 2+ instances) + mooncake_master + (optionally)
    // mooncake_store_service + N × vLLM instances + a shared config JSON.
    // Each instance's command was built ABOVE by the selected serving
    // strategy (TP/TEP/DEP, single- or multi-node) with the
    // MooncakeStoreConnector args/env appended — the KV layer never decides
    // parallelism.
    //
    // Two orthogonal axes (mirrors agentx's `count` × `nodes_per_instance`):
    //   instances — how many independent vLLM engines sit behind the router
    //               (kvInstances param → kv YAML default_instances → 2).
    //   nodeCount — nodes PER INSTANCE, straight from the Nodes row. 1 =
    //               single-node engine; >1 = each instance shards across
    //               nodes via the serving strategy's own mp-backend args.
    // The router lists one endpoint per instance (each instance's head node).
    const instances = kvInstanceCount;
    const multiNodeInstance = strategy.deploy_type === "multi_node" && nodeCount > 1;
    const vllmArgs = multiNodeInstance ? buildArgs(null, "head") : buildArgs(null, null);
    const workerArgs = multiNodeInstance ? buildArgs(null, "worker") : null;
    const mooncakeVllmConfig = kvStoreStrat.mooncake_vllm_config?.template || {};
    const mooncakeStoreConfig = kvStoreStrat.mooncake_store_config?.config || null;

    // Router construction — mirrors pd_cluster pattern
    const routerPolicy = kvStoreStrat.router?.policy || "cache_aware";
    const routerPort = kvStoreStrat.router?.port || 30080;
    const vllmPort = kvStoreStrat.vllm?.port || 8000;
    const vllmEndpoints = Array.from(
      { length: instances },
      (_, i) => `    --backend-url http://$VLLM_INSTANCE_${i + 1}:${vllmPort} \\`,
    );
    const routerLines = [
      `vllm-router --policy ${routerPolicy} \\`,
      ...vllmEndpoints,
      `    --host $ROUTER_HOST \\`,
      // $ROUTER_PORT (not the literal) so the Endpoints panel's curl/bench
      // target edits also retarget the bind; router.port seeds the default.
      `    --port $ROUTER_PORT \\`,
      `    --cache-threshold 0.5 \\`,
      `    --balance-abs-threshold 4 \\`,
      `    --balance-rel-threshold 1.1`,
    ];
    const routerCommand = routerLines.join("\n");

    return {
      deployType: "kv_store_lb",
      // The strategy each instance runs — surfaced so UI/API consumers can
      // name the composition ("Single-node TEP · Mooncake") without rederiving.
      servingStrategy: strategyName,
      nodeCount,
      instances,
      currentInstance: kvCurrentInstance,
      master: {
        command: kvStoreStrat.mooncake_master?.command || "mooncake_master",
        description: kvStoreStrat.mooncake_master?.description || "",
      },
      store: kvStoreStrat.mooncake_store_config ? {
        command: kvStoreStrat.mooncake_store_config.command,
        description: kvStoreStrat.mooncake_store_config?.description || "",
        config: mooncakeStoreConfig,
        note: kvStoreStrat.mooncake_store_config?.note || "",
      } : null,
      vllm: {
        command: formatCommand(vllmArgs),
        argv: formatArgv(vllmArgs),
        ...(workerArgs ? {
          workerCommand: formatCommand(workerArgs),
          workerArgv: formatArgv(workerArgs),
        } : {}),
        env: buildEnv(null),
        install: resolveBrandInstall(kvStoreStrat.vllm?.install),
      },
      // A single instance needs no LB — clients hit it directly on the vllm
      // port; the router (and its $VLLM_INSTANCE_N naming) only exists at 2+.
      router: instances > 1 ? {
        command: routerCommand,
        install: kvStoreStrat.router?.install || "uv pip install vllm-router",
        port: routerPort,
      } : null,
      mooncakeConfig: mooncakeVllmConfig,
      // Sizing/NIC guidance rendered as # lines above the config heredoc.
      mooncakeConfigNote: kvStoreStrat.mooncake_vllm_config?.note || "",
    };
  }

  if (deployType === "multi_node" && nodeCount > 1) {
    // Every multi-node strategy renders the same shape: head + worker tabs.
    // Workers use --headless (TP/TEP) or --data-parallel-start-rank offset (DEP).
    const headArgs = buildArgs(null, "head");
    const workerArgs = buildArgs(null, "worker");
    return {
      deployType: "multi_node",
      nodeCount,
      headCommand: formatCommand(headArgs),
      workerCommand: formatCommand(workerArgs),
      headArgv: formatArgv(headArgs),
      workerArgv: formatArgv(workerArgs),
      env: buildEnv(null),
    };
  }

  const singleArgs = buildArgs(null, null);
  // Companion processes — helpers that must run alongside `vllm serve` on the
  // same node, rendered as PD-style tabs next to the serve command. Two
  // sources, same gating as their args so a companion never leaks onto an
  // excluded strategy:
  //   - enabled features declaring `companion: { label, description?, command }`
  //   - the active composing KV-offload option (e.g. LMCache's MP server)
  const companions = (enabledFeatures || []).flatMap((f) => {
    const feat = recipe.features?.[f];
    if (!feat?.companion?.command) return [];
    if (!isFeatureAllowedForStrategy(feat, strategyName)) return [];
    return [{
      feature: f,
      label: feat.companion.label || f,
      description: feat.companion.description || "",
      command: String(feat.companion.command).trimEnd(),
    }];
  });
  const kvCompanionOpt = taxonomy?.kv_offload?.[kvOffload];
  if (kvCompanionOpt?.companion?.command
      && isKvOffloadAllowedForStrategy(kvCompanionOpt, strategyName, strategy)) {
    companions.push({
      feature: `kv_offload:${kvOffload}`,
      label: kvCompanionOpt.companion.label || kvOffload,
      description: [
        kvCompanionOpt.companion.description || "",
        kvCompanionOpt.install ? `Requires: ${kvCompanionOpt.install}` : "",
      ].filter(Boolean).join(" "),
      command: String(kvCompanionOpt.companion.command).trimEnd(),
    });
  }
  return {
    deployType,
    command: formatCommand(singleArgs),
    argv: formatArgv(singleArgs),
    env: buildEnv(null),
    ...(companions.length ? { companions } : {}),
  };
}
