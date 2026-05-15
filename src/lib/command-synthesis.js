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
 * List hardware profiles compatible with a variant by precision constraint
 * only. VRAM is NOT a blocking constraint — users can scale out via multi-node
 * TP/DP, so any profile that satisfies the precision requirement is valid.
 */
export function listCompatibleHardware(hwProfiles, variant, recipe) {
  return Object.entries(hwProfiles)
    .filter(([id, p]) => isPrecisionCompatible(p, variant) && isHardwareSupported(recipe, id))
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
    ([id, p]) => matchesConstraint(p, constraint) && isHardwareSupported(recipe, id)
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
//
// When a CUDA map is in play, `cudaMap` is returned so the caller can pick by
// the user's `dockerCudaVariant` toggle instead of appending `-cu129`/`-cu130`.
export function computeDockerMeta(recipe, variant, hwProfile) {
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
      };
  const isAmd = hwProfile?.brand === "AMD";
  const isTpu = hwProfile?.generation === "tpu";
  const brandKey = isTpu ? "tpu" : isAmd ? "amd" : "nvidia";
  const override = variant?.docker_image || recipe.model?.docker_image;

  const isCudaMap = (v) =>
    v && typeof v === "object" && ("cu129" in v || "cu130" in v);

  let pinned = null;
  let cudaMap = null;
  if (typeof override === "string") {
    if (brandKey === "nvidia") pinned = override;
  } else if (override && typeof override === "object") {
    const isBrandKeyed = "nvidia" in override || "amd" in override || "tpu" in override;
    if (isBrandKeyed) {
      const brandValue = override[brandKey];
      if (typeof brandValue === "string") pinned = brandValue;
      else if (brandKey === "nvidia" && isCudaMap(brandValue)) cudaMap = brandValue;
    } else if (brandKey === "nvidia" && isCudaMap(override)) {
      cudaMap = override;
    }
  }

  const image = pinned || DEFAULT_IMAGE[brandKey];
  const gpuFlags = isTpu
    ? "--privileged --network host \\\n  -v /dev/shm:/dev/shm"
    : isAmd
      ? "--device=/dev/kfd --device=/dev/dri \\\n  --security-opt seccomp=unconfined --group-add video"
      : "--gpus all";
  return { image, gpuFlags, brandKey, isAmd, isTpu, pinned, cudaMap, nightlyRequired };
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

/**
 * Resolve a complete vllm serve command from recipe + user selections.
 *
 * Returns: { command, env, deployType } for single_node/multi_node,
 *          { prefillCommand, decodeCommand, routerConfig, env, deployType } for pd_cluster.
 */
export function resolveCommand(recipe, variantKey, strategyName, hwProfileId, enabledFeatures, strategies, taxonomy, advancedArgs = [], nodeCount = 1, pdNodes = null) {
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

  const modelId = variant.model_id || recipe.model?.model_id || "unknown";

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

    // 3. Strategy args + parallel size (grouped together so -tp/-dp sits next to -ep etc.)
    if (strategy.deploy_type !== "pd_cluster") {
      if (strategy.vllm_args) args.push(...strategy.vllm_args);
    } else if (roleOverride && strategy[roleOverride]?.vllm_args) {
      args.push(...strategy[roleOverride].vllm_args);
    }
    const parallelFlag = strategy.parallel_flag || "--tensor-parallel-size";
    const isMulti = strategy.deploy_type === "multi_node" && nodeCount > 1;
    const isPdMulti = strategy.deploy_type === "pd_cluster" && nodeCount > 1;

    if (isMulti && parallelFlag === "--data-parallel-size") {
      // Multi-node DEP: DP across all GPUs, each worker owns N local ranks.
      args.push("--data-parallel-size", String(totalGpus));
      args.push("--data-parallel-size-local", String(gpuCount));
      args.push("--data-parallel-address", "$HEAD_IP");
      if (nodeRole === "worker") {
        // Example worker = node 1 (start rank offset by one node's worth of GPUs).
        args.push("--data-parallel-start-rank", String(gpuCount));
      }
    } else if (isMulti && strategy.parallelism === "tp_pp") {
      // TP inside each node, PP across nodes. Cross-node traffic flows through
      // the PP stage boundaries only — much less bandwidth than pure TP across
      // nodes. Suited for very large models on commodity inter-node links.
      args.push("--tensor-parallel-size", String(gpuCount));
      args.push("--pipeline-parallel-size", String(nodeCount));
      args.push("--nnodes", String(nodeCount));
      args.push("--node-rank", nodeRole === "worker" ? "1" : "0");
      args.push("--master-addr", "$HEAD_IP");
      if (nodeRole === "worker") args.push("--headless");
    } else if (isMulti) {
      // Multi-node TP/TEP via vLLM multiprocessing (mp) backend:
      // TP spans all GPUs in the cluster; every node runs the same command,
      // varying only --node-rank and (for rank > 0) --headless.
      args.push("--tensor-parallel-size", String(totalGpus));
      args.push("--nnodes", String(nodeCount));
      args.push("--node-rank", nodeRole === "worker" ? "1" : "0");
      args.push("--master-addr", "$HEAD_IP");
      if (nodeRole === "worker") args.push("--headless");
    } else if (strategy.deploy_type === "pd_cluster") {
      // PD splits inference across separate prefill and decode pools.
      //
      // New model (per-role nodes + parallelism):
      //   pdNodes = { prefill: <int>, decode: <int> } — user-set node counts
      //   role config (strategy_overrides.pd_cluster.<role>):
      //     parallelism: "tp" | "dep"          (default: "tp")
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
      const parallelism = soRoleCfg.parallelism || roleCfg.parallelism || "tp";

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
        // TP pool. With >1 node, layer on vLLM mp multi-node flags.
        const roleParallelFlag =
          soRoleCfg.parallel_flag ||
          roleCfg.parallel_flag ||
          strategy.parallel_flag ||
          "--tensor-parallel-size";
        args.push(roleParallelFlag, String(Math.max(1, poolGpus)));
        if (rolePoolNodes > 1) {
          args.push("--nnodes", String(rolePoolNodes));
          args.push("--node-rank", "0");
          // TP master = node 0 of pool = NODE_1 (same naming as router endpoints).
          args.push("--master-addr", `$${roleKey.toUpperCase()}_NODE_1`);
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
      const featHo = feat.hardware_overrides?.[gen]
        || (isNvidia ? feat.hardware_overrides?.nvidia : null);
      const featArgs = featHo?.args ?? feat.args;
      if (featArgs) args.push(...featArgs);
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

    // Strategy env
    if (strategy.deploy_type !== "pd_cluster") {
      Object.assign(env, strategy.env || {});
    } else if (roleOverride && strategy[roleOverride]?.env) {
      Object.assign(env, strategy[roleOverride].env);
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
    // side channel etc). For DEP pools where the UI shows a non-zero rank's
    // command, point those values at NODE_{rank+1} so the rendered command
    // matches that physical node. TP pools always render the head node, so
    // NODE_1 is already correct.
    if (strategy.deploy_type === "pd_cluster" && roleOverride) {
      const raw = pdNodes ? pdNodes[roleOverride] : undefined;
      const pdRoleObj =
        typeof raw === "number" ? { nodes: raw }
        : (raw && typeof raw === "object") ? raw
        : {};
      const rank = pdRoleObj.rank || 0;
      if (rank > 0) {
        const oldVar = `$${roleOverride.toUpperCase()}_NODE_1`;
        const newVar = `$${roleOverride.toUpperCase()}_NODE_${rank + 1}`;
        for (const k of Object.keys(env)) {
          if (env[k] === oldVar) env[k] = newVar;
        }
      }
    }

    return env;
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
      const parallelism = soRoleCfg.parallelism || roleCfg.parallelism || "tp";
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
    const prefillEndpoints = Array.from(
      { length: Math.max(1, pMeta.nodes || 1) },
      (_, i) => `    --prefill http://$PREFILL_NODE_${i + 1}:8001 \\`,
    );
    const decodeEndpoints = Array.from(
      { length: Math.max(1, dMeta.nodes || 1) },
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
    return {
      deployType,
      nodeCount,
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
  return {
    deployType,
    command: formatCommand(singleArgs),
    argv: formatArgv(singleArgs),
    env: buildEnv(null),
  };
}
