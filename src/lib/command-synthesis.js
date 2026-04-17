/**
 * Core command synthesis: Recipe + Variant + Strategy + Hardware + Features → vllm serve command.
 * Pure functions, no I/O.
 */

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
 * Given a recipe and hardware profile, recommend the default strategy.
 *
 * Tensor Parallel is the default for every model — it's the most widely
 * tested, works for both dense and MoE. TEP / DEP / PD-cluster are
 * advanced strategies that users can opt into explicitly.
 */
export function recommendStrategy(recipe, hwProfile, nodeCount = 1) {
  const compatible = recipe.compatible_strategies || [];
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
  fp4:   { brand: "NVIDIA", generation: "blackwell" },
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
 * List hardware profiles compatible with a variant by precision constraint
 * only. VRAM is NOT a blocking constraint — users can scale out via multi-node
 * TP/DP, so any profile that satisfies the precision requirement is valid.
 */
export function listCompatibleHardware(hwProfiles, variant) {
  return Object.entries(hwProfiles)
    .filter(([, p]) => isPrecisionCompatible(p, variant))
    .map(([id]) => id);
}

/**
 * Given a variant, pick the preferred default hardware:
 * - If variant requires Blackwell (e.g., NVFP4), prefer B200 then GB200
 * - Otherwise H200 is the canonical default
 */
export function pickDefaultHardware(hwProfiles, variant) {
  const constraint = PRECISION_HARDWARE_CONSTRAINTS[variant?.precision];
  const compatible = Object.entries(hwProfiles).filter(([, p]) => matchesConstraint(p, constraint));

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

/**
 * Resolve a complete vllm serve command from recipe + user selections.
 *
 * Returns: { command, env, deployType } for single_node/multi_node,
 *          { prefillCommand, decodeCommand, routerConfig, env, deployType } for pd_cluster.
 */
export function resolveCommand(recipe, variantKey, strategyName, hwProfileId, enabledFeatures, strategies, taxonomy, advancedArgs = [], nodeCount = 1) {
  const variant = recipe.variants?.[variantKey] || recipe.variants?.default || {};
  const strategy = strategies[strategyName] || {};
  const hwProfile = taxonomy.hardware_profiles?.[hwProfileId] || {};
  const gen = normalizeGeneration(hwProfile.generation || hwProfile.gpu_generation);
  const gpuCount = typeof hwProfile.gpu_count === "number" ? hwProfile.gpu_count : 1;
  const totalGpus = gpuCount * Math.max(1, nodeCount);

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
      // PD worker is a single `vllm serve` process per GPU; TP=1 is the default.
      // Multi-node PD extends each role (prefill/decode) across nodes via the
      // mp backend — same --nnodes/--node-rank pattern as multi-node TP.
      args.push("--tensor-parallel-size", "1");
      if (isPdMulti) {
        args.push("--nnodes", String(nodeCount));
        args.push("--node-rank", nodeRole === "worker" ? "1" : "0");
        const roleHost = roleOverride === "decode" ? "$DECODE_HOST_IP" : "$PREFILL_HOST_IP";
        args.push("--master-addr", roleHost);
        if (nodeRole === "worker") args.push("--headless");
      }
    } else {
      args.push(parallelFlag, String(gpuCount));
    }

    // 4. Strategy overrides from recipe
    const so = recipe.strategy_overrides?.[strategyName];
    if (so) {
      if (roleOverride && so[roleOverride]?.extra_args) {
        args.push(...so[roleOverride].extra_args);
      } else if (!roleOverride && so.extra_args) {
        args.push(...so.extra_args);
      }
    }

    // 5. Hardware overrides
    const ho = recipe.hardware_overrides?.[gen];
    if (ho?.extra_args) args.push(...ho.extra_args);

    // 6. Advanced tuning args (from UI's Advanced panel)
    if (advancedArgs && advancedArgs.length) args.push(...advancedArgs);

    // 7. Features last — tool_calling, reasoning, mtp, etc.
    for (const f of enabledFeatures || []) {
      const feat = recipe.features?.[f];
      if (feat?.args) args.push(...feat.args);
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

    // Strategy overrides env
    const so = recipe.strategy_overrides?.[strategyName];
    if (so) {
      if (so.env) Object.assign(env, so.env);
      if (roleOverride && so[roleOverride]?.env) Object.assign(env, so[roleOverride].env);
      if (!roleOverride && so.extra_env) Object.assign(env, so.extra_env);
    }

    // Hardware overrides env
    const ho = recipe.hardware_overrides?.[gen];
    if (ho?.extra_env) Object.assign(env, ho.extra_env);

    return env;
  }

  function formatCommand(args) {
    const filtered = args.filter(Boolean);
    if (filtered.length === 0) return `vllm serve ${modelId}`;
    return `vllm serve ${modelId} \\\n  ${filtered.join(" \\\n  ")}`;
  }

  const deployType = strategy.deploy_type || "single_node";

  if (deployType === "pd_cluster") {
    const multi = nodeCount > 1;

    // Router command — assumes one `vllm serve` per GPU (DP=gpuCount per node).
    // Multi-node example follows a typical asymmetric layout (1 prefill node, 1 decode
    // node; users scale decode wider in practice). Single-node collapses to 1+1.
    const prefillCount = multi ? gpuCount : 1;
    const decodeCount = multi ? gpuCount : 1;
    const policy = strategy.router?.policy || "round_robin";
    const routerLines = [
      `vllm-router --policy ${policy} \\`,
      `    --vllm-pd-disaggregation \\`,
      ...Array.from({ length: prefillCount }, (_, i) =>
        `    --prefill http://PREFILL_ADDR${i + 1}:8000 \\`
      ),
      ...Array.from({ length: decodeCount }, (_, i) =>
        `    --decode http://DECODE_ADDR${i + 1}:8000 \\`
      ),
      `    --host 127.0.0.1 \\`,
      `    --port 30000 \\`,
      `    --intra-node-data-parallel-size ${gpuCount}`,
    ];
    const routerCommand = routerLines.join("\n");

    return {
      deployType,
      nodeCount,
      prefill: {
        head: formatCommand(buildArgs("prefill", "head")),
        worker: multi ? formatCommand(buildArgs("prefill", "worker")) : null,
        env: buildEnv("prefill"),
      },
      decode: {
        head: formatCommand(buildArgs("decode", "head")),
        worker: multi ? formatCommand(buildArgs("decode", "worker")) : null,
        env: buildEnv("decode"),
      },
      router: {
        command: routerCommand,
        install: "uv pip install vllm-router",
      },
      routerConfig: strategy.router || { policy: "round_robin" },
      // Legacy fields kept for any lingering consumer
      prefillCommand: formatCommand(buildArgs("prefill", "head")),
      decodeCommand: formatCommand(buildArgs("decode", "head")),
      prefillEnv: buildEnv("prefill"),
      decodeEnv: buildEnv("decode"),
    };
  }

  if (deployType === "multi_node" && nodeCount > 1) {
    // Every multi-node strategy renders the same shape: head + worker tabs.
    // Workers use --headless (TP/TEP) or --data-parallel-start-rank offset (DEP).
    return {
      deployType: "multi_node",
      nodeCount,
      headCommand: formatCommand(buildArgs(null, "head")),
      workerCommand: formatCommand(buildArgs(null, "worker")),
      env: buildEnv(null),
    };
  }

  return {
    deployType,
    command: formatCommand(buildArgs(null, null)),
    env: buildEnv(null),
  };
}
