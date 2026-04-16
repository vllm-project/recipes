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
 * Given a recipe and hardware profile, recommend the best strategy.
 */
export function recommendStrategy(recipe, hwProfile) {
  const compatible = recipe.compatible_strategies || [];
  const arch = recipe.model?.architecture;
  const gpuCount = typeof hwProfile.gpu_count === "number" ? hwProfile.gpu_count : 8;
  const multiNode = hwProfile.multi_node;

  if (multiNode) {
    if (arch === "moe" && compatible.includes("multi_node_dep")) return "multi_node_dep";
    if (arch === "moe" && compatible.includes("multi_node_tep")) return "multi_node_tep";
    if (compatible.includes("multi_node_tp")) return "multi_node_tp";
  }

  if (arch === "moe" && gpuCount >= 4) {
    if (compatible.includes("single_node_dep")) return "single_node_dep";
    if (compatible.includes("single_node_tep")) return "single_node_tep";
  }

  if (compatible.includes("single_node_tp")) return "single_node_tp";
  return compatible[0] || "single_node_tp";
}

/**
 * Filter hardware profiles to those with enough VRAM for a given variant.
 */
export function filterHardwareByVram(hwProfiles, variant) {
  const minVram = variant?.vram_minimum_gb || 0;
  return Object.entries(hwProfiles)
    .filter(([, p]) => {
      const vram = typeof p.vram_gb === "number" ? p.vram_gb : 0;
      return vram >= minVram || p.multi_node;
    })
    .map(([id]) => id);
}

/**
 * Resolve a complete vllm serve command from recipe + user selections.
 *
 * Returns: { command, env, deployType } for single_node/multi_node,
 *          { prefillCommand, decodeCommand, routerConfig, env, deployType } for pd_cluster.
 */
export function resolveCommand(recipe, variantKey, strategyName, hwProfileId, enabledFeatures, strategies, taxonomy) {
  const variant = recipe.variants?.[variantKey] || recipe.variants?.default || {};
  const strategy = strategies[strategyName] || {};
  const hwProfile = taxonomy.hardware_profiles?.[hwProfileId] || {};
  const gen = normalizeGeneration(hwProfile.gpu_generation);
  const gpuCount = typeof hwProfile.gpu_count === "number" ? hwProfile.gpu_count : 1;

  const modelId = variant.model_id || recipe.model?.model_id || "unknown";

  // Helper to merge args
  function buildArgs(roleOverride) {
    const args = [];

    // 1. Base args
    if (recipe.model?.base_args) args.push(...recipe.model.base_args);

    // 2. Variant extra args
    if (variantKey !== "default" && variant.extra_args) args.push(...variant.extra_args);

    // 3. Strategy args (for non-pd_cluster, or role-specific for pd_cluster)
    if (strategy.deploy_type !== "pd_cluster") {
      if (strategy.vllm_args) args.push(...strategy.vllm_args);
    } else if (roleOverride && strategy[roleOverride]?.vllm_args) {
      args.push(...strategy[roleOverride].vllm_args);
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

    // 6. Features
    for (const f of enabledFeatures) {
      const feat = recipe.features?.[f];
      if (feat?.args) args.push(...feat.args);
    }

    // 7. TP size
    args.push("--tensor-parallel-size", String(gpuCount));

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
    return {
      deployType,
      prefillCommand: formatCommand(buildArgs("prefill")),
      decodeCommand: formatCommand(buildArgs("decode")),
      routerConfig: strategy.router || { policy: "round_robin" },
      prefillEnv: buildEnv("prefill"),
      decodeEnv: buildEnv("decode"),
    };
  }

  return {
    deployType,
    command: formatCommand(buildArgs(null)),
    env: buildEnv(null),
  };
}
