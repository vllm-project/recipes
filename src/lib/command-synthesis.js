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
export function recommendStrategy(recipe, hwProfile) {
  const compatible = recipe.compatible_strategies || [];
  const multiNode = hwProfile.multi_node;

  if (multiNode && compatible.includes("multi_node_tp")) return "multi_node_tp";
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
 * Filter hardware profiles to those with enough VRAM for a given variant.
 * Does NOT filter by precision constraints — those are rendered as disabled
 * pills in the UI instead of being hidden.
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
 * Given a variant, pick the preferred default hardware:
 * - If variant requires Blackwell (e.g., NVFP4), prefer b200
 * - Otherwise, the first hardware that both fits VRAM and matches precision constraint
 */
export function pickDefaultHardware(hwProfiles, variant) {
  const constraint = PRECISION_HARDWARE_CONSTRAINTS[variant?.precision];
  const minVram = variant?.vram_minimum_gb || 0;
  const compatible = Object.entries(hwProfiles).filter(([, p]) => {
    const vram = typeof p.vram_gb === "number" ? p.vram_gb : 0;
    return (vram >= minVram || p.multi_node) && matchesConstraint(p, constraint);
  });

  // Prefer B200 for Blackwell-constrained variants
  if (constraint?.generation === "blackwell") {
    const b200 = compatible.find(([id]) => id === "b200");
    if (b200) return "b200";
    const gb200 = compatible.find(([id]) => id === "gb200");
    if (gb200) return "gb200";
  }

  // Otherwise: prefer H200 as canonical default if it fits, else smallest that fits
  if (compatible.some(([id]) => id === "h200")) return "h200";
  compatible.sort((a, b) => (a[1].vram_gb || 0) - (b[1].vram_gb || 0));
  return compatible[0]?.[0] || "h200";
}

/**
 * Resolve a complete vllm serve command from recipe + user selections.
 *
 * Returns: { command, env, deployType } for single_node/multi_node,
 *          { prefillCommand, decodeCommand, routerConfig, env, deployType } for pd_cluster.
 */
export function resolveCommand(recipe, variantKey, strategyName, hwProfileId, enabledFeatures, strategies, taxonomy, advancedArgs = []) {
  const variant = recipe.variants?.[variantKey] || recipe.variants?.default || {};
  const strategy = strategies[strategyName] || {};
  const hwProfile = taxonomy.hardware_profiles?.[hwProfileId] || {};
  const gen = normalizeGeneration(hwProfile.generation || hwProfile.gpu_generation);
  const gpuCount = typeof hwProfile.gpu_count === "number" ? hwProfile.gpu_count : 1;

  const modelId = variant.model_id || recipe.model?.model_id || "unknown";

  // Helper to merge args
  function buildArgs(roleOverride) {
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
    args.push(parallelFlag, String(gpuCount));

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
