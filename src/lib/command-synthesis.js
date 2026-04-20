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
 * in practice. MXFP4 weights ship for AMD CDNA 4 (MI355X) only.
 * AWQ/GPTQ/INT quants run on most NVIDIA+AMD hardware.
 */
const PRECISION_HARDWARE_CONSTRAINTS = {
  nvfp4: { brand: "NVIDIA", generation: "blackwell" },
  fp4: { brand: "NVIDIA", generation: "blackwell" },
  mxfp4: { brand: "AMD", generation: "cdna4" },
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
  if (!matchesConstraint(profile, constraint)) return false;
  // HuggingFace checkpoints under amd/ are published for ROCm / AITER flows.
  if (variant?.model_id?.startsWith("amd/") && profile?.brand !== "AMD") return false;
  return true;
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
 */
export function pickDefaultHardware(hwProfiles, variant) {
  const constraint = PRECISION_HARDWARE_CONSTRAINTS[variant?.precision];
  const compatible = Object.entries(hwProfiles).filter(([, p]) => matchesConstraint(p, constraint));

  if (constraint?.generation === "blackwell") {
    if (compatible.some(([id]) => id === "b200")) return "b200";
    if (compatible.some(([id]) => id === "gb200")) return "gb200";
  }
  if (constraint?.generation === "cdna4") {
    if (compatible.some(([id]) => id === "mi355x")) return "mi355x";
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
      // PD splits the node between prefill and decode.
      // - Single-node: 50/50 split, each role uses TP=gpuCount/2, pinned via
      //   CUDA_VISIBLE_DEVICES (set in buildEnv).
      // - Multi-node (2 nodes): one full node per role, TP=gpuCount per role.
      const tpPerRole = isPdMulti ? gpuCount : Math.floor(gpuCount / 2);
      args.push("--tensor-parallel-size", String(Math.max(1, tpPerRole)));
    } else {
      args.push(parallelFlag, String(gpuCount));
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
    const isNvidia = hwProfile?.brand === "NVIDIA";
    const ho = recipe.hardware_overrides?.[gen]
      || (isNvidia ? recipe.hardware_overrides?.nvidia : null);
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

    // PD: pin GPUs per role via CUDA_VISIBLE_DEVICES
    // - Single-node: first half (prefill) / second half (decode)
    // - Multi-node: each role owns a whole node, so list all its GPUs
    if (strategy.deploy_type === "pd_cluster" && roleOverride) {
      const multi = nodeCount > 1;
      if (multi) {
        const all = Array.from({ length: gpuCount }, (_, i) => i).join(",");
        env.CUDA_VISIBLE_DEVICES = all;
      } else {
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
    // first, then brand-wide `nvidia:` for NVIDIA GPUs.
    const envIsNvidia = hwProfile?.brand === "NVIDIA";
    const envHo = recipe.hardware_overrides?.[gen]
      || (envIsNvidia ? recipe.hardware_overrides?.nvidia : null);
    if (envHo?.extra_env) Object.assign(env, envHo.extra_env);

    return env;
  }

  function formatCommand(args) {
    const filtered = args.filter(Boolean);
    if (filtered.length === 0) return `vllm serve ${modelId}`;
    // Pair each --flag with its immediate value on the same line so the output
    // reads like the human-written command in the recipe guide, not
    // --flag\n value\n --flag\n value\n ...
    const lines = [];
    for (let i = 0; i < filtered.length; i++) {
      const cur = filtered[i];
      const next = filtered[i + 1];
      if (cur.startsWith("-") && next !== undefined && !next.startsWith("-")) {
        lines.push(`${cur} ${next}`);
        i++;
      } else {
        lines.push(cur);
      }
    }
    return `vllm serve ${modelId} \\\n  ${lines.join(" \\\n  ")}`;
  }

  const deployType = strategy.deploy_type || "single_node";

  if (deployType === "pd_cluster") {
    // Single-node PD splits one node 50/50 (prefill TP=gpuCount/2 + decode TP=gpuCount/2).
    // Multi-node PD dedicates one node per role (prefill TP=gpuCount + decode TP=gpuCount).
    // Either way the router sees 1 prefill + 1 decode endpoint — scale by replicating.
    const policy = strategy.router?.policy || "round_robin";
    // `--intra-node-data-parallel-size` is the DP size inside one pool.
    // Our default PD layout is TP-only per role (DP=1), so the router sees
    // a single DP rank per endpoint. Users running DP+EP should raise this
    // to match their `--data-parallel-size`.
    // Port convention: prefill on 8001, decode on 8002 (keeps them distinct
    // when co-located on a single node). The router listens on 30000.
    const routerLines = [
      `vllm-router --policy ${policy} \\`,
      `    --vllm-pd-disaggregation \\`,
      `    --prefill http://PREFILL_ADDR:8001 \\`,
      `    --decode http://DECODE_ADDR:8002 \\`,
      `    --host 127.0.0.1 \\`,
      `    --port 30000 \\`,
      `    --intra-node-data-parallel-size 1`,
    ];
    const routerCommand = routerLines.join("\n");

    return {
      deployType,
      nodeCount,
      prefill: {
        command: formatCommand(buildArgs("prefill", null)),
        env: buildEnv("prefill"),
      },
      decode: {
        command: formatCommand(buildArgs("decode", null)),
        env: buildEnv("decode"),
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
