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
export function resolveCommand(recipe, variantKey, strategyName, hwProfileId, enabledFeatures, strategies, taxonomy, advancedArgs = [], nodeCount = 1, pdNodes = null) {
  const variant = recipe.variants?.[variantKey] || recipe.variants?.default || {};
  const strategy = strategies[strategyName] || {};
  const hwProfile = taxonomy.hardware_profiles?.[hwProfileId] || {};
  const gen = normalizeGeneration(hwProfile.generation || hwProfile.gpu_generation);
  const gpuCount = typeof hwProfile.gpu_count === "number" ? hwProfile.gpu_count : 1;
  const totalGpus = gpuCount * Math.max(1, nodeCount);

  // Recipes can declare a TP size lower than the full node via
  // `strategy_overrides.single_node_tp.tp` (matches PD's per-role `tp:`
  // convention) so the generated single-node command matches the guide
  // instead of always fanning out to every GPU on the node. Clamped to
  // [1, gpu_count]. Only single_node_tp reads this — TEP/DEP require full
  // TP by topology, and multi-node is explicit scale-out.
  const declaredTp = recipe.strategy_overrides?.[strategyName]?.tp;
  const singleNodeTp =
    strategyName === "single_node_tp" && typeof declaredTp === "number" && declaredTp > 0
      ? Math.min(declaredTp, gpuCount)
      : gpuCount;

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
        args.push("--data-parallel-address", `$${roleKey.toUpperCase()}_DP_LEADER_IP`);
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
          args.push("--master-addr", `$${roleKey.toUpperCase()}_HEAD_IP`);
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
    // first, then brand-wide `nvidia:` for NVIDIA GPUs.
    const envIsNvidia = hwProfile?.brand === "NVIDIA";
    const envHo = recipe.hardware_overrides?.[gen]
      || (envIsNvidia ? recipe.hardware_overrides?.nvidia : null);
    if (envHo?.extra_env) Object.assign(env, envHo.extra_env);

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
    // Placeholder hostnames (PREFILL_NODE_1 … N, DECODE_NODE_1 … N) — deployers
    // substitute real IPs. 1-indexed in the placeholder so it matches what the
    // UI shows the user.
    const prefillEndpoints = Array.from(
      { length: Math.max(1, pMeta.nodes || 1) },
      (_, i) => `    --prefill http://PREFILL_NODE_${i + 1}:8001 \\`,
    );
    const decodeEndpoints = Array.from(
      { length: Math.max(1, dMeta.nodes || 1) },
      (_, i) => `    --decode http://DECODE_NODE_${i + 1}:8002 \\`,
    );
    // intra-node-data-parallel-size = max dp_local across the two pools for
    // DEP setups; 1 for pure-TP PD.
    const intraDp = Math.max(pMeta.dpLocal || 0, dMeta.dpLocal || 0, 1);
    const routerLines = [
      `vllm-router --policy ${policy} \\`,
      `    --vllm-pd-disaggregation \\`,
      ...prefillEndpoints,
      ...decodeEndpoints,
      `    --host 127.0.0.1 \\`,
      `    --port 30000 \\`,
      `    --intra-node-data-parallel-size ${intraDp}`,
    ];
    const routerCommand = routerLines.join("\n");

    return {
      deployType,
      nodeCount,
      prefill: {
        command: formatCommand(buildArgs("prefill", null)),
        env: buildEnv("prefill"),
        ...pMeta,
      },
      decode: {
        command: formatCommand(buildArgs("decode", null)),
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
