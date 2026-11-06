// Synthetic renderer inputs, not a model deployment recommendation. Keep model
// tuning out of this fixture: each test declares only the contract it exercises.
export function createAscendPdRecipe({
  modelId = "test/Moe-PD", architecture = "moe", hardware = "ascend_950dt",
  prefill = { nodes: 4, nodes_per_instance: 4, parallelism: "tp", tp: 8, pp: 4 },
  decode = { nodes: 4, nodes_per_instance: 4, parallelism: "dep", tp: 8, pp: 1 },
  speculation = "external",
} = {}) {
  const specArgs = speculation === "mtp"
    ? { method: "mtp", num_speculative_tokens: 1 }
    : { method: "eagle3", model: "test/Draft", num_speculative_tokens: 2 };
  const role = (layout, isPrefill) => ({
    ...Object.fromEntries(Object.entries(layout).map(([key, value]) => [key, { [hardware]: value }])),
    hardware_overrides: { [hardware]: {
      remove_args: ["--data-parallel-hybrid-lb"],
      extra_args: layout.parallelism === "dep" ? []
        : ["--data-parallel-size", "1", "--distributed-executor-backend", "mp", "--master-port", "41000"],
      extra_env: { HCCL_IF_IP: "$LOCAL_IP", HCCL_SOCKET_IFNAME: "$IFACE_NAME", GLOO_SOCKET_IFNAME: "$IFACE_NAME" },
      ...(speculation ? { feature_overrides: { spec_decoding: { args: ["--speculative-config",
        JSON.stringify({ ...specArgs, draft_tensor_parallel_size: layout.tp, enforce_eager: isPrefill })] } } } : {}),
    } },
  });
  return {
    guide: "Synthetic inputs for renderer regression tests.",
    meta: { title: "Ascend PD fixture", provider: "test", description: "Renderer test fixture",
      date_added: "2026-01-01", date_updated: "2026-01-01", tasks: ["text"], hardware: { [hardware]: "supported" } },
    model: { model_id: modelId, architecture, parameter_count: "1B", context_length: 8192,
      base_args: ["--data-parallel-hybrid-lb"] },
    variants: { default: { precision: "bf16", vram_minimum_gb: 4, supported_hardware: [hardware],
      hardware_overrides: { [hardware]: { docker_image: "example/ascend:fixture", docker_serve_command: "vllm serve" } } } },
    ...(speculation ? { features: { spec_decoding: { description: "Synthetic speculation example",
      args: ["--speculative-config", JSON.stringify(specArgs)] } } } : {}),
    compatible_strategies: ["pd_cluster"],
    default_strategy: "pd_cluster",
    strategy_overrides: { pd_cluster: {
      connectors: { [hardware]: ["mooncakev2"] }, prefill: role(prefill, true), decode: role(decode, false),
    } },
  };
}

// Vary architecture, checkpoint selection, speculation, device count and
// independent role layouts; renaming one fixed topology is not this coverage.
export function ascendPdCases() {
  return [
    { hardware: "ascend_950dt", variant: "default", recipe: createAscendPdRecipe({
      modelId: "test/Dense-PD", architecture: "dense", speculation: null,
      prefill: { nodes: 2, nodes_per_instance: 1, parallelism: "tp", tp: 4, pp: 1 },
      decode: { nodes: 3, nodes_per_instance: 1, parallelism: "tp", tp: 8, pp: 1 },
    }) },
    { hardware: "ascend_950dt", variant: "default", recipe: createAscendPdRecipe({
      modelId: "test/Moe-MTP-PD", speculation: "mtp",
      prefill: { nodes: 4, nodes_per_instance: 2, parallelism: "tp", tp: 8, pp: 2 },
      decode: { nodes: 6, nodes_per_instance: 3, parallelism: "dep", tp: 4, pp: 1 },
    }) },
    { hardware: "ascend_910c", variant: "quantized", recipe: (() => {
      const recipe = createAscendPdRecipe({ modelId: "test/Moe-External-PD", hardware: "ascend_910c",
        prefill: { nodes: 4, nodes_per_instance: 2, parallelism: "dep", tp: 8, pp: 1 },
        decode: { nodes: 2, nodes_per_instance: 1, parallelism: "dep", tp: 4, pp: 1 },
      });
      recipe.variants.quantized = { ...structuredClone(recipe.variants.default), precision: "int8",
        model_id: "test/Moe-External-PD-INT8", extra_args: ["--quantization", "ascend"] };
      return recipe;
    })() },
  ];
}
