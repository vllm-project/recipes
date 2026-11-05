// Standalone renderer inputs; tests must not depend on a source PR's recipe.
export function createAscendPdRecipe() {
  const serviceArgs = ["--max-model-len", "262144", "--max-num-seqs", "32", "--gpu-memory-utilization", "0.80", "--enable-prefix-caching"];
  const roleOverride = (prefill) => ({
    remove_args: ["--data-parallel-hybrid-lb", "--no-enable-prefix-caching", "--disable-flashinfer-autotune"],
    extra_args: [...serviceArgs, ...(prefill
      ? ["--data-parallel-size", "1", "--distributed-executor-backend", "mp", "--master-port", "41000", "--max-num-batched-tokens", "16384", "--enforce-eager"]
      : ["--max-num-batched-tokens", "768", "--block-size", "768", "--cp-kv-cache-interleave-size", "768"])],
    extra_env: { VLLM_USE_V2_MODEL_RUNNER: "1", HCCL_OP_EXPANSION_MODE: "CCU_SCHED", HCCL_BUFFSIZE: "800",
      HCCL_IF_IP: "$LOCAL_IP", HCCL_SOCKET_IFNAME: "$IFACE_NAME", GLOO_SOCKET_IFNAME: "$IFACE_NAME",
      ...(prefill ? { VLLM_PP_LAYER_PARTITION: "24,24,24,21" } : {}) },
    feature_overrides: { spec_decoding: { args: ["--speculative-config", JSON.stringify({
      method: "dspark", model: "Inferact/Kimi-K3-DSpark-Block5", num_speculative_tokens: 7,
      draft_tensor_parallel_size: 8, enforce_eager: prefill,
    })] } },
  });
  return {
    guide: "# Test deployment\n\nSynthetic inputs for renderer regression tests.",
    meta: { title: "Ascend PD fixture", provider: "test", description: "Renderer test fixture", date_added: "2026-01-01", date_updated: "2026-01-01", tasks: ["text"],
      hardware: { ascend_950dt: "supported", ascend_910c: "supported" },
      default_variant: { ascend_950dt: "ascend_mxfp4", ascend_910c: "ascend_w4a8" } },
    model: { model_id: "test/Ascend-PD", architecture: "moe", parameter_count: "1T", context_length: 262144,
      docker_image: "example/vllm:fixture", base_args: ["--trust-remote-code", "--data-parallel-hybrid-lb", "--no-enable-prefix-caching", "--disable-flashinfer-autotune"] },
    variants: {
      default: { precision: "bf16", vram_minimum_gb: 128, hardware: { ascend_950dt: "unsupported" } },
      fp8: { precision: "fp8", vram_minimum_gb: 64, model_id: "test/Ascend-PD-FP8", hardware: { ascend_950dt: "unsupported" } },
      ascend_mxfp4: { precision: "mxfp4", vram_minimum_gb: 1711, model_id: "test/Ascend-PD-MXFP4",
        precision_hardware: { brand: "Huawei", generation: "npu" },
        model_url: "https://www.modelscope.cn/models/test/Ascend-PD-MXFP4", supported_hardware: ["ascend_950dt"],
        extra_args: ["--quantization", "ascend"], extra_env: { VLLM_USE_MODELSCOPE: "true" }, docker_image: "example/ascend:fixture" },
      ascend_w4a8: { precision: "int4", vram_minimum_gb: 1711, model_id: "test/Ascend-PD-W4A8", supported_hardware: ["ascend_910c"], extra_args: ["--quantization", "ascend"] },
    },
    features: { spec_decoding: { description: "Test draft", args: ["--speculative-config", '{"method":"dspark","model":"test/Draft","num_speculative_tokens":7}'] } },
    compatible_strategies: ["single_node_tp", "multi_node_tp", "multi_node_tep", "multi_node_dep", "multi_node_tp_dp", "pd_cluster"],
    default_strategy: "single_node_tp",
    strategy_overrides: { pd_cluster: {
      connectors: { ascend_950dt: ["mooncakev2"] },
      prefill: { nodes: { ascend_950dt: 4 }, nodes_per_instance: { ascend_950dt: 4 }, parallelism: { ascend_950dt: "tp" },
        tp: { ascend_950dt: 8 }, pp: { ascend_950dt: 4 }, hardware_overrides: { ascend_950dt: roleOverride(true) } },
      decode: { nodes: { ascend_950dt: 4 }, nodes_per_instance: { ascend_950dt: 4 }, parallelism: { ascend_950dt: "dep" },
        tp: { ascend_950dt: 8 }, pp: { ascend_950dt: 1 }, hardware_overrides: { ascend_950dt: roleOverride(false) } },
    } },
  };
}

export function createMixedRecipe() {
  const recipe = createAscendPdRecipe();
  recipe.strategy_overrides.multi_node_tp_dp = { hardware_overrides: {
    ascend_910c: { worker_args: ["--headless"] }, ascend_950dt: { worker_args: ["--headless"] },
  } };
  return recipe;
}
