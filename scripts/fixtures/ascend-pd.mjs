// Synthetic, public-checkpoint-independent coverage of a single pool per role.
export function ascendPdRecipe() {
  return {
    hf_id: "test/Ascend-PD",
    meta: {
      title: "PD fixture", slug: "pd-fixture", provider: "test", description: "PD regression fixture",
      date_added: "2026-01-01", date_updated: "2026-01-01", tasks: ["text"],
      hardware: { ascend_950dt: "supported" },
    },
    model: { model_id: "test/Ascend-PD", architecture: "moe", parameter_count: "32B", min_vllm_version: "0.23.0", base_args: ["--max-model-len", "1024"] },
    variants: { default: { model_id: "test/Ascend-PD-W8A8", precision: "int8", vram_minimum_gb: 64, supported_hardware: ["ascend_950dt"] } },
    compatible_strategies: ["single_node_tp", "multi_node_dep", "pd_cluster"],
    guide: "Synthetic deployment fixture.",
    features: { spec_decoding: { description: "MTP", args: ["--speculative-config", '{"method":"mtp","num_speculative_tokens":2}'] } },
    presets: {
      ascend_pd: {
        label: "Ascend PD", hardware: "ascend_950dt", strategy: "pd_cluster",
        image: "example/ascend:fixture", select: { pd_router: "ascend", pd_connector: "mooncakev2", frontend: "python" },
        roles: {
          prefill: { nodes: 4, parallelism: "tp", tp: 8, pp: 4, remove_args: ["--max-model-len"], env: { HCCL_IF_IP: "$PREFILL_NODE_1", VLLM_HOST_IP: "$PREFILL_NODE_1" } },
          decode: { nodes: 4, parallelism: "dep", tp: 4, pp: 1, json_args: { "--speculative-config": { num_speculative_tokens: 3 } }, env: { HCCL_IF_IP: "$DECODE_NODE_1", VLLM_HOST_IP: "$DECODE_NODE_1" } },
        },
      },
    },
  };
}

export const ascendPdPools = (rank = 0) => ({
  prefill: { nodes: 4, parallelism: "tp", rank },
  decode: { nodes: 4, parallelism: "dep", rank },
});
