// Global Advanced-row toggles of the command builder (`?advanced=<id>,…`);
// a recipe adds its own under top-level `advanced_options:`. Shared with the
// server-side renderer so a linked `advanced=` renders the same flags.

export const ADVANCED_OPTIONS = [
  {
    id: "max_batched_8k",
    label: "max-num-batched-tokens = 8192",
    description: "Tunable batch budget; 8192 is a common sweet spot",
    args: ["--max-num-batched-tokens", "8192"],
  },
  {
    id: "max_num_seqs_256",
    label: "max-num-seqs = 256",
    description: "Max concurrent sequences per batch; lower for latency, higher for throughput",
    args: ["--max-num-seqs", "256"],
  },
  {
    id: "gpu_mem_095",
    label: "gpu-memory-utilization = 0.95",
    description: "Push KV cache further; use with caution on shared GPUs",
    args: ["--gpu-memory-utilization", "0.95"],
  },
  {
    id: "max_model_len_auto",
    label: "max-model-len = auto",
    description: "Auto-size context window to what KV cache can hold on your hardware",
    args: ["--max-model-len", "auto"],
  },
  {
    id: "dcp_8",
    label: "decode-context-parallel-size = 8",
    description:
      "Shard the KV cache at decode time across TP ranks. MLA-attention models only (DeepSeek, Kimi-K2). Max DCP = tensor-parallel-size ÷ num_kv_heads.",
    args: ["--decode-context-parallel-size", "8"],
    gatedBy: (recipe) => recipe?.model?.supports_dcp === true,
  },
  {
    id: "no_flashinfer_autotune",
    label: "no-enable-flashinfer-autotune",
    description: "Skip FlashInfer autotuning at startup (faster startup, may lose autotuned perf)",
    args: ["--no-enable-flashinfer-autotune"],
    gatedBy: (recipe) => recipe?.model?.flashinfer_autotune === true,
  },
  {
    id: "ep_weight_filter",
    label: "enable-ep-weight-filter",
    description: "Skip loading expert weights that don't belong to this EP rank — speeds up weight loading for large MoE models",
    args: ["--enable-ep-weight-filter"],
    gatedBy: (_recipe, activeStrategy) => /(?:^|_)(?:tep|dep)$/.test(activeStrategy || ""),
  },
];
