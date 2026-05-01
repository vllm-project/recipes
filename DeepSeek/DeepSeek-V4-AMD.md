# DeepSeek-V4 on AMD (ROCm) Usage Guide

This page is aligned with the DeepSeek-V4-Pro recipe layout on recipes.vllm.ai and
captures the AMD MI355X validated settings from [vllm-project/vllm#40871](https://github.com/vllm-project/vllm/pull/40871).

## Overview

DeepSeek-V4-Pro is the flagship of the V4 preview family: a 1.6T-total / 49B-active
Mixture-of-Experts model. It pairs a **hybrid attention stack** — Compressed Sparse
Attention (CSA) + Heavily Compressed Attention (HCA) — with **Manifold-Constrained
Hyper-Connections (mHC)** to reach 27% of V3.2's per-token inference FLOPs and 10% of
V3.2's KV cache at 1M context. Pre-trained on 32T+ tokens with the **Muon optimizer**
for faster convergence; post-training is a two-stage pipeline (domain-specific expert
cultivation + unified consolidation via on-policy distillation).

Checkpoint is **FP4+FP8 mixed**: MoE expert weights are stored in FP4 while the
remaining (attention / norm / router) params stay in FP8.

## Reasoning modes

The chat template exposes three reasoning-effort modes:

- **Non-think** — fast, intuitive responses.
- **Think High** — explicit chain-of-thought for complex problem-solving and planning.
- **Think Max** — maximum reasoning effort; requires `--max-model-len >= 393216`
  (384K tokens) to avoid truncation.

Recommended sampling: `temperature = 1.0`, `top_p = 1.0`.

### OpenAI Client Example

For DeepSeek-V4, keep reasoning controls in `chat_template_kwargs`, as it exposes a
custom **Think Max** mode via `"reasoning_effort": "max"`.

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
model = "deepseek-ai/DeepSeek-V4-Pro"
messages = [{"role": "user", "content": "What is 17*19? Return only the final integer."}]

# Non-think
resp = client.chat.completions.create(
    model=model,
    messages=messages,
)

# Think High
resp = client.chat.completions.create(
    model=model,
    messages=messages,
    extra_body={
        "chat_template_kwargs": {
            "thinking": True,
            "reasoning_effort": "high",
        },
    },
)

# Think Max
resp = client.chat.completions.create(
    model=model,
    messages=messages,
    extra_body={
        "chat_template_kwargs": {
            "thinking": True,
            "reasoning_effort": "max",
        },
    },
)
```

## Recommended deployments

- **B300 (8× GPU)**: single-node DP + EP with `--data-parallel-size 8`.
- **H200 (8× GPU)**: DP + EP with `--data-parallel-size 8`. Context is capped at
  800K tokens (`--max-model-len 800000`) to leave KV headroom with dense params
  replicated across ranks — applies to both single-node and multi-node H200.
- **MI355X (8× GPU)**: validated with ROCm + AITER
  (`VLLM_ROCM_USE_AITER=1`, `VLLM_ROCM_USE_AITER_LINEAR=1`), `--moe-backend triton_unfused`,
  `--gpu-memory-utilization 0.6`, `--max-num-seqs 128`,
  `--max-num-batched-tokens 8192`, and `--distributed-executor-backend mp`.
- **GB200 NVL4 (4× GPU per tray)**: the ~960 GB mixed-precision checkpoint does not
  fit on one tray; run multi-node DP + EP across **2 trays** (8 GPUs total) with
  `--data-parallel-size 8`. Pick the "Multi-Node" tab and set nodes to 2.

## Feature matrix

The table below is a static equivalent of the interactive matrix shown on
recipes.vllm.ai (hardware / variant / strategy / features).

| Model | Hardware | Variant | Recommended strategies | Tool calling | Reasoning | Spec decoding |
| --- | --- | --- | --- | --- | --- | --- |
| DeepSeek-V4-Pro | MI355X (8x288GB) | FP8 (~960GB) | Tensor+Expert Parallel, Data+Expert Parallel | Yes (`deepseek_v4`) | Yes (`deepseek_v4`) | Yes (`mtp`) |
| DeepSeek-V4-Flash | MI355X (8x288GB) | FP8 (~170GB) | Tensor+Expert Parallel, Data+Expert Parallel | Yes (`deepseek_v4`) | Yes (`deepseek_v4`) | Yes (`mtp`) |

### MI355X recommended presets

| Model | TP | Max num seqs | Max batched tokens | GPU memory utilization | Key ROCm env |
| --- | --- | ---: | ---: | ---: | --- |
| DeepSeek-V4-Pro | 8 | 128 | 8192 | 0.6 | `VLLM_ROCM_USE_AITER=1`, `VLLM_ROCM_USE_AITER_LINEAR=1` |
| DeepSeek-V4-Flash | 4 | 16 | 1024 | 0.35 | `VLLM_ROCM_USE_AITER=1` |

### Feature toggles

| Feature | Server args |
| --- | --- |
| Tool Calling | `--tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 --enable-auto-tool-choice` |
| Reasoning | `--reasoning-parser deepseek_v4` |
| Spec Decoding | `--speculative-config '{"method":"mtp","num_speculative_tokens":1}'` (start) / `2` (tune) |

## AMD validation command snippets

### DeepSeek-V4-Pro (MI355X, TP=8)

```bash
export HF_HOME=/data/huggingface-cache
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_LINEAR=1

vllm serve /home/models/DeepSeek-V4-Pro \
  --host localhost \
  --port 8001 \
  --dtype auto \
  --kv-cache-dtype fp8 \
  --tensor-parallel-size 8 \
  --max-num-seqs 128 \
  --max-num-batched-tokens 8192 \
  --distributed-executor-backend mp \
  --trust-remote-code \
  --gpu-memory-utilization 0.6 \
  --moe-backend triton_unfused \
  --tokenizer-mode deepseek_v4 \
  --reasoning-parser deepseek_v4 \
  --async-scheduling \
  --enforce-eager
```

### DeepSeek-V4-Flash (MI355X, TP=4)

```bash
export HF_HOME=/data/huggingface-cache
export VLLM_ROCM_USE_AITER=1

vllm serve /home/models/DeepSeek-V4-Flash \
  --host localhost \
  --port 8001 \
  --dtype auto \
  --tensor-parallel-size 4 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 1024 \
  --distributed-executor-backend mp \
  --trust-remote-code \
  --gpu-memory-utilization 0.35 \
  --moe-backend triton_unfused \
  --tokenizer-mode deepseek_v4 \
  --async-scheduling \
  --enforce-eager
```

