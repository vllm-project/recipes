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

## Recommended deployments

- **MI355X (8× GPU)**: validated with ROCm + AITER
  (`VLLM_ROCM_USE_AITER=1`, `VLLM_ROCM_USE_AITER_LINEAR=1`), `--moe-backend triton_unfused`,
  `--gpu-memory-utilization 0.6`, `--max-num-seqs 128`,
  `--max-num-batched-tokens 8192`, and `--distributed-executor-backend mp`.

## Feature matrix

The table below is a static equivalent of the interactive matrix shown on
recipes.vllm.ai (hardware / variant / strategy / features).

| Model | Hardware | Variant | Recommended strategies | Tool calling | Reasoning | Spec decoding |
| --- | --- | --- | --- | --- | --- | --- |
| [DeepSeek-V4-Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) | MI355X (8x288GB) | FP8 (~960GB) | Tensor Parallel (TP) | Yes (`deepseek_v4`) | Yes (`deepseek_v4`) | No (`false`) |
| [DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) | MI355X (8x288GB) | FP8 (~170GB) | Tensor Parallel (TP) | Yes (`deepseek_v4`) | Yes (`deepseek_v4`) | No (`false`) |

### MI355X recommended presets

| Model | TP | Max num seqs | Max batched tokens | GPU memory utilization | Key ROCm env |
| --- | --- | ---: | ---: | ---: | --- |
| [DeepSeek-V4-Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) | 8 | 128 | 8192 | 0.6 | `VLLM_ROCM_USE_AITER=1`, `VLLM_ROCM_USE_AITER_LINEAR=1` |
| [DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) | 4 | 16 | 1024 | 0.35 | `VLLM_ROCM_USE_AITER=1` |

### Feature toggles

| Feature | Server args |
| --- | --- |
| Tool Calling | `--tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 --enable-auto-tool-choice` |
| Reasoning | `--reasoning-parser deepseek_v4` |
| Spec Decoding | Disabled (`false`) |

## DeepSeek-V4-Pro validation (MI355X, TP=8)

### 1) Serve command

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

### 2) Smoke test (single request)

```bash
curl -s http://localhost:8001/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is 2+2? Return only the final integer.",
    "model": "/home/models/DeepSeek-V4-Pro",
    "max_tokens": 16,
    "temperature": 0.0
  }'
```

Sample result:

```json
{"id":"cmpl-973e09361657d259","object":"text_completion","created":1777640598,"model":"/models/DeepSeek-V4-Pro","choices":[{"index":0,"text":" Do not include any other text or explanation. The answer is 4.\nWhat","logprobs":null,"finish_reason":"length","stop_reason":null,"token_ids":null,"prompt_logprobs":null,"prompt_token_ids":null}],"service_tier":null,"system_fingerprint":"vllm-0.20.1rc1.dev135+ge786a2dfc-tp8-868a6cb7","usage":{"prompt_tokens":13,"total_tokens":29,"completion_tokens":16,"prompt_tokens_details":null},"kv_transfer_params":null}
```

Smoke-test success criteria:

- HTTP status is `200`
- `choices[0].text` is non-empty

### 3) GSM8K validation

```bash
MODEL=/home/models/DeepSeek-V4-Pro
lm_eval --model local-completions \
  --model_args model=$MODEL,base_url=http://0.0.0.0:8001/v1/completions,num_concurrent=2,max_retries=10,max_gen_toks=2048,timeout=60000 \
  --batch_size auto \
  --tasks gsm8k \
  --num_fewshot 8 \
  --output_path .
```

Reported result from PR #40871:

```text
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     8|exact_match|↑  |0.9538|±  |0.0058|
|     |       |strict-match    |     8|exact_match|↑  |0.9545|±  |0.0057|
```

## DeepSeek-V4-Flash validation (MI355X, TP=4)

### 1) Serve command

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

### 2) Smoke test (single request)

```bash
curl -s http://localhost:8001/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Introduce the capital of US",
    "model": "/models/DeepSeek-V4-Flash",
    "max_tokens": 100,
    "temperature": 0.0
  }'
```

Sample result:

```json
{"id":"cmpl-86e0959d4415d914","object":"text_completion","created":1777638722,"model":"/models/DeepSeek-V4-Flash","choices":[{"index":0,"text":"\",\"answer\":\"Washington, D.C.\",\"type\":\"text\"},{\"question\":\"Introduce the capital of Canada.\",\"answer\":\"Ottawa\",\"type\":\"text\"},{\"question\":\"Introduce the capital of Mexico.\",\"answer\":\"Mexico City\",\"type\":\"text\"},{\"question\":\"Introduce the capital of Brazil.\",\"answer\":\"Brasília\",\"type\":\"text\"},{\"question\":\"Introduce the capital of Argentina.\",\"answer\":\"Buenos Aires\",\"type\":\"text\"},{\"question\":\"Introdu","logprobs":null,"finish_reason":"length","stop_reason":null,"token_ids":null,"prompt_logprobs":null,"prompt_token_ids":null}],"service_tier":null,"system_fingerprint":"vllm-0.20.1rc1.dev135+ge786a2dfc-tp4-015676fd","usage":{"prompt_tokens":7,"total_tokens":107,"completion_tokens":100,"prompt_tokens_details":null},"kv_transfer_params":null}
```

### 3) GSM8K validation

```bash
MODEL=/home/models/DeepSeek-V4-Flash
lm_eval --model local-completions \
  --model_args model=$MODEL,base_url=http://0.0.0.0:8001/v1/completions,num_concurrent=4,max_retries=10,max_gen_toks=2048,timeout=60000 \
  --batch_size auto \
  --tasks gsm8k \
  --num_fewshot 8 \
  --output_path .
```

Reported result from PR #40871:

```text
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     8|exact_match|↑  |0.9439|±  |0.0063|
|     |       |strict-match    |     8|exact_match|↑  |0.9431|±  |0.0064|
```

