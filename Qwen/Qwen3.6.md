# Qwen3.6 Usage Guide

[Qwen3.6](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) is a multimodal mixture-of-experts model featuring a gated delta networks architecture with 35B total parameters and 3B active parameters (256 experts, 8 routed + 1 shared). Designed for agentic coding and repository-level reasoning, this guide covers how to efficiently deploy and serve the model using vLLM.

Qwen3.6 shares the same architecture family as [Qwen3.5](Qwen3.5.md). For installation instructions, Docker setup, configuration tips, troubleshooting, API usage examples, and ultra-long text processing, please refer to the [Qwen3.5 Usage Guide](Qwen3.5.md).

## Running Qwen3.6

!!! tip
    We recommend using the official FP8 checkpoint [Qwen/Qwen3.6-35B-A3B-FP8](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8) for optimal serving efficiency.

### Throughput-Focused Serving

#### Text-Only

For maximum text throughput under high concurrency, use `--language-model-only` to skip loading the vision encoder and free up memory for KV cache as well as enabling Expert Parallelism.

```bash
vllm serve Qwen/Qwen3.6-35B-A3B-FP8 \
  -dp 8 \
  --enable-expert-parallel \
  --language-model-only \
  --reasoning-parser qwen3 \
  --enable-prefix-caching
```

#### Multimodal

For multimodal workloads, use `--mm-encoder-tp-mode data` for data-parallel vision encoding and `--mm-processor-cache-type shm` to efficiently cache and transfer preprocessed multimodal inputs in shared memory.

```bash
vllm serve Qwen/Qwen3.6-35B-A3B-FP8 \
  -dp 8 \
  --enable-expert-parallel \
  --mm-encoder-tp-mode data \
  --mm-processor-cache-type shm \
  --reasoning-parser qwen3 \
  --enable-prefix-caching
```

!!! tip
    To enable tool calling, add `--enable-auto-tool-choice --tool-call-parser qwen3_coder` to the serve command.

### Latency-Focused Serving

For latency-sensitive workloads at low concurrency, enable MTP speculative decoding and disable prefix caching. MTP reduces time-per-output-token (TPOT) with a high acceptance rate, at the cost of lower throughput under load.

!!! note
    MTP speculative decoding for AMD GPUs is under development.

```bash
vllm serve Qwen/Qwen3.6-35B-A3B-FP8 \
  --tensor-parallel-size 8 \
  --speculative-config '{"method": "qwen3_next_mtp", "num_speculative_tokens": 2}' \
  --reasoning-parser qwen3
```

### Benchmarking

Once the server is running, open another terminal and run the benchmark client:

```bash
vllm bench serve \
  --backend openai-chat \
  --endpoint /v1/chat/completions \
  --model Qwen/Qwen3.6-35B-A3B \
  --dataset-name random \
  --random-input-len 2048 \
  --random-output-len 512 \
  --num-prompts 1000 \
  --request-rate 20
```
