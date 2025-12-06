# GLM-4V Usage Guide

This guide describes how to run GLM-4.5V / GLM-4.6V with native FP8.
In the GLM-4.5V / GLM-4.6V series, FP8 models have minimal accuracy loss. 
Unless you need strict reproducibility for benchmarking or similar scenarios, 
we recommend using FP8 to run at a lower cost.

## Installing vLLM

```bash
uv venv
source .venv/bin/activate
uv pip install -U vllm --torch-backend auto # vllm>=0.12.0 is required
```

## Running GLM-4.5V / GLM-4.6V with FP8 or BF16 on 4xH100

There are two ways to parallelize the model over multiple GPUs: (1) Tensor-parallel or (2) Data-parallel. Each one has its own advantages, where tensor-parallel is usually more beneficial for low-latency / low-load scenarios and data-parallel works better for cases where there is a lot of data with heavy-loads.

run tensor-parallel like this:

```bash

# Start server with FP8 model on 4 GPUs, the model can also changed to BF16 as zai-org/GLM-4.5V
vllm serve zai-org/GLM-4.5V-FP8 \
     --tensor-parallel-size 4 \
     --tool-call-parser glm45 \
     --reasoning-parser glm45 \
     --enable-auto-tool-choice \
     --enable-expert-parallel \
     --allowed-local-media-path / \
     --mm-encoder-tp-mode data
```

* You can set `--max-model-len` to preserve memory. `--max-model-len=65536` is usually good for most scenarios. Note that GLM-4.5V only supports a 64K context length, while GLM-4.6V supports a 128K context length.
* You can set `--max-num-batched-tokens` to balance throughput and latency, higher means higher throughput but higher latency. `--max-num-batched-tokens=32768` is usually good for prompt-heavy workloads. But you can reduce it to 16k and 8k to reduce activation memory usage and decrease latency.
* vLLM conservatively use 90% of GPU memory, you can set `--gpu-memory-utilization=0.95` to maximize KVCache.
* Make sure to follow the command-line instructions to ensure the tool-calling functionality is properly enabled.

### Benchmark on VisionArena-Chat Dataset

Once the server for the `zai-org/GLM-4.5V-FP8` model is running, open another terminal and run the benchmark client:

```bash
vllm bench serve \
  --backend openai-chat \
  --endpoint /v1/chat/completions \
  --model zai-org/GLM-4.5V-FP8 \
  --dataset-name hf \
  --dataset-path lmarena-ai/VisionArena-Chat \
  --num-prompts 1000 \
  --request-rate 20
```