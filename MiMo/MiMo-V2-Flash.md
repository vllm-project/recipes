# MiMo-V2-Flash Usage Guide

## Introduction
[MiMo-V2-Flash](https://huggingface.co/XiaomiMiMo/MiMo-V2-Flash) is a MoE language model with 309B total parameters and 15B active parameters. Designed for high-speed reasoning and agentic workflows, it utilizes a novel hybrid attention architecture and Multi-Token Prediction (MTP) to achieve state-of-the-art performance while significantly reducing inference costs.

## Installing vLLM

```bash
uv venv
source .venv/bin/activate
uv pip install vllm --extra-index-url https://wheels.vllm.ai/nightly
```

## Running MiMo-V2-Flash

run `TP` like this:
```bash
vllm serve XiaomiMiMo/MiMo-V2-Flash \
    --host 0.0.0.0 \
    --port 9001 \
    --seed 1024 \
    --served-model-name mimo_v2_flash \
    --tensor-parallel-size 4 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --generation-config vllm
```

run `Tool Call` like this:
```bash
vllm serve XiaomiMiMo/MiMo-V2-Flash \
    --host 0.0.0.0 \
    --port 9001 \
    --seed 1024 \
    --served-model-name mimo_v2_flash \
    --tensor-parallel-size 4 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --tool-call-parser qwen3_xml \
    --reasoning-parser qwen3 \
    --generation-config vllm
```

run `DP/TP/EP` like this:
```bash
vllm serve XiaomiMiMo/MiMo-V2-Flash \
    --host 0.0.0.0 \
    --port 9001 \
    --seed 1024 \
    --served-model-name mimo_v2_flash \
    --data-parallel-size 2 \
    --tensor-parallel-size 4 \
    --enable-expert-parallel \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --generation-config vllm
```

* You can set `--max-model-len` to preserve memory. `--max-model-len=65536` is usually good for most scenarios and max is 128k.
* You can set `--max-num-batched-tokens` to balance throughput and latency, higher means higher throughput but higher latency. `--max-num-batched-tokens=32768` is usually good for prompt-heavy workloads. But you can reduce it to 16k and 8k to reduce activation memory usage and decrease latency.
* vLLM conservatively uses 90% of GPU memory, you can set `--gpu-memory-utilization=0.95` to maximize KVCache.
* Make sure to follow the command-line instructions to ensure the tool-calling functionality is properly enabled.

### curl Example

You can run the following `curl` command:

```bash
curl -X POST http://localhost:9001/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "mimo_v2_flash",
        "messages": [
            {
                "role": "user",
                "content": "Hello MiMo!"
            }
        ],
        "chat_template_kwargs": {
            "enable_thinking": true
        }
    }'
```

* Set `"enable_thinking": false` or remove the `chat_template_kwargs` section to disable thinking mode.

## Benchmarking

For benchmarking, disable prefix caching by adding `--no-enable-prefix-caching` to the server command.

### Benchmark

```bash
# Prompt-heavy benchmark (8k/1k)
vllm bench serve \
  --model XiaomiMiMo/MiMo-V2-Flash \
  --dataset-name random \
  --random-input-len 8000 \
  --random-output-len 1000 \
  --request-rate 3 \
  --num-prompts 1800 \
  --ignore-eos
```

### Benchmark Configurations

Test different workloads by adjusting input/output lengths:

- **Prompt-heavy**: 8000 input / 1000 output
- **Decode-heavy**: 1000 input / 8000 output
- **Balanced**: 1000 input / 1000 output

### Expected Output
It currently just runs, and many features—such as MTP—have not yet been added, so performance testing will be conducted later.

## Accuracy

### GSM8K

Script
```bash
lm_eval \
    --model local-chat-completions \
    --tasks gsm8k \
    --num_fewshot 5 \
    --apply_chat_template \
    --model_args model=mimo_v2_flash,base_url=http://0.0.0.0:9001/v1/chat/completions,num_concurrent=100,max_retries=20,tokenized_requests=False,tokenizer_backend=none,max_gen_toks=256
```
Result
```text
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9128|±  |0.0078|
|     |       |strict-match    |     5|exact_match|↑  |0.9075|±  |0.0080|
```

## AMD GPU Support

Please follow the steps here to install and run MiMo-V2-Flash on AMD MI300X/MI325X/MI355X

### Step 1: Install vLLM
> Note: The vLLM wheel for ROCm requires Python 3.12, ROCm 7.0, and glibc >= 2.35. If your environment does not meet these requirements, please use the Docker-based setup as described in the [documentation](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/#pre-built-images). 
```bash
uv venv
source .venv/bin/activate
uv pip install vllm --extra-index-url https://wheels.vllm.ai/rocm
```

### Step 2: Start the vLLM server

Run the vllm online serving:

```shell
export VLLM_ROCM_USE_AITER=0
vllm serve XiaomiMiMo/MiMo-V2-Flash \
    --served-model-name mimo_v2_flash \
    --tensor-parallel-size 4 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --generation-config vllm
```

## TODO

- [ ] Supports MTP.
- [ ] Provide stress test performance data.