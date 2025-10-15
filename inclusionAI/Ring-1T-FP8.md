# Ring-1T-FP8 Usage Guide

This guide describes how to run Ring-1T-FP8.

## Installing vLLM

```bash
uv venv
source .venv/bin/activate
uv pip install -U vllm --torch-backend auto
```

## Running Ring-1T-FP8 with FP8 KV Cache on 8xH200

This guide covers the simplest way to run the model, using pure tensor parallel across 8 GPUs.

```bash

# Start server with FP8 model on 8 GPUs
vllm serve inclusionAI/Ring-1T-FP8 \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.97 \
  --max_num_seqs 32 \
  --kv-cache-dtype fp8 \
  --compilation-config '{"use_inductor": false}' \
  --served-model-name Ring-1T-FP8
```

* You can set `--max-model-len` to preserve memory. `--max-model-len=65536` is usually good for most scenarios.
* You can set `--max-num-batched-tokens` to balance throughput and latency, higher means higher throughput but higher latency. `--max-num-batched-tokens=32768` is usually good for prompt-heavy workloads. But you can reduce it to 16384 and 8192 to reduce activation memory usage and decrease latency.
* In the example, 97% of the total memory is used for this model, you can reduce it to a smaller number if an Out-Of-Memory (OOM) error occurs.

## Sending Example Request

You can send a request like the following to quickly verify the deployment.

```bash
curl http://localhost:8000/v1/chat/completions
    -H "Content-Type: application/json" \
    -d '{
        "model": "Ring-1T-FP8",
        "messages": [
            {
                "role": "user",
                "content": "9.11 and 9.8, which is greater?"
            }
        ]
    }'
```
