# Qwen3.5 Usage Guide

[Qwen3.5](https://huggingface.co/Qwen/Qwen3.5-397B-A17B) is a multimodal mixture-of-experts model featuring a gated delta networks architecture with 397B total parameters and 17B active parameters. This guide covers how to efficiently deploy and serve the model across different hardware configurations and workload profiles using vLLM.

## Installing vLLM

You can either install vLLM from pip or use the pre-built Docker image.

### Pip Install
```bash
uv venv
source .venv/bin/activate
uv pip install -U vllm \
    --torch-backend=auto \
    --extra-index-url https://wheels.vllm.ai/nightly
```

### Docker
```bash
docker run --gpus all \
  -p 8000:8000 \
  --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:qwen3_5 Qwen/Qwen3.5-397B-A17B \
    --tensor-parallel-size 8 \
    --reasoning-parser qwen3 \
    --enable-prefix-caching
```
(See detailed deployment configurations below)

For Blackwell GPUs, use `vllm/vllm-openai:qwen3_5-cu130`

## Running Qwen3.5

The configurations below have been verified on 8x H200 GPUs.

### Throughput-Focused Serving

#### Text-Only

For maximum text throughput under high concurrency, use `--language-model-only` to skip loading the vision encoder and free up memory for KV cache.

```bash
vllm serve Qwen/Qwen3.5-397B-A17B \
  --tensor-parallel-size 8 \
  --language-model-only \
  --reasoning-parser qwen3 \
  --enable-prefix-caching
```

#### Multimodal

For multimodal workloads, use `--mm-encoder-tp-mode data` for data-parallel vision encoding and `--mm-processor-cache-type shm` to efficiently cache and transfer preprocessed multimodal inputs in shared memory.

```bash
vllm serve Qwen/Qwen3.5-397B-A17B \
  --tensor-parallel-size 8 \
  --mm-encoder-tp-mode data \
  --mm-processor-cache-type shm \
  --reasoning-parser qwen3 \
  --enable-prefix-caching
```

!!! tip
    To enable tool calling, add `--enable-auto-tool-choice --tool-call-parser qwen3_coder` to the serve command.

### Latency-Focused Serving

For latency-sensitive workloads at low concurrency, enable MTP-1 speculative decoding and disable prefix caching. MTP-1 reduces time-per-output-token (TPOT) with a high acceptance rate, at the cost of lower throughput under load.

```bash
vllm serve Qwen/Qwen3.5-397B-A17B \
  --tensor-parallel-size 8 \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 1}' \
  --reasoning-parser qwen3
```

### GB200 Deployment (2 Nodes x 4 GPUs)

You can also deploy across 2 GB200 nodes with 4 GPUs each. Use the same base configuration as H200, but add multi-node arguments.

!!! important
    The default FlashInfer backend has a degradation issue on Blackwell GPUs that is currently under investigation. Please use `--attention-backend FLASH_ATTN` for Blackwell deployments.

On the head node:
```bash
vllm serve Qwen/Qwen3.5-397B-A17B \
  --tensor-parallel-size 8 \
  --reasoning-parser qwen3 \
  --enable-prefix-caching \
  --attention-backend FLASH_ATTN \
  --nnodes 2 \
  --node-rank 0 \
  --master-addr <head_node_ip>
```

On the worker node:
```bash
vllm serve Qwen/Qwen3.5-397B-A17B \
  --tensor-parallel-size 8 \
  --reasoning-parser qwen3 \
  --enable-prefix-caching \
  --attention-backend FLASH_ATTN \
  --nnodes 2 \
  --node-rank 1 \
  --master-addr <head_node_ip> \
  --headless
```

### Configuration Tips

- **Prefix Caching**: Prefix caching for Mamba cache "align" mode is currently experimental. Please report any issues you may observe.
- **Multi-token Prediction**: MTP-1 reduces per-token latency but degrades text throughput under high concurrency because speculative tokens consume KV cache capacity, reducing effective batch size. Depending on your use case, you may adjust `num_speculative_tokens`: higher values can improve latency further but may have varying acceptance rates and throughput trade-offs.
- **Encoder Data Parallelism**: Specifying `--mm-encoder-tp-mode data` deploys the vision encoder in a data-parallel fashion for better throughput performance. This consumes additional memory and may require adjustment of `--gpu-memory-utilization`.
- **Media Embedding Size**: You can adjust the maximum media embedding size allowed by modifying the HuggingFace processor config at server startup via passing `--mm-processor-kwargs`. For example: `--mm-processor-kwargs '{"video_kwargs": {"size": {"longest_edge": 234881024, "shortest_edge": 4096}}}'`

### Benchmarking

Once the server is running, open another terminal and run the benchmark client:

```bash
vllm bench serve \
  --backend openai-chat \
  --endpoint /v1/chat/completions \
  --model Qwen/Qwen3.5-397B-A17B \
  --dataset-name random \
  --random-input-len 2048 \
  --random-output-len 512 \
  --num-prompts 1000 \
  --request-rate 20
```

### Consume the OpenAI API Compatible Server

```python
import time
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
    timeout=3600
)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://ofasys-multimodal-wlcb-3-toshanghai.oss-accelerate.aliyuncs.com/wpf272043/keepme/image/receipt.png"
                }
            },
            {
                "type": "text",
                "text": "Read all the text in the image."
            }
        ]
    }
]

start = time.time()
response = client.chat.completions.create(
    model="Qwen/Qwen3.5-397B-A17B",
    messages=messages,
    max_tokens=2048
)
print(f"Response costs: {time.time() - start:.2f}s")
print(f"Generated text: {response.choices[0].message.content}")
```
