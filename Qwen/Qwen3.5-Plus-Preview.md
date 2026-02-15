# Qwen3.5-Plus-Preview Usage Guide

[Qwen3.5-Plus-Preview](https://huggingface.co/Qwen/Qwen3.5-Plus-Preview) is a large multimodal model from the Qwen series created by Alibaba Cloud. It supports both text-only and multimodal (image/video) inputs.

## Installing vLLM

You can either install vLLM from pip or use the pre-built Docker image.

### Pip Install
```bash
uv venv
source .venv/bin/activate
uv pip install -U vllm --torch-backend auto
```

### Docker
```bash
docker run --gpus all -p 8000:8000 vllm/vllm-openai:qwen3_5
```

## Running Qwen3.5-Plus-Preview

The configurations below have been verified on 8x H200 GPUs.

### Throughput-Focused Serving

#### Text-Only

For maximum text throughput under high concurrency, use `--language-model-only` to skip loading the vision encoder and free up memory for KV cache.

```bash
vllm serve Qwen/Qwen3.5-Plus-Preview \
  --tensor-parallel-size 8 \
  --language-model-only \
  --reasoning-parser deepseek_r1 \
  --enable-prefix-caching
```

#### Multimodal

For multimodal workloads, use `--mm-encoder-tp-mode data` for data-parallel vision encoding and `--mm-processor-cache-type shm` to efficiently cache and transfer preprocessed multimodal inputs in shared memory.

```bash
vllm serve Qwen/Qwen3.5-Plus-Preview \
  --tensor-parallel-size 8 \
  --mm-encoder-tp-mode data \
  --mm-processor-cache-type shm \
  --reasoning-parser deepseek_r1 \
  --enable-prefix-caching
```

### Latency-Focused Serving

For latency-sensitive workloads at low concurrency, enable MTP-1 speculative decoding and disable prefix caching. MTP-1 reduces time-per-output-token (TPOT) with a high acceptance rate, at the cost of lower throughput under load.

```bash
vllm serve Qwen/Qwen3.5-Plus-Preview \
  --tensor-parallel-size 8 \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 1}' \
  --reasoning-parser deepseek_r1
```

### Configuration Tips

- **Prefix Caching**: Prefix caching for Mamba cache "align" mode is currently experimental. Please report any issues you may observe.
- **Multi-token Prediction**: MTP-1 reduces per-token latency but degrades text throughput under high concurrency because speculative tokens consume KV cache capacity, reducing effective batch size. MTP-2 offers diminishing returns — the second speculative position has a lower acceptance rate and introduces higher ITL variance with similar TPOT gains.
- **Encoder Data Parallelism**: Specifying `--mm-encoder-tp-mode data` deploys the vision encoder in a data-parallel fashion for better throughput performance. This consumes additional memory and may require adjustment of `--gpu-memory-utilization`.
- **Asynchronous Scheduling for Hybrid Models**: `--async-scheduling` has been turned on by default to improve overall system performance by overlapping scheduling overhead with the decoding process. It is already compatible with prefix caching and MTP for models with hybrid models. If you run into issues with this feature, please try turning it off and file a bug report to vLLM.

### Benchmarking

Once the server is running, open another terminal and run the benchmark client:

```bash
vllm bench serve \
  --backend openai-chat \
  --endpoint /v1/chat/completions \
  --model Qwen/Qwen3.5-Plus-Preview \
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
    model="Qwen/Qwen3.5-Plus-Preview",
    messages=messages,
    max_tokens=2048
)
print(f"Response costs: {time.time() - start:.2f}s")
print(f"Generated text: {response.choices[0].message.content}")
```
