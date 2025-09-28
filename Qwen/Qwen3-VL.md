# Qwen3-VL Usage Guide
[Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) is the most powerful vision-language model in the Qwen series to date created by Alibaba Cloud. 

This generation delivers comprehensive upgrades across the board: superior text understanding & generation, deeper visual perception & reasoning, extended context length, enhanced spatial and video dynamics comprehension, and stronger agent interaction capabilities.

Available in Dense and MoE architectures that scale from edge to cloud, with Instruct and reasoning‑enhanced Thinking editions for flexible, on‑demand deployment.


## Installing vLLM

```bash
uv venv
source .venv/bin/activate

# Install vLLM nightly
uv pip install -U vllm \
    --torch-backend=auto \
    --extra-index-url https://wheels.vllm.ai/nightly

# Install Qwen-VL utility library (recommended for offline inference)
uv pip install qwen-vl-utils==0.0.14

# Install transformers (from source or 4.57.0 when released)
# uv pip install transformers>=4.57.0
uv pip install git+https://github.com/huggingface/transformers.git
```


## Running Qwen3-VL


### Qwen3-VL-235B-A22B-Instruct
This is the Qwen3-VL flagship MoE model, which requires a minimum of 8 GPUs, each with at least 80 GB of memory (e.g., A100, H100, or H200). On some types of hardware the model may not launch successfully with its default setting. Recommended approaches by hardware type are:

- **H100 with `fp8`**: Use FP8 for optimal memory efficiency. An FP8 version of the model will be released soon. Stay tuned!
- **A100 & H100 with `bfloat16`**: Either reduce `--max-model-len` or restrict inference to images only.
- **H200 & B200 GPUs**: Run the model out of the box, supporting full context length and concurrent image and video processing.

See sections below for detailed launch arguments for each configuration. We are actively working on optimizations and the recommended ways to launch the model will be updated accordingly.

#### A100 & H100 (Image Inference)
```bash
vllm serve Qwen/Qwen3-VL-235B-A22B-Instruct \
  --tensor-parallel-size 8 \
  --limit-mm-per-prompt.video 0 \
  --mm-encoder-tp-mode data \
  --mm-processor-cache-type shm \
  --async-scheduling \
  --gpu-memory-utilization 0.95
```

#### A100 & H100 (Image + Video Inference)
```bash
vllm serve Qwen/Qwen3-VL-235B-A22B-Instruct \
  --tensor-parallel-size 8 \
  --max-model-len 128000 \
  --mm-processor-cache-type shm \
  --async-scheduling
```

#### H100 (Image + Video Inference + FP8)
```bash
vllm serve Qwen/Qwen3-VL-235B-A22B-Instruct \
  --tensor-parallel-size 8 \
  --mm-encoder-tp-mode data \
  --quantization fp8 \
  --mm-processor-cache-type shm \
  --async-scheduling
```

#### H200 & B200
```bash
vllm serve Qwen/Qwen3-VL-235B-A22B-Instruct \
  --tensor-parallel-size 8 \
  --mm-encoder-tp-mode data \
  --mm-processor-cache-type shm \
  --async-scheduling
```

### Configuration Tips
- It's highly recommended to specify `--limit-mm-per-prompt.video 0` if your inference server will only process image inputs since enabling video inputs consumes more memory reserved for long video embeddings. Alternatively, you can skip memory profiling for multimodal inputs by `--skip-mm-profiling` and lower `--gpu-memory-utilization` accordingly at your own risk.
- You can set `--max-model-len` to preserve memory. By default the model's context length is 262K, but `--max-model-len=128000` is usually good for most scenarios.
- Specifying `--mm-encoder-tp-mode data` deploys the vision encoder in a data-parallel fashion for better performance. This is because the vision encoder is very small, thus tensor parallelism brings little gain but incurs significant communication overhead. Enabling this feature does consume additional memory and may require adjustment on `--gpu-memory-utilization`.
- Specifying `--mm-processor-cache-type shm` utilizes host shared memory to cache preprocessed input images and/or videos which shows better performance at a high TP setting. If your workload involves only unique multimodal inputs, you may remove this argument and pass `--mm-processor-cache-gb 0` instead.
- You can use [benchmark_moe](https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_moe.py) to perform MoE Triton kernel tuning for your hardware.


### Benchmark on VisionArena-Chat Dataset

Once the server for the `Qwen3-VL-235B-A22B-Instruct` model is running, open another terminal and run the benchmark client:

```bash
vllm bench serve \
  --backend openai-chat \
  --endpoint /v1/chat/completions \
  --model Qwen/Qwen3-VL-235B-A22B-Instruct \
  --dataset-name hf \
  --dataset-path lmarena-ai/VisionArena-Chat \
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
    model="Qwen/Qwen3-VL-235B-A22B-Instruct",
    messages=messages,
    max_tokens=2048
)
print(f"Response costs: {time.time() - start:.2f}s")
print(f"Generated text: {response.choices[0].message.content}")
```

For more usage examples, check out the [vLLM user guide for multimodal models](https://docs.vllm.ai/en/latest/features/multimodal_inputs.html) and the [official Qwen3-VL GitHub Repository](https://github.com/QwenLM/Qwen3-VL)!
