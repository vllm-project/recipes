# Qwen3.5 Usage Guide
[Qwen3.5](https://github.com/QwenLM/Qwen3.5) is the most powerful vision-language model in the Qwen series to date created by Alibaba Cloud. 

This generation delivers comprehensive upgrades across the board: superior text understanding & generation, deeper visual perception & reasoning, extended context length, enhanced spatial and video dynamics comprehension, and stronger agent interaction capabilities.

Available in Dense and MoE architectures that scale from edge to cloud, with Instruct and reasoning‑enhanced Thinking editions for flexible, on‑demand deployment.


## Installing vLLM

```bash
uv venv

# After vLLM 0.16.0 release
source .venv/bin/activate
uv pip install -U vllm --pre \
    --extra-index-url https://wheels.vllm.ai/nightly/cu129 \
    --extra-index-url https://download.pytorch.org/whl/cu129 \
    --index-strategy unsafe-best-match

# install transformers from source
uv pip install git+https://github.com/huggingface/transformers.git

# Install Qwen-VL utility library (recommended for offline inference)
uv pip install qwen-vl-utils
```


## Running Qwen3.5


### Qwen3.5-35B-A3B-Instruct
This is the Qwen3.5 flagship MoE model, which requires a minimum of 8 GPUs, each with at least 96 GB of memory (e.g., H200 and B200). On some types of hardware the model may not launch successfully with its default setting. Recommended approaches by hardware type are:

- **H200 & B200**: Run the model out of the box, supporting full context length and concurrent image and video processing.

See sections below for detailed launch arguments for each configuration. We are actively working on optimizations and the recommended ways to launch the model will be updated accordingly.

<details>
<summary>H200 (Image + Video Inference, BF16)</summary>

```bash
vllm serve Qwen/Qwen3.5-35B-A3B-Instruct \
  --tensor-parallel-size 8 \
  --mm-encoder-tp-mode data \
  --speculative-config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 2}' \
  --enable-expert-parallel
```

</details>


<details>
<summary>H200 (Text-only Inference, BF16)</summary>

```bash
vllm serve Qwen/Qwen3.5-35B-A3B-Instruct \
  --tensor-parallel-size 8 \
  --language-model-only \
  --speculative-config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 2}' \
  --enable-expert-parallel
```

</details>


> ℹ️ **Note**  
> Qwen3.5 also supports Multi-Token Prediction (MTP in short), you can launch the model server with `--speculative-config` flag to enable MTP.
> You can enable text-only mode by passing `--language-model-only`, which skips the vision encoder and multimodal profiling to free up memory for additional KV cache.


### Configuration Tips
- It's highly recommended to specify `--limit-mm-per-prompt.video 0` if your inference server will only process image inputs since enabling video inputs consumes more memory reserved for long video embeddings. Alternatively, you can skip memory profiling for multimodal inputs by `--skip-mm-profiling` and lower `--gpu-memory-utilization` accordingly at your own risk.
- To avoid undesirable CPU contention, it's recommended to limit the number of threads allocated to preprocessing by setting the environment variable `OMP_NUM_THREADS=1`. This is particulaly useful and shows significant throughput improvement when deploying multiple vLLM instances on the same host.
- You can set `--max-model-len` to preserve memory. By default the model's context length is 262K, but `--max-model-len 128000` is good for most scenarios.
- Specifying `--mm-encoder-tp-mode data` deploys the vision encoder in a data-parallel fashion for better performance. This is because the vision encoder is very small, thus tensor parallelism brings little gain but incurs significant communication overhead. Enabling this feature does consume additional memory and may require adjustment on `--gpu-memory-utilization`.
- If your workload involves mostly **unique** multimodal inputs only, it is recommended to pass `--mm-processor-cache-gb 0` to avoid caching overhead. Otherwise, specifying `--mm-processor-cache-type shm` enables this experimental feature which utilizes host shared memory to cache preprocessed input images and/or videos which shows better performance at a high TP setting.
- vLLM supports Expert Parallelism (EP) via `--enable-expert-parallel`, which allows experts in MoE models to be deployed on separate GPUs for better throughput. Check out [Expert Parallelism Deployment](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment.html) for more details.
- You can use [benchmark_moe](https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_moe.py) to perform MoE Triton kernel tuning for your hardware.


### Benchmark on VisionArena-Chat Dataset

Once the server for the `Qwen3.5-35B-A3B-Instruct` model is running, open another terminal and run the benchmark client:

```bash
vllm bench serve \
  --backend openai-chat \
  --endpoint /v1/chat/completions \
  --model Qwen/Qwen3.5-35B-A3B-Instruct \
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
    model="Qwen/Qwen3.5-35B-A3B-Instruct",
    messages=messages,
    max_tokens=2048
)
print(f"Response costs: {time.time() - start:.2f}s")
print(f"Generated text: {response.choices[0].message.content}")
```

For more usage examples, check out the [vLLM user guide for multimodal models](https://docs.vllm.ai/en/latest/features/multimodal_inputs.html) and the [official Qwen3.5 GitHub Repository](https://github.com/QwenLM/Qwen3.5)!
