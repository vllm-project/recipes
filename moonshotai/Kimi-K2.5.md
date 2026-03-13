# moonshotai/Kimi-K2.5 Usage Guide
[Kimi K2.5](https://huggingface.co/moonshotai/Kimi-K2.5) is an open-source, native multimodal agentic model built through continual pretraining on approximately 15 trillion mixed visual and text tokens atop Kimi-K2-Base. It seamlessly integrates vision and language understanding with advanced agentic capabilities, instant and thinking modes, as well as conversational and agentic paradigms.

## Installing vLLM

You can either install vLLM from pip or use the pre-built Docker image.

### Pip Install

#### NVIDIA 

```bash
uv venv
source .venv/bin/activate
uv pip install vllm --torch-backend auto
```

#### AMD 

> Note: The vLLM wheel for ROCm requires Python 3.12, ROCm 7.0, and glibc >= 2.35. If your environment does not meet these requirements, please use the Docker-based setup as described below. Supported GPUs: MI300X, MI325X, MI355X.

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install vllm --extra-index-url https://wheels.vllm.ai/rocm
```

### Docker 

#### AMD

```bash
docker run --device=/dev/kfd --device=/dev/dri \
  --security-opt seccomp=unconfined \
  --group-add video \
  --ipc=host \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai-rocm:latest \
  moonshotai/Kimi-K2.5 \
  --tensor-parallel-size 8 \
  --reasoning-parser kimi_k2 \
  --enable-prefix-caching \
  --trust-remote-code
```

## Running Kimi-K2.5
See the following commands to deploy Kimi-K2.5 with the vLLM inference server. 

### NVIDIA 

The configuration below has been verified on 8xH200 GPUs.
```bash
vllm serve moonshotai/Kimi-K2.5 -tp 8 \
    --mm-encoder-tp-mode data \
    --compilation_config.pass_config.fuse_allreduce_rms true \
    --tool-call-parser kimi_k2 \
    --reasoning-parser kimi_k2 \
    --enable-auto-tool-choice \
    --trust-remote-code
```
The `--reasoning-parser` flag specifies the reasoning parser to use for extracting reasoning content from the model output.

### AMD 

The configuration below has been verified on 8x MI300X/MI355X GPUs.
```bash
vllm serve moonshotai/Kimi-K2.5 -tp 8 \
    --mm-encoder-tp-mode data \
    --tool-call-parser kimi_k2 \
    --reasoning-parser kimi_k2 \
    --enable-auto-tool-choice \
    --trust-remote-code
```

### Configuration Tips
- `--async-scheduling` has been turned on by default to improve the overall system performance by overlapping scheduling overhead with the decoding process. If you run into issue with this feature, please try turning off this feature and file a bug report to vLLM.
- Specifying `--mm-encoder-tp-mode data` deploys the vision encoder in a data-parallel fashion for better performance. This is because the vision encoder is very small, thus tensor parallelism brings little gain but incurs significant communication overhead. Enabling this feature does consume additional memory and may require adjustment on `--gpu-memory-utilization`.
- If your workload involves mostly **unique** multimodal inputs only, it is recommended to pass `--mm-processor-cache-gb 0` to avoid caching overhead. Otherwise, specifying `--mm-processor-cache-type shm` enables this experimental feature which utilizes host shared memory to cache preprocessed input images and/or videos which shows better performance at a high TP setting.
- vLLM supports Expert Parallelism (EP) via `--enable-expert-parallel`, which allows experts in MoE models to be deployed on separate GPUs for better throughput. Check out [Expert Parallelism Deployment](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment.html) for more details.
- You can use [benchmark_moe](https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_moe.py) to perform MoE Triton kernel tuning for your hardware.

### Benchmark on VisionArena-Chat Dataset

Once the server for the `moonshotai/Kimi-K2.5` model is running, open another terminal and run the benchmark client:

```bash
vllm bench serve \
  --backend openai-chat \
  --endpoint /v1/chat/completions \
  --model moonshotai/Kimi-K2.5 \
  --dataset-name hf \
  --dataset-path lmarena-ai/VisionArena-Chat \
  --num-prompts 1000 \
  --request-rate 20 \ 
  --trust_remote_code
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
    model="moonshotai/Kimi-K2.5",
    messages=messages,
    max_tokens=2048
)
print(f"Response costs: {time.time() - start:.2f}s")
print(f"Generated text: {response.choices[0].message.content}")
```

For more usage examples, check out the [vLLM user guide for multimodal models](https://docs.vllm.ai/en/latest/features/multimodal_inputs.html) and the [official Kimi-K2.5 Hugging Face page](https://huggingface.co/moonshotai/Kimi-K2.5)!
