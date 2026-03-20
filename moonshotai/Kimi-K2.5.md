# moonshotai/Kimi-K2.5 Usage Guide
[Kimi K2.5](https://huggingface.co/moonshotai/Kimi-K2.5) is an open-source, native multimodal agentic model built through continual pretraining on approximately 15 trillion mixed visual and text tokens atop Kimi-K2-Base. It seamlessly integrates vision and language understanding with advanced agentic capabilities, instant and thinking modes, as well as conversational and agentic paradigms.

## Use vLLM with Docker

Pull the vLLM release image from Docker Hub ([CUDA](https://hub.docker.com/r/vllm/vllm-openai/tags?name=17.0) | [ROCm](https://hub.docker.com/r/vllm/vllm-openai-rocm/tags)):

```bash
docker pull vllm/vllm-openai:v0.17.0-cu130       # CUDA 13.0
docker pull vllm/vllm-openai:v0.17.0              # Other CUDA versions
docker pull vllm/vllm-openai-rocm:v0.16.0         # ROCm (AMD)
```

### Hopper (x86_64)

Verified on 8×H200 GPUs:

```bash
docker run --gpus all \
  -p 8000:8000 \
  --ipc=host \
  vllm/vllm-openai:v0.17.0-cu130 moonshotai/Kimi-K2.5 \
    --tensor-parallel-size 8 \
    --mm-encoder-tp-mode data \
    --compilation_config.pass_config.fuse_allreduce_rms true \
    --tool-call-parser kimi_k2 \
    --reasoning-parser kimi_k2 \
    --enable-auto-tool-choice \
    --trust-remote-code
```

### Blackwell (aarch64)

NVIDIA Blackwell (e.g., GB200) is also supported via the aarch64 image:

```bash
docker run --gpus all \
  -p 8000:8000 \
  --ipc=host \
  vllm/vllm-openai:v0.17.0-aarch64-cu130 moonshotai/Kimi-K2.5 \
    --tensor-parallel-size 4 \
    --mm-encoder-tp-mode data \
    --compilation_config.pass_config.fuse_allreduce_rms true \
    --tool-call-parser kimi_k2 \
    --reasoning-parser kimi_k2 \
    --enable-auto-tool-choice \
    --trust-remote-code
```

### AMD ROCm (MI355X)

For AMD Instinct MI355X GPUs, use the [MXFP4-quantized model](https://huggingface.co/amd/Kimi-K2.5-MXFP4) with the ROCm vLLM image:

Run with 4×MI355X GPUs (TP=4):

```bash
docker run \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --shm-size=16G \
  -p 8000:8000 \
  -e VLLM_ROCM_USE_AITER=1 \
  -e VLLM_ROCM_USE_AITER_MLA=1 \
  -e VLLM_ROCM_USE_AITER_MOE=1 \
  -e VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT8 \
  -e VLLM_ROCM_USE_AITER_TRITON_ROPE=1 \
  vllm/vllm-openai-rocm:v0.16.0 \
  amd/Kimi-K2.5-MXFP4 \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.90 \
    --block-size 1 \
    --mm-encoder-tp-mode data \
    --tool-call-parser kimi_k2 \
    --reasoning-parser kimi_k2 \
    --enable-auto-tool-choice \
    --trust-remote-code
```

For 8×MI355X GPUs, you can enable Expert Parallelism (TP=4, EP=2) for higher throughput on long-output workloads:

```bash
docker run \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --shm-size=16G \
  -p 8000:8000 \
  -e VLLM_ROCM_USE_AITER=1 \
  -e VLLM_ROCM_USE_AITER_MLA=1 \
  -e VLLM_ROCM_USE_AITER_MOE=1 \
  -e VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT8 \
  -e VLLM_ROCM_USE_AITER_TRITON_ROPE=1 \
  vllm/vllm-openai-rocm:v0.16.0 \
  amd/Kimi-K2.5-MXFP4 \
    --tensor-parallel-size 4 \
    --enable-expert-parallel \
    --gpu-memory-utilization 0.90 \
    --block-size 1 \
    --mm-encoder-tp-mode data \
    --tool-call-parser kimi_k2 \
    --reasoning-parser kimi_k2 \
    --enable-auto-tool-choice \
    --trust-remote-code
```

> **Note:** On systems with MEC firmware older than version 177, set `HSA_NO_SCRATCH_RECLAIM=1` (via `-e HSA_NO_SCRATCH_RECLAIM=1`) to prevent RCCL memory reclaim crashes. Check your firmware version with `rocm-smi --showfw | grep MEC`.

## Installing vLLM

```bash
uv venv
source .venv/bin/activate
uv pip install vllm --torch-backend auto
```

## Running Kimi-K2.5 with vLLM

### Running on NVIDIA

See the following command to deploy Kimi-K2.5 with the vLLM inference server. The configuration below has been verified on 8xH200 GPUs.
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

### Running on AMD ROCm

Deploy the MXFP4-quantized model on AMD MI355X GPUs with AITER acceleration:

```bash
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MLA=1
export VLLM_ROCM_USE_AITER_MOE=1
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT8
export VLLM_ROCM_USE_AITER_TRITON_ROPE=1

vllm serve amd/Kimi-K2.5-MXFP4 -tp 4 \
    --gpu-memory-utilization 0.90 \
    --block-size 1 \
    --mm-encoder-tp-mode data \
    --tool-call-parser kimi_k2 \
    --reasoning-parser kimi_k2 \
    --enable-auto-tool-choice \
    --trust-remote-code
```

To enable Expert Parallelism on 8 GPUs (TP=4, EP=2):

```bash
vllm serve amd/Kimi-K2.5-MXFP4 -tp 4 \
    --enable-expert-parallel \
    --gpu-memory-utilization 0.90 \
    --block-size 1 \
    --mm-encoder-tp-mode data \
    --tool-call-parser kimi_k2 \
    --reasoning-parser kimi_k2 \
    --enable-auto-tool-choice \
    --trust-remote-code
```

The AITER environment variables enable optimized AMD kernels:
| Variable | Description |
|----------|-------------|
| `VLLM_ROCM_USE_AITER` | Enable AITER acceleration (general) |
| `VLLM_ROCM_USE_AITER_MLA` | Enable AITER Multi-head Latent Attention kernels |
| `VLLM_ROCM_USE_AITER_MOE` | Enable AITER Mixture-of-Experts kernels |
| `VLLM_ROCM_QUICK_REDUCE_QUANTIZATION` | Quantized AllReduce communication (INT8) |
| `VLLM_ROCM_USE_AITER_TRITON_ROPE` | Enable Triton-based RoPE kernels |

### Configuration Tips
- `--async-scheduling` has been turned on by default to improve the overall system performance by overlapping scheduling overhead with the decoding process. If you run into issue with this feature, please try turning off this feature and file a bug report to vLLM.
- Specifying `--mm-encoder-tp-mode data` deploys the vision encoder in a data-parallel fashion for better performance. This is because the vision encoder is very small, thus tensor parallelism brings little gain but incurs significant communication overhead. Enabling this feature does consume additional memory and may require adjustment on `--gpu-memory-utilization`.
- If your workload involves mostly **unique** multimodal inputs only, it is recommended to pass `--mm-processor-cache-gb 0` to avoid caching overhead. Otherwise, specifying `--mm-processor-cache-type shm` enables this experimental feature which utilizes host shared memory to cache preprocessed input images and/or videos which shows better performance at a high TP setting.
- vLLM supports Expert Parallelism (EP) via `--enable-expert-parallel`, which allows experts in MoE models to be deployed on separate GPUs for better throughput. Check out [Expert Parallelism Deployment](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment.html) for more details.
- You can use [benchmark_moe](https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_moe.py) to perform MoE Triton kernel tuning for your hardware.


### Accelerating Kimi 2.5 with Eagle3 MTP
We recommend using [lightseekorg/kimi-k2.5-eagle3](https://huggingface.co/lightseekorg/kimi-k2.5-eagle3) and [nvidia/Kimi-K2.5-Thinking-Eagle3](https://huggingface.co/nvidia/Kimi-K2.5-Thinking-Eagle3) to accelerate inference of Kimi 2.5. This feature is supported in vLLM nightly and `vLLM>=0.18.0`.

```bash
vllm serve moonshotai/Kimi-K2.5 \
    --tensor-parallel-size 8 \
    --speculative-config '{"model": <eagle3_model_name>, "method": "eagle3", "num_speculative_tokens": 3}' \
    --trust-remote-code
```

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
    model="moonshotai/Kimi-K2.5",
    messages=messages,
    max_tokens=2048
)
print(f"Response costs: {time.time() - start:.2f}s")
print(f"Generated text: {response.choices[0].message.content}")
```

For more usage examples, check out the [vLLM user guide for multimodal models](https://docs.vllm.ai/en/latest/features/multimodal_inputs.html) and the [official Kimi-K2.5 Hugging Face page](https://huggingface.co/moonshotai/Kimi-K2.5)!
