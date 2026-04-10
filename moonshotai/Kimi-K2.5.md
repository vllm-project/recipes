# moonshotai/Kimi-K2.5 Usage Guide
[Kimi K2.5](https://huggingface.co/moonshotai/Kimi-K2.5) is an open-source, native multimodal agentic model built through continual pretraining on approximately 15 trillion mixed visual and text tokens atop Kimi-K2-Base. It seamlessly integrates vision and language understanding with advanced agentic capabilities, instant and thinking modes, as well as conversational and agentic paradigms.

## Use vLLM with Docker

Pull the vLLM release image from [Docker Hub](https://hub.docker.com/r/vllm/vllm-openai/tags?name=17.0):

```bash
docker pull vllm/vllm-openai:v0.17.0-cu130 # CUDA 13.0
docker pull vllm/vllm-openai:v0.17.0       # Other CUDA versions
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

## Installing vLLM

```bash
uv venv
source .venv/bin/activate
uv pip install vllm --torch-backend auto
```

## Running Kimi-K2.5 with vLLM
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

### Deployment Scenarios

You can use 8×B200 to launch `nvidia/Kimi-K2.5-NVFP4`. See sections below for low latency and high throughput launch configurations.

<details>
<summary>Low Latency (TP8)</summary>

Use tensor parallelism across all 8 GPUs for minimum latency:

```bash
vllm serve nvidia/Kimi-K2.5-NVFP4 --host 0.0.0.0 --port 8888 \
    --tensor-parallel-size 8 \
    --gpu-memory-utilization 0.90 \
    --reasoning-parser kimi_k2 \
    --tool-call-parser kimi_k2 \
    --trust-remote-code
```

</details>

<details>
<summary>High Throughput (TP4 + EP4)</summary>

Use TP4 with expert parallelism (EP degree is derived automatically as `world_size / tensor_parallel_size`) for maximum throughput:

```bash
vllm serve nvidia/Kimi-K2.5-NVFP4 --host 0.0.0.0 --port 8888 \
    --tensor-parallel-size 4 \
    --enable-expert-parallel \
    --gpu-memory-utilization 0.90 \
    --reasoning-parser kimi_k2 \
    --tool-call-parser kimi_k2 \
    --trust-remote-code
```

</details>

### Benchmark Results

Benchmarks run on 8×B200 with [`nvidia/Kimi-K2.5-NVFP4`](https://huggingface.co/nvidia/Kimi-K2.5-NVFP4) (FP4 precision), vLLM v0.17.0, ISL=1024, OSL=1024, concurrency=4. Results sourced from [SemiAnalysis InferenceX](https://github.com/SemiAnalysisAI/InferenceX) ([artifact](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/23669977901/artifacts/6279124070)).

#### Low Latency (TP8, EP1)

```
============ Serving Benchmark Result ============
Hardware:                                8×B200
Model:                                   nvidia/Kimi-K2.5-NVFP4
Tensor Parallel Size:                    8
Expert Parallel Size:                    1
Input Sequence Length:                   1024
Output Sequence Length:                  1024
Concurrency:                             4
--------------------------------------------------
Total Token Throughput/GPU (tok/s):      128.44
Output Token Throughput/GPU (tok/s):     63.91
Input Token Throughput/GPU (tok/s):      64.53
---------------Time to First Token----------------
Mean TTFT (ms):                          163.73
Median TTFT (ms):                        152.39
P90 TTFT (ms):                           170.05
P99 TTFT (ms):                           279.02
P99.9 TTFT (ms):                         279.05
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          7.42
Median TPOT (ms):                        7.44
P90 TPOT (ms):                           7.61
P99 TPOT (ms):                           7.64
P99.9 TPOT (ms):                         7.64
---------------Inter-token Latency----------------
Mean ITL (ms):                           7.42
Median ITL (ms):                         7.06
P90 ITL (ms):                            7.17
P99 ITL (ms):                            7.38
P99.9 ITL (ms):                          134.69
--------------End-to-End Latency (s)--------------
Mean E2EL (s):                           6.97
Median E2EL (s):                         7.07
P90 E2EL (s):                            7.65
P99 E2EL (s):                            7.76
==================================================
```

#### High Throughput (TP4, EP4)

```
============ Serving Benchmark Result ============
Hardware:                                8×B200
Model:                                   nvidia/Kimi-K2.5-NVFP4
Tensor Parallel Size:                    4
Expert Parallel Size:                    4
Input Sequence Length:                   1024
Output Sequence Length:                  1024
Concurrency:                             4
--------------------------------------------------
Total Token Throughput/GPU (tok/s):      228.62
Output Token Throughput/GPU (tok/s):     113.75
Input Token Throughput/GPU (tok/s):      114.87
---------------Time to First Token----------------
Mean TTFT (ms):                          169.94
Median TTFT (ms):                        169.94
P90 TTFT (ms):                           175.14
P99 TTFT (ms):                           181.79
P99.9 TTFT (ms):                         182.10
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          8.36
Median TPOT (ms):                        8.41
P90 TPOT (ms):                           8.51
P99 TPOT (ms):                           8.60
P99.9 TPOT (ms):                         8.62
---------------Inter-token Latency----------------
Mean ITL (ms):                           8.36
Median ITL (ms):                         7.96
P90 ITL (ms):                            8.05
P99 ITL (ms):                            8.47
P99.9 ITL (ms):                          148.61
--------------End-to-End Latency (s)--------------
Mean E2EL (s):                           7.84
Median E2EL (s):                         7.96
P90 E2EL (s):                            8.63
P99 E2EL (s):                            8.73
==================================================
```

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
