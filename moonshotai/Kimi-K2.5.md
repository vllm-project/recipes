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

TP8 spreads computation across all 8 GPUs, minimizing per-request latency at the cost of higher inter-GPU communication overhead. Use this when time-to-first-token matters most.

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
<summary>High Throughput (EP4)</summary>

EP4 uses 4 GPUs with expert parallelism, routing MoE experts across GPUs to reduce per-expert communication overhead and improve throughput in high-concurrency workloads. Use this when maximizing tokens-per-second across concurrent requests matters most.

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

#### Low Latency (TP8)

Concurrency is 4. [Full run](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/23669977901/job/70014502441).

```
============ Serving Benchmark Result ============
Successful requests:                     40        
Benchmark duration (s):                  100.21    
Total input tokens:                      37068     
Total generated tokens:                  36709     
Request throughput (req/s):              0.40      
Output token throughput (tok/s):         366.32    
Total Token throughput (tok/s):          736.23    
---------------Time to First Token----------------
Mean TTFT (ms):                          148.61    
Median TTFT (ms):                        137.49    
P90 TTFT (ms):                           160.43    
P99 TTFT (ms):                           335.19    
P99.9 TTFT (ms):                         335.23    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          10.43     
Median TPOT (ms):                        10.45     
P90 TPOT (ms):                           10.59     
P99 TPOT (ms):                           10.65     
P99.9 TPOT (ms):                         10.68     
---------------Inter-token Latency----------------
Mean ITL (ms):                           10.43     
Median ITL (ms):                         10.15     
P90 ITL (ms):                            10.47     
P99 ITL (ms):                            10.84     
P99.9 ITL (ms):                          111.75    
----------------End-to-end Latency----------------
Mean E2EL (ms):                          9713.73   
Median E2EL (ms):                        9823.84   
P90 E2EL (ms):                           10617.68  
P99 E2EL (ms):                           10881.95  
P99.9 E2EL (ms):                         10888.45  
==================================================
```

#### High Throughput (EP4)

Benchmark results coming soon.

### Configuration Tips
- `--async-scheduling` has been turned on by default to improve the overall system performance by overlapping scheduling overhead with the decoding process. If you run into issues with this feature, please try turning off this feature and file a bug report to vLLM.
- Specifying `--mm-encoder-tp-mode data` deploys the vision encoder in a data-parallel fashion for better performance. This is because the vision encoder is very small, thus tensor parallelism brings little gain but incurs significant communication overhead. Enabling this feature does consume additional memory and may require adjustment on `--gpu-memory-utilization`.
- If your workload involves mostly **unique** multimodal inputs only, it is recommended to pass `--mm-processor-cache-gb 0` to avoid caching overhead. Otherwise, specifying `--mm-processor-cache-type shm` enables this experimental feature which utilizes host shared memory to cache preprocessed input images and/or videos which shows better performance at a high TP setting.
- You can use [benchmark_moe](https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_moe.py) to perform MoE Triton kernel tuning for your hardware.
- vLLM supports Expert Parallelism (EP) via `--enable-expert-parallel`, which allows experts in MoE models to be deployed on separate GPUs for better throughput. Check out [Expert Parallelism Deployment](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment.html) for more details.


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
