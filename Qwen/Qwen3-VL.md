# Qwen3-VL Usage Guide
[Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) is the most powerful vision-language model in the Qwen series to date created by Alibaba Cloud. 

This generation delivers comprehensive upgrades across the board: superior text understanding & generation, deeper visual perception & reasoning, extended context length, enhanced spatial and video dynamics comprehension, and stronger agent interaction capabilities.

Available in Dense and MoE architectures that scale from edge to cloud, with Instruct and reasoning‑enhanced Thinking editions for flexible, on‑demand deployment.


## Installing vLLM

```bash
uv venv
source .venv/bin/activate

# Install vLLM >=0.11.0
uv pip install -U vllm

# Install Qwen-VL utility library (recommended for offline inference)
uv pip install qwen-vl-utils==0.0.14
```


## Running Qwen3-VL


### Qwen3-VL-235B-A22B-Instruct
This is the Qwen3-VL flagship MoE model, which requires a minimum of 8 GPUs, each with at least 80 GB of memory (e.g., A100, H100, or H200). On some types of hardware the model may not launch successfully with its default setting. Recommended approaches by hardware type are:

- **H100 with `fp8`**: Use FP8 checkpoint for optimal memory efficiency.
- **A100 & H100 with `bfloat16`**: Either reduce `--max-model-len` or restrict inference to images only.
- **H200 & B200**: Run the model out of the box, supporting full context length and concurrent image and video processing.

See sections below for detailed launch arguments for each configuration. We are actively working on optimizations and the recommended ways to launch the model will be updated accordingly.

<details>
<summary>H100 (Image + Video Inference, FP8)</summary>
```bash
vllm serve Qwen/Qwen3-VL-235B-A22B-Instruct-FP8 \
  --tensor-parallel-size 8 \
  --mm-encoder-tp-mode data \
  --enable-expert-parallel \
  --async-scheduling
```
</details>

<details>
<summary>H100 (Image Inference, FP8, TP4)</summary>
```bash
vllm serve Qwen/Qwen3-VL-235B-A22B-Instruct-FP8 \
  --tensor-parallel-size 4 \
  --limit-mm-per-prompt.video 0 \
  --async-scheduling \
  --gpu-memory-utilization 0.95 \
  --max-num-seqs 128
```
</details>

<details>
<summary>A100 & H100 (Image Inference, BF16)</summary>
```bash
vllm serve Qwen/Qwen3-VL-235B-A22B-Instruct \
  --tensor-parallel-size 8 \
  --limit-mm-per-prompt.video 0 \
  --async-scheduling
```
</details>

<details>
<summary>A100 & H100 (Image + Video Inference, BF16)</summary>
```bash
vllm serve Qwen/Qwen3-VL-235B-A22B-Instruct \
  --tensor-parallel-size 8 \
  --max-model-len 128000 \
  --async-scheduling
```
</details>

<details>
<summary>H200 & B200</summary>
```bash
vllm serve Qwen/Qwen3-VL-235B-A22B-Instruct \
  --tensor-parallel-size 8 \
  --mm-encoder-tp-mode data \
  --async-scheduling
```
</details>

> ℹ️ **Note**  
> Qwen3-VL-235B-A22B-Instruct also excels on text-only tasks, ranking as the [#1 open model on text by lmarena.ai](https://x.com/arena/status/1973151703563460942) at the time this guide was created.  
> You can enable text-only mode by passing `--limit-mm-per-prompt.video 0 --limit-mm-per-prompt.image 0`, which skips the vision encoder and multimodal profiling to free up memory for additional KV cache.


### Configuration Tips
- It's highly recommended to specify `--limit-mm-per-prompt.video 0` if your inference server will only process image inputs since enabling video inputs consumes more memory reserved for long video embeddings. Alternatively, you can skip memory profiling for multimodal inputs by `--skip-mm-profiling` and lower `--gpu-memory-utilization` accordingly at your own risk.
- To avoid undesirable CPU contention, it's recommended to limit the number of threads allocated to preprocessing by setting the environment variable `OMP_NUM_THREADS=1`. This is particulaly useful and shows significant throughput improvement when deploying multiple vLLM instances on the same host.
- You can set `--max-model-len` to preserve memory. By default the model's context length is 262K, but `--max-model-len 128000` is good for most scenarios.
- Specifying `--async-scheduling` improves the overall system performance by overlapping scheduling overhead with the decoding process. **Note: With vLLM >= 0.11.1, compatibility has been improved for structured output and sampling with penalties, but it may still be incompatible with speculative decoding (features merged but not yet released).** Check the latest releases for continued improvements.
- Specifying `--mm-encoder-tp-mode data` deploys the vision encoder in a data-parallel fashion for better performance. This is because the vision encoder is very small, thus tensor parallelism brings little gain but incurs significant communication overhead. Enabling this feature does consume additional memory and may require adjustment on `--gpu-memory-utilization`.
- If your workload involves mostly **unique** multimodal inputs only, it is recommended to pass `--mm-processor-cache-gb 0` to avoid caching overhead. Otherwise, specifying `--mm-processor-cache-type shm` enables this experimental feature which utilizes host shared memory to cache preprocessed input images and/or videos which shows better performance at a high TP setting.
- vLLM supports Expert Parallelism (EP) via `--enable-expert-parallel`, which allows experts in MoE models to be deployed on separate GPUs for better throughput. Check out [Expert Parallelism Deployment](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment.html) for more details.
- You can use [benchmark_moe](https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_moe.py) to perform MoE Triton kernel tuning for your hardware.
- You can further extend the model's context window with `YaRN` by passing `--rope-scaling '{"rope_type":"yarn","factor":3.0,"original_max_position_embeddings": 262144,"mrope_section":[24,20,20],"mrope_interleaved": true}' --max-model-len 1000000`


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



## AMD GPU Support 
Please follow the steps here to install and run Qwen3-VL models on AMD MI300X GPU.
### Step 1: Prepare Docker Environment
Pull the latest vllm docker:
```shell
docker pull rocm/vllm-dev:nightly
```
Launch the ROCm vLLM docker: 
```shell
docker run -it --ipc=host --network=host --privileged --cap-add=CAP_SYS_ADMIN --device=/dev/kfd --device=/dev/dri --device=/dev/mem --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v $(pwd):/work -e SHELL=/bin/bash  --name Qwen3-VL rocm/vllm-dev:nightly 
```
### Step 2: Log in to Hugging Face
Huggingface login
```shell
huggingface-cli login
```

### Step 3: Start the vLLM server

Run the vllm online serving

#### Inside the Docker container, create a new directory named `miopen` under `/app/`.
```shell
mkdir -p /app/miopen

Sample Command
```shell


MIOPEN_USER_DB_PATH=/app/miopen \
MIOPEN_FIND_MODE=FAST \
VLLM_USE_V1=1 \
VLLM_ROCM_USE_AITER=1 \
SAFETENSORS_FAST_GPU=1 \
vllm serve Qwen/Qwen3-VL-235B-A22B-Instruct \
--tensor-parallel  4 \
--mm-encoder-tp-mode "data" \
--no-enable-prefix-caching \
--trust-remote-code

```


### Step 4: Run Benchmark
Open a new terminal and run the following command to execute the benchmark script inside the container.
```shell
docker exec -it Qwen3-VL vllm bench serve \
  --model "Qwen/Qwen3-VL-235B-A22B-Instruct" \
  --dataset-name random \
  --random-input-len 8192 \
  --random-output-len 1024 \
  --request-rate 10000 \
  --num-prompts 16 \
  --ignore-eos \
  --trust-remote-code 
```


  
