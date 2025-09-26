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
This is the Qwen3-VL flagship MoE model and it requires at least 8 GPUs with at least 80GB of memory each (e.g., H100 or H200) to run.

To launch an online inference server for `Qwen3-VL-235B-A22B-Instruct`:

#### A100 & H100 (Image Inference)
```bash
vllm serve Qwen/Qwen3-VL-235B-A22B-Instruct \
  --tensor-parallel-size 8 \
  --limit-mm-per-prompt.video 0 \
  --mm-processor-cache-type shm \
  --async-scheduling
```

#### A100 & H100 (Image + Video Inference)
```bash
vllm serve Qwen/Qwen3-VL-235B-A22B-Instruct \
  --tensor-parallel-size 8 \
  --max-num-batched-tokens 2048 \
  --max-num-seqs 128 \
  --max-model-len 128000 \
  --mm-processor-cache-type shm \
  --async-scheduling
```

#### H100 (Image + Video Inference + FP8)
```bash
vllm serve Qwen/Qwen3-VL-235B-A22B-Instruct \
  --tensor-parallel-size 8 \
  --max-num-batched-tokens 2048 \
  --max-num-seqs 128 \
  --mm-processor-cache-type shm \
  --quantization fp8 \
  --async-scheduling
```

#### H200 (Image Inference)
```bash
vllm serve Qwen/Qwen3-VL-235B-A22B-Instruct \
  --tensor-parallel-size 8 \
  --limit-mm-per-prompt.video 0 \
  --mm-encoder-tp-mode data \
  --mm-processor-cache-type shm \
  --async-scheduling
```

#### H200 (Image + Video Inference)
```bash
vllm serve Qwen/Qwen3-VL-235B-A22B-Instruct \
  --tensor-parallel-size 8 \
  --mm-processor-cache-type shm \
  --async-scheduling
```

####  Usage Tips
- It's highly recommended to specify `--limit-mm-per-prompt.video 0` if your inference server will only process image inputs since enabling video inputs will consume additional memory reserved for long video embeddings. Alternatively, you can skip memory profiling for multimodal inputs by `--skip-mm-profiling` and lower `--gpu-memory-utilization` accordingly at your own risk.
- You can set `--max-model-len` to preserve memory. By default the model's context length is 262K, but `--max-model-len=128000` is usually good for most scenarios.
- Specifying `--mm-encoder-tp-mode data` deploys the vision encoder in a data-parallel fashion for better performance. This is because the vision encoder is very small compared to the language decoder, thus tensor parallelism brings little gain but incurs significant communication overhead. However, for long-video input this may overload individual accelerators.
- Specifying `--mm-processor-cache-type shm` utilizes host shared memory to cache preprocessed input images and/or videos. If your workload involves only unique multimodal inputs, you may remove this argument and pass `--mm-processor-cache-gb 0` instead.
- You can use [benchmark_moe](https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_moe.py) to perform MoE Triton kernel tuning for your hardware.


#### Benchmark on VisionArena-Chat Dataset

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

For more examples, please refer to the official [Qwen3-VL GitHub Repository](https://github.com/QwenLM/Qwen3-VL).
