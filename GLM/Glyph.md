# Glyph Usage Guide

## Introduction
[Glyph](https://github.com/thu-coai/Glyph) is a framework from Zhipu AI for scaling the context length through visual-text compression. It renders long textual sequences into images and processes them using vision–language models. In this guide, we demonstrate how to use vLLM to deploy the [zai-org/Glyph](https://huggingface.co/zai-org/Glyph) model as a key component in this framework for image understanding tasks.

## Installing vLLM

```bash
uv venv
source .venv/bin/activate
uv pip install -U vllm --torch-backend auto
```

## Deploying Glyph

```bash
vllm serve zai-org/Glyph \
    --no-enable-prefix-caching \
    --mm-processor-cache-gb 0 \
    --reasoning-parser glm45 \
    --limit-mm-per-prompt.video 0
```

## Configuration Tips
- `zai-org/Glyph` itself is a reasoning multimodal model, therefore we recommend using `--reasoning-parser glm45` for parsing reasoning traces from model outputs.
- Unlike multi-turn chat use cases, we do not expect OCR tasks to benefit significantly from prefix caching or image reuse, therefore it's recommended to turn off these features to avoid unnecessary hashing and caching.
- Depending on your hardware capability, adjust `max_num_batched_tokens` for better throughput performance.
- Check out the [official Glyph documentation](https://github.com/thu-coai/Glyph?tab=readme-ov-file#model-deployment-vllm-acceleration) for more details on utilizing the vLLM deployment inside the end-to-end Glyph framework.


## AMD GPU Support

Please follow the steps here to install and run Glyph models on AMD MI300X, MI325X, MI355X GPUs.

### Step 1: Prepare Environment
#### Option 1: Installation from pre-built wheels (For AMD ROCm: MI300x/MI325x/MI355x)
We recommend using the official package for AMD GPUs (MI300x/MI325x/MI355x). 
```bash
uv pip install vllm --extra-index-url https://wheels.vllm.ai/rocm
```
⚠️ The vLLM wheel for ROCm is compatible with Python 3.12, ROCm 7.0, and glibc >= 2.35. If your environment is incompatible, please use docker flow in [vLLM](https://vllm.ai/).

#### Option 2: Docker image
Pull the latest vllm docker:

```bash
docker pull vllm/vllm-openai-rocm:latest
```

Launch the ROCm vLLM docker: 

```bash
docker run -it \
  --ipc=host \
  --network=host \
  --privileged \
  --cap-add=CAP_SYS_ADMIN \
  --device=/dev/kfd \
  --device=/dev/dri \
  --device=/dev/mem \
  --group-add video \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -v $(pwd):/work \
  -e bash=/bin/bash \
  --name Glyph \
  vllm/vllm-openai-rocm:latest
```

After running the command above, you are already inside the container. Proceed to Step 2 in that shell. If you detached from the container or started it in detached mode, attach to the container with:

```bash
docker attach Glyph
```

### Step 2: Log in to Hugging Face

Hugging Face login:

```bash
huggingface-cli login
```

### Step 3: Start the vLLM server

Run the vllm online serving
Sample Command
```bash
VLLM_ROCM_USE_AITER=1 \
SAFETENSORS_FAST_GPU=1 \
VLLM_USE_TRITON_FLASH_ATTN=0 \
vllm serve zai-org/Glyph \
    --no-enable-prefix-caching \
    --mm-processor-cache-gb 0 \
    --reasoning-parser glm45 \
    --limit-mm-per-prompt.video 0
```

### Step 4: Run Benchmark
Open a new terminal and run the following command to execute the benchmark script:

```bash
vllm bench serve \
  --model zai-org/Glyph \
  --dataset-name random \
  --random-input-len 8192 \
  --random-output-len 512 \
  --request-rate 10000 \
  --num-prompts 16 \
  --ignore-eos
```

If you are using a Docker environment, open a new terminal and run the benchmark inside the container with:

```bash
docker exec -it Glyph vllm bench serve \
  --model zai-org/Glyph \
  --dataset-name random \
  --random-input-len 8192 \
  --random-output-len 512 \
  --request-rate 10000 \
  --num-prompts 16 \
  --ignore-eos
```
