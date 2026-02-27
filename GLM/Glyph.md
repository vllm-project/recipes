# Glyph Usage Guide

## Introduction
[Glyph](https://github.com/thu-coai/Glyph) is a framework from Zhipu AI for scaling the context length through visual-text compression. It renders long textual sequences into images and processes them using vision–language models. In this guide, we demonstrate how to use vLLM to deploy the [zai-org/Glyph](https://huggingface.co/zai-org/Glyph) model as a key component in this framework for image understanding tasks.

## Installing vLLM

```bash
uv venv
source .venv/bin/activate
uv pip install -U vllm --torch-backend auto
```

## Installing vLLM (For AMD ROCm: MI300x/MI325x/MI355x)
We recommend to use the official package for AMD GPUs (MI300x/MI325x/MI355x). 
```bash
uv pip install vllm --extra-index-url https://wheels.vllm.ai/rocm
```
⚠️ The vLLM wheel for ROCm is compatible with Python 3.12, ROCm 7.0, and glibc >= 2.35. If your environment is incompatible, please use docker flow in [vLLM](https://vllm.ai/).

## Deploying Glyph

### Serving Glyph Model on 1xH100 GPU
```bash
vllm serve zai-org/Glyph \
    --no-enable-prefix-caching \
    --mm-processor-cache-gb 0 \
    --reasoning-parser glm45 \
    --limit-mm-per-prompt.video 0
```

### Serving Glyph Model on 1xMI300x/MI325x
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

### Configuration Tips
- `zai-org/Glyph` itself is a reasoning multimodal model, therefore we recommend using `--reasoning-parser glm45` for parsing reasoning traces from model outputs.
- Unlike multi-turn chat use cases, we do not expect OCR tasks to benefit significantly from prefix caching or image reuse, therefore it's recommended to turn off these features to avoid unnecessary hashing and caching.
- Depending on your hardware capability, adjust `max_num_batched_tokens` for better throughput performance.
- Check out the [official Glyph documentation](https://github.com/thu-coai/Glyph?tab=readme-ov-file#model-deployment-vllm-acceleration) for more details on utilizing the vLLM deployment inside the end-to-end Glyph framework.


## Run Benchmark
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
