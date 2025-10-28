# Glyph Usage Guide

## Introduction
[Glyph](https://github.com/thu-coai/Glyph) is a framework from Zhipu AI for scaling the context length through visual-text compression. It renders long textual sequences into images and processes them using vision–language models. In this guide, we demonstrate how to use vLLM to deploy the [zai-org/Glyph](https://huggingface.co/zai-org/Glyph) model as a key component in this framework for image understanding tasks.

## Installing vLLM

```bash
uv venv
source .venv/bin/activate
uv pip install -U vllm
```

## Deploying Glyph

```bash
vllm serve zai-org/Glyph \
    --no-enable-prefix-caching \
    --mm-processor-cache-gb 0 \
    --reasoning-parser glm45
```

## Configuration Tips
- `zai-org/Glyph` itself is a reasoning multimodal model, therefore we recommend using `--reasoning-parser glm45` for parsing reasoning traces from model outputs.
- Unlike multi-turn chat use cases, we do not expect OCR tasks to benefit significantly from prefix caching or image reuse, therefore it's recommended to turn off these features to avoid unnecessary hashing and caching.
- Depending on your hardware capability, adjust `max_num_batched_tokens` for better throughput performance.
- Check out the [official Glyph documentation](https://github.com/thu-coai/Glyph?tab=readme-ov-file#model-deployment-vllm-acceleration) for more details on utilizing the vLLM deployment inside the end-to-end Glyph framework.
