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

# Glyph on AMD GPU
Please follow the steps here to install and run Glyph models on AMD MI300X GPU.
### Step 1: Prepare Docker Environment
Pull the latest vllm docker:
```shell
docker pull rocm/vllm-dev:nightly
```
Launch the ROCm vLLM docker: 
```shell
docker run -it --ipc=host --network=host --privileged --cap-add=CAP_SYS_ADMIN --device=/dev/kfd --device=/dev/dri --device=/dev/mem --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v $(pwd):/work -e SHELL=/bin/bash  --name Glyph rocm/vllm-dev:nightly 
```
### Step 2: Log in to Hugging Face
Huggingface login
```shell
huggingface-cli login
```

### Step 3: Start the vLLM server

Run the vllm online serving
Sample Command
```shell


SAFETENSORS_FAST_GPU=1 \
VLLM_USE_TRITON_FLASH_ATTN=0 vllm serve zai-org/Glyph \
    --no-enable-prefix-caching \
    --mm-processor-cache-gb 0 \
    --reasoning-parser glm45 \
    --limit-mm-per-prompt.video 0
	
```


### Step 4: Run Benchmark
Open a new terminal and run the following command to execute the benchmark script inside the container.
```shell
docker exec -it Glyph vllm bench serve \
  --model "zai-org/Glyph" \
  --dataset-name random \
  --random-input-len 8192 \
  --random-output-len 512 \
  --request-rate 10000 \
  --num-prompts 16 \
  --ignore-eos
```
