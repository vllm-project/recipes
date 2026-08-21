# Quick Start Recipe for Llama 3.1 on vLLM

## Introduction

This quick start recipe provides step-by-step instructions for running the Llama 3.1 Instruct model using vLLM. The recipe is intended for developers and practitioners seeking high-throughput or low-latency inference on the targeted accelerated stack.

## Access & Licensing

To use the Llama 3.1 model, you must first gain access to the model repo under Hugging Face.
- [Llama 3.1 8B Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)

## Prerequisites

### AMD GPU
- OS: Linux
- Drivers: ROCm 7.0 or above
- GPU: AMD MI300X, MI325X, MI350X, MI355X

## Deployment

### TPU Deployment

- [Llama3.x-70B on Trillium (v6e)](https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/inference/trillium/vLLM/Llama3.x)
- [Llama3.1-8B on Trillium (v6e)](https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/inference/trillium/vLLM/Llama3.x)

### AMD GPU (ROCm)

#### Using vLLM Docker Image

```bash
docker run -it \
  --network=host \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add=video \
  --ipc=host \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --shm-size 32G \
  -v /data:/data \
  -v $HOME:/myhome \
  -w /myhome \
  --entrypoint /bin/bash \
  vllm/vllm-openai-rocm:latest
```

Or install using uv environment:

> Note: The vLLM wheel for ROCm requires Python 3.12, ROCm 7.0, and glibc >= 2.35. If your environment does not meet these requirements, please use the Docker-based setup as described in the [documentation](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/#pre-built-images).

```bash
uv venv
source .venv/bin/activate
uv pip install vllm --extra-index-url https://wheels.vllm.ai/rocm/
```

#### Launch vLLM Server

You can use 1x MI300X/MI325X/MI350X/MI355X GPU to launch this model with [AITER](https://github.com/ROCm/aiter) acceleration enabled:

```bash
export TP=1
export VLLM_ROCM_USE_AITER=1
export MODEL="meta-llama/Llama-3.1-8B-Instruct"
vllm serve $MODEL \
  --disable-log-requests \
  -tp $TP &
```

> **Note**: The first launch with AITER may take several minutes as AITER JIT-compiles optimized kernels (CK-based FP8 MoE, RMSNorm, activation, etc.). Subsequent launches will use cached kernels.

#### Performance Benchmark

```bash
export MODEL="meta-llama/Llama-3.1-8B-Instruct"
export ISL=1024
export OSL=1024
export REQ=10
export CONC=10
vllm bench serve \
  --backend vllm \
  --model $MODEL \
  --dataset-name random \
  --random-input-len $ISL \
  --random-output-len $OSL \
  --num-prompts $REQ \
  --ignore-eos \
  --max-concurrency $CONC \
  --percentile-metrics ttft,tpot,itl,e2el
```
