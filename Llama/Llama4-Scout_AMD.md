# Llama 4 Maverick & Scout on vLLM - AMD Hardware

## Introduction

This quick start recipe explains how to run Llama 4 Scout 16 experts and Maverick 128 experts models on MI300X and MI355X GPUs. 

## Key benefits of AMD GPUs on large models and developers

The AMD Instinct GPUs accelerators are purpose-built to handle the demands of next-gen models like Llama 4:
- Massive HBM memory capacity enables support for extended context lengths, delivering smooth and efficient performance.
- Using Optimized Triton and AITER kernels provide best-in-class performance and TCO for production deployment.

## Access & Licensing

### License and Model parameters

To use Llama 4 Scout model, you must first need to gain access to the model repos under Huggingface.
- [Llama4 Scout 16E](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct)


## Prerequisites

- OS: Linux
- Drivers: ROCm 7.0 or above
- GPU: AMD MI300X, MI325X, and MI355X

## Deployment Steps

### 1. Using vLLM docker image (For AMD users)

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
or you can use uv environment.
 > Note: The vLLM wheel for ROCm requires Python 3.12, ROCm 7.0, and glibc >= 2.35. If your environment does not meet these requirements, please use the Docker-based setup as described in the [documentation](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/#pre-built-images).  
 ```bash 
 uv venv 
 source .venv/bin/activate 
 uv pip install vllm --extra-index-url https://wheels.vllm.ai/rocm/
 ```
### 2. Start vLLM online server (run in background)

```bash
export TP=8 
export MODEL="meta-llama/Llama-4-Scout-17B-16E-Instruct"
export VLLM_ROCM_USE_AITER=1
vllm serve $MODEL \
  --disable-log-requests \
  -tp $TP \  
  --max-num-seqs 64 \
  --no-enable-prefix-caching \
  --max-num-batched-tokens=16384 \
  --max-model-len 32000 &
``` 

### 3. Performance benchmark 

```bash
export MODEL="meta-llama/Llama-4-Scout-17B-16E-Instruct"
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
