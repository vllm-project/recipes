# Ministral 3 14B Instruct on vLLM - AMD Hardware

## Introduction

This quick start recipe explains how to run the Ministral 3 14B Instruct model on AMD MI300X/MI355X GPUs using vLLM.

## Key benefits of AMD GPUs on large models and developers

The AMD Instinct GPUs accelerators are purpose-built to handle the demands of next-gen models like Ministral:
- Large HBM memory enables longer contexts and higher concurrency.
- Optimized Triton and AITER kernels provide best-in-class performance and TCO for production deployment.
- Strong single-node performance reduces infrastructure complexity for serving.

## Access & Licensing

### License and Model parameters

Please check whether you have access to the following model:
- [Ministral 3 14B Instruct](https://huggingface.co/mistralai/Ministral-3-14B-Instruct-2512)

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

### 2. Start vLLM online server (run in background)

```bash
export TP=1
export VLLM_ROCM_USE_AITER=1
export MODEL="mistralai/Ministral-3-14B-Instruct-2512"
vllm serve $MODEL \
  --disable-log-requests \  
  -tp $TP \
  --config_format mistral \
  --load_format mistral \
  --enable-auto-tool-choice \
  --tool-call-parser mistral &
```

### 3. Performance benchmark

```bash
export MODEL="mistralai/Ministral-3-14B-Instruct-2512"
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
