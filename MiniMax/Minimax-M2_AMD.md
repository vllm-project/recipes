# MiniMax M2 on vLLM - AMD Hardware

## Introduction

This quick start recipe explains how to run the MiniMax M2 model on AMD MI300X/MI355X GPUs using vLLM.

## Key benefits of AMD GPUs on large models and developers

The AMD Instinct GPUs accelerators are purpose-built to handle the demands of next-gen models like MiniMax M2:
- Large HBM memory enables long-context inference and larger batch sizes.
- Optimized Triton and AITER kernels provide best-in-class performance and TCO for production deployment.

## Access & Licensing

### License and Model parameters

To use the MiniMax M2 model, please check whether you have access to the following model:
- [MiniMax M2](https://huggingface.co/MiniMaxAI/MiniMax-M2)

## Prerequisites

- OS: Linux
- Drivers: ROCm 7.0 or above
- GPU: AMD MI300X, MI325X, and MI355X

## Deployment Steps

### 1. Using vLLM docker image (For AMD users)

```bash
alias drun='sudo docker run -it --network=host --device=/dev/kfd --device=/dev/dri --group-add=video --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --shm-size 32G -v /data:/data -v $HOME:/myhome -w /myhome --entrypoint /bin/bash'
drun vllm/vllm-openai-rocm:v0.14.1
```

### 2. Start vLLM online server (run in background)

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3
export VLLM_ROCM_USE_AITER=1
vllm serve MiniMaxAI/MiniMax-M2 \
  --tensor-parallel-size 4 \
  --tool-call-parser minimax_m2 \
  --reasoning-parser minimax_m2_append_think \
  --enable-auto-tool-choice \
  --trust-remote-code \
  --disable-log-requests \
  --port 8007 &
```

### 3. Running Inference using benchmark script

Test the model with a text-only prompt.

```bash
curl http://localhost:8007/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "MiniMaxAI/MiniMax-M2",
        "prompt": "Explain how speculative decoding can improve LLM throughput.",
        "max_tokens": 128,
        "temperature": 0
    }'
```

### 4. Performance benchmark

```bash
export MODEL="MiniMaxAI/MiniMax-M2"
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
  --port 8007 \
  --percentile-metrics ttft,tpot,itl,e2el
```
