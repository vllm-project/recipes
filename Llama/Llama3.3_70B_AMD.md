# Llama 3.3 70B Instruct on vLLM - AMD Hardware

## Introduction

This quick start recipe explains how to run the Llama 3.3 70B Instruct model on AMD MI300X/MI355X GPUs using vLLM.

## Key benefits of AMD GPUs on large models and developers

The AMD Instinct GPUs accelerators are purpose-built to handle the demands of next-gen models like Llama 3.3:
- Can run large 70B-parameter models with strong throughput on a single node.
- Massive HBM memory capacity enables support for extended context lengths and larger batch sizes.
- Using Optimized Triton and AITER kernels provide best-in-class performance and TCO for production deployment.

## Access & Licensing

### License and Model parameters

To use the Llama 3.3 model, you must first gain access to the model repo under Hugging Face.
- [Llama 3.3 70B Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct)

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
export TP=2
export MODEL="meta-llama/Llama-3.3-70B-Instruct"
export VLLM_ROCM_USE_AITER=1
vllm serve $MODEL \
  --disable-log-requests \
  --port 8005 \
  -tp $TP &  
```

### 3. Running Inference using benchmark script

Test the model with a text-only prompt.

```bash
curl http://localhost:8005/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "meta-llama/Llama-3.3-70B-Instruct",
        "prompt": "Summarize the key differences between throughput and latency in LLM serving.",
        "max_tokens": 128,
        "temperature": 0
    }'
```

### 4. Performance benchmark

```bash
export MODEL="meta-llama/Llama-3.3-70B-Instruct"
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
  --port 8005 \
  --percentile-metrics ttft,tpot,itl,e2el
```


