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
alias drun='sudo docker run -it --network=host --device=/dev/kfd --device=/dev/dri --group-add=video --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --shm-size 32G -v /data:/data -v $HOME:/myhome -w /myhome --entrypoint /bin/bash'
drun vllm/vllm-openai-rocm:v0.14.0
```

### 2. Start vLLM online server (run in background)

```bash
export TP=1
export VLLM_ROCM_USE_AITER=1
export MODEL="mistralai/Ministral-3-14B-Instruct-2512"
vllm serve $MODEL \
  --disable-log-requests \
  --port 9090 \
  -tp $TP \
  --config_format mistral \
  --load_format mistral \
  --enable-auto-tool-choice \
  --tool-call-parser mistral &
```

### 3. Running Inference using benchmark script

Test the model with a text-only prompt.

```bash
curl http://localhost:9090/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "mistralai/Ministral-3-14B-Instruct-2512",
        "prompt": "Explain the benefits of KV cache in transformer decoding.",
        "max_tokens": 128,
        "temperature": 0
    }'
```

Test result (local run):
```bash
"text":" How does it help in reducing the computational cost?\n\n### Understanding KV Cache in Transformer Decoding\n\nThe **KV cache** (Key-Value cache) is a technique used in **transformer-based models** (like GPT, BERT, etc.) during **autoregressive decoding** to improve efficiency. Here's how it works and why it's beneficial:\n\n---\n\n### **1. What is KV Cache?**\nDuring decoding, a transformer model generates tokens one by one. For each new token, the model computes **attention scores** between the current token and all previous tokens in the sequence. The attention mechanism involves:\n"
```

### 4. Performance benchmark

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
  --port 9090 \
  --percentile-metrics ttft,tpot,itl,e2el
```