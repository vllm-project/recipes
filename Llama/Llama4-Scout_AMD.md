# Llama 4 Maverick & Scout on vLLM - AMD Hardware

## Introduction

This quick start recipe explains how to run Llama 4 Scout 16 experts and Maverick 128 experts models on MI300X and MI355X GPUs. 

## Key benefits of AMD GPUs on large models and developers

The AMD Instinct GPUs accelerators are purpose-built to handle the demands of next-gen models like Llama 4:
- Massive HBM memory capacity enables support for extended context lengths, delivering smooth and efficient performance.
- Using Optimized Triton and AITER kernels provide best-in-class performance and TCO for production deployment.

## Access & Licensing

### License and Model parameters

To use Llama 4 Scout and Maverick models, you must first need to gain access to the model repos under Huggingface.
- [Llama4 Scout 16E](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct)
- [Llama4 Maverick 128E](https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct)

## Prerequisites

- OS: Linux
- Drivers: ROCm 7.0 or above
- GPU: AMD MI300X, MI325X, and MI355X

## Deployment Steps

### 1. Using vLLM docker image (For AMD users)

```bash
alias drun='sudo docker run -it --network=host --device=/dev/kfd --device=/dev/dri --group-add=video --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --shm-size 32G -v /data:/data -v $HOME:/myhome -w /myhome --entrypoint /bin/bash'
drun vllm/vllm-openai-rocm:v0.14.1

### 2. Start vLLM online server (run in background)

```bash
export TP=8 
#export MODEL="meta-llama/Llama-4-Maverick-17B-128E-Instruct" 
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

### 3. Running Inference using benchmark script

Let the Ll4 Scout model to describe the following two images.
![first image](./images/rabbit.jpg)
![second image](./images/cat.png)

```bash
curl http://localhost:8000/v1/completions     -H "Content-Type: application/json"     -d '{
        "model": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "prompt": "<image>https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg</image><image>https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/datasets/cat_style_layout.png</image> Can you describe how these two images are similar, and how they differ?",
        "max_tokens": 256,
        "temperature": 0
    }'
``` 

### 4. Performance benchmark 

```bash
#export MODEL="meta-llama/Llama-4-Maverick-17B-128E-Instruct" 
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
