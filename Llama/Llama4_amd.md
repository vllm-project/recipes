# Llama 4 Maverick & Scout on vLLM - AMD Hardware

## Introduction

This quick start recipe explains how to run Llama 4 Scout 16 experts and Maverick 128 experts models on MI300X and MI355X GPUs. 

## Key benefits of AMD GPUs on large models and developers

The AMD Instinct GPUs accelerators are purpose-built to handle the demands of next-gen models like Llama 4:
- MI300X and MI325X can run the full 400B-parameter Llama 4 Maverick model in BF16 on a single node, reducing infrastructure complexity.
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
alias drun='sudo docker run -it --network=host --device=/dev/kfd --device=/dev/dri --group-add=video --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --shm-size 32G -v /data:/data -v $HOME:/myhome -w /myhome'
drun rocm/vllm-dev:nightly
``` 

### 2. Start vLLM online server (run in background)

```bash
export TP=8 
#export MODEL="meta-llama/Llama-4-Maverick-17B-128E-Instruct" 
export MODEL="meta-llama/Llama-4-Scout-17B-16E-Instruct"
vllm serve $MODEL \
  --disable-log-requests \
  -tp $TP \
  --max-num-seqs 64 \
  --no-enable-prefix-caching \
  --max_num_batched_tokens=16384 \
  --max_model_len 32000 &
``` 

### 3. Runing Inference using benchmark script

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

The expected results are like this: 
```bash
"text":"assistant\n\nThe two images you've shared appear to be quite distinct in their content and style. Here's a description of their similarities and differences:\n\n### Similarities:\n1. **Subject Matter**: Both images feature animals. The first image is of a rabbit, and the second seems to be a cat or a stylized representation of a cat. Both are common household pets, which could be a point of similarity in their subject matter.\n\n2. **Image Style**: Both images seem to be digital creations. The rabbit appears to be rendered in a photorealistic style, while the cat is presented in a more stylized or artistic layout, possibly indicating a graphic design or digital art approach.\n\n### Differences:\n1. **Content and Realism**: The most obvious difference is the realism and detail in the images. The rabbit image appears to be highly detailed and realistic, suggesting it could be a photograph or a highly detailed digital rendering. In contrast, the cat image is more stylized, with a simplified layout that might be used in graphic design, advertising, or educational materials.\n\n2. **Purpose and Context**: The purpose and context of the images seem different. The rabbit image, given its realism, could be used in a variety of contexts, such as in articles, as a
``` 

### 4. Accuracy check with lm_eval

```bash
pip install lm-eval[api]
lm_eval --model local-completions --model_args model=meta-llama/Llama-4-Scout-17B-16E-Instruct,base_url=http://localhost:8000/v1/completions,num_concurrent=256,max_retries=2,max_gen_toks=2048 --tasks gsm8k --batch_size auto --num_fewshot 5 --trust_remote_code --apply_chat_template
``` 

The expected gsm8k results are like this: 
```bash
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9553|±  |0.0057|
|     |       |strict-match    |     5|exact_match|↑  |0.0804|±  |0.0075|
``` 

### 5. Performance benchmark 

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