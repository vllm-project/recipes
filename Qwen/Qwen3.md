# Qwen3 Usage Guide

## Introduction

This guide provides step-by-step instructions for running the Qwen3 series using vLLM. The guide is intended for developers and practitioners seeking high-throughput or low-latency inference on the targeted accelerated stack.

### TPU Deployment

- [Qwen3-32B on Trillium (v6e)](https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/inference/trillium/vLLM/Qwen3)
- [Qwen3-4B on Trillium (v6e)](https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/inference/trillium/vLLM/Qwen3)


## AMD GPU Support
Recommended approaches by hardware type are:


MI300X/MI325X/MI355X 

Please follow the steps here to install and run Qwen3-Next models on AMD MI300X/MI325X/MI355X GPU.

### Step 1: Installing vLLM (AMD ROCm Backend: MI300X, MI325X, MI355X) 
 > Note: The vLLM wheel for ROCm requires Python 3.12, ROCm 7.0, and glibc >= 2.35. If your environment does not meet these requirements, please use the Docker-based setup as described in the [documentation](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/#pre-built-images).  
 ```bash 
 uv venv 
 source .venv/bin/activate 
 uv pip install vllm --extra-index-url https://wheels.vllm.ai/rocm/0.14.1/rocm700
 ```


### Step 2: Start the vLLM server

### BF16 


```shell
HIP_VISIBLE_DEVICES="4,5,6,7" \
VLLM_USE_V1=1 \
VLLM_ROCM_USE_AITER=1 \
VLLM_ROCM_USE_AITER_MHA=0 \
VLLM_V1_USE_PREFILL_DECODE_ATTENTION=1 \
VLLM_USE_TRITON_FLASH_ATTN=0 \
SAFETENSORS_FAST_GPU=1  \
vllm serve Qwen/Qwen3-235B-A22B \
--trust-remote-code \
-tp 4 \
--disable-log-requests \
--swap-space 32 \
--distributed-executor-backend mp \
--max-num-batched-tokens 32768 \
--max-model-len 32768 \
--no-enable-prefix-caching \
--gpu-memory-utilization 0.8
```

### FP8 

```shell

HIP_VISIBLE_DEVICES="4,5,6,7" \
VLLM_USE_V1=1 \
VLLM_ROCM_USE_AITER=1 \
VLLM_ROCM_USE_AITER_MHA=0 \
VLLM_V1_USE_PREFILL_DECODE_ATTENTION=1 \
VLLM_USE_TRITON_FLASH_ATTN=0 \
SAFETENSORS_FAST_GPU=1  \
vllm serve Qwen/Qwen3-235B-A22B-FP8 \
--trust-remote-code \
-tp 4 \
--disable-log-requests \
--swap-space 16 \
--distributed-executor-backend mp \
--max-num-batched-tokens 32768 \
--max-model-len 32768 \
--no-enable-prefix-caching \
--gpu-memory-utilization 0.8

```


### Step 4: Run Benchmark

```shell
vllm bench serve \
  --model "Qwen/Qwen3-235B-A22B-FP8" \
  --dataset-name random \
  --random-input-len 8192 \
  --random-output-len 1024 \
  --request-rate 10000 \
  --num-prompts 16 \
  --ignore-eos \
  --trust-remote-code 
```
