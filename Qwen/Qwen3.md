# Qwen3 Usage Guide

## Introduction

This guide provides step-by-step instructions for running the Qwen3 series using vLLM. The guide is intended for developers and practitioners seeking high-throughput or low-latency inference on the targeted accelerated stack.

### TPU Deployment

- [Qwen3-32B on Trillium (v6e)](https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/inference/trillium/vLLM/Qwen3)
- [Qwen3-4B on Trillium (v6e)](https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/inference/trillium/vLLM/Qwen3)


## AMD GPU Support
Recommended approaches by hardware type are:


MI300X/MI325X/MI355X  with fp8: Use FP8 checkpoint for optimal memory efficiency.

- **MI300X/MI325X/MI355X with `fp8`**: Use FP8 checkpoint for optimal memory efficiency.
- **MI300X/MI325X/MI355X with `bfloat16`**


Please follow the steps here to install and run Qwen3 models on AMD MI300X/MI325X/MI355X GPU.

### Step 1: Prepare Docker Environment
Pull the latest vllm docker:
```shell
docker pull vllm/vllm-openai-rocm:v0.14.1
```
Launch the ROCm vLLM docker: 
```shell

docker run -d -it --entrypoint /bin/bash --ipc=host --network=host --privileged --cap-add=CAP_SYS_ADMIN --device=/dev/kfd --device=/dev/dri --device=/dev/mem --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v /:/work   -v ~/.cache/huggingface:/root/.cache/huggingface -p 8000:8000 --name Qwen3 vllm/vllm-openai-rocm:v0.14.1
```
### Step 2: Log in to Hugging Face
Log in to your Hugging Face account:
```shell
hf auth login
```

### Step 3: Start the vLLM server

Run the vllm online serving
```shell
docker exec -it Qwen3 /bin/bash 
```

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
Open a new terminal and run the following command to execute the benchmark script inside the container.
```shell
docker exec -it Qwen3 vllm bench serve \
  --model "Qwen/Qwen3-235B-A22B-FP8" \
  --dataset-name random \
  --random-input-len 8192 \
  --random-output-len 1024 \
  --request-rate 10000 \
  --num-prompts 16 \
  --ignore-eos \
  --trust-remote-code 
```
