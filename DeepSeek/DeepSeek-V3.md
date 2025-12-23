# DeepSeek-V3 (R1) Usage Guide

This guide describes how to run DeepSeek-V3 or DeepSeek-R1 with native FP8 or FP4. 
In the guide, we use DeepSeek-R1 as an example, but the same applies to DeepSeek-V3 given they have the same model architecture.

## Installing vLLM

```bash
uv venv
source .venv/bin/activate
uv pip install -U vllm --torch-backend auto
```

## Running DeepSeek-R1 with FP8 on 8xH200

There are two ways to parallelize the model over multiple GPUs: (1) Tensor-parallel or (2) Data-parallel. Each one has its own advantages, where tensor-parallel is usually more beneficial for low-latency / low-load scenarios and data-parallel works better for cases where there is a lot of data with heavy-loads.

For blackwell, first enable flashinfer env vars:

```bash
export VLLM_ATTENTION_BACKEND=CUTLASS_MLA
export VLLM_USE_FLASHINFER_MOE_FP8=1
```

Then, run tensor-parallel like this:

```bash

# Start server with FP8 model on 8 GPUs
vllm serve deepseek-ai/DeepSeek-R1-0528 \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --enable-expert-parallel
```

Or data-parallel like this:

```bash
# Start server with FP8 model on 8 GPUs
vllm serve deepseek-ai/DeepSeek-R1-0528 \
  --trust-remote-code \
  --data-parallel-size 8 \
  --enable-expert-parallel
```

Note that in both cases we enable expert-parallel as well, so the first run is what we call TP+EP and the second one is DP+EP.

Additional flags:

* For non-flashinfer runs, one can use VLLM_USE_DEEP_GEMM and VLLM_ALL2ALL_BACKEND. For example:
```bash
export VLLM_USE_DEEP_GEMM=1 
export VLLM_ALL2ALL_BACKEND="deepep_high_throughput" # or "deepep_low_latency"
```
* You can set `--max-model-len` to preserve memory. `--max-model-len=65536` is usually good for most scenarios.
* You can set `--max-num-batched-tokens` to balance throughput and latency, higher means higher throughput but higher latency. `--max-num-batched-tokens=32768` is usually good for prompt-heavy workloads. But you can reduce it to 16k and 8k to reduce activation memory usage and decrease latency.
* vLLM conservatively use 90% of GPU memory, you can set `--gpu-memory-utilization=0.95` to maximize KVCache.

## Running DeepSeek-R1 with FP4 on 4xB200

For Blackwell GPUs, add these environment variables before running:

```bash
export VLLM_ATTENTION_BACKEND=CUTLASS_MLA
export VLLM_USE_FLASHINFER_MOE_FP4=1
```

Then run tensor-parallel:

```bash
# The model is runnable on 4 or 8 GPUs, here we show usage of 4.
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve nvidia/DeepSeek-R1-FP4 \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --enable-expert-parallel
```

Or, data-parallel:
```bash
# The model is runnable on 4 or 8 GPUs, here we show usage of 4.
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve nvidia/DeepSeek-R1-FP4 \
  --trust-remote-code \
  --data-parallel-size 4 \
  --enable-expert-parallel
```

## Benchmarking

For benchmarking, disable prefix caching by adding `--no-enable-prefix-caching` to the server command.

### FP8 Benchmark

```bash
# Prompt-heavy benchmark (8k/1k)
vllm bench serve \
  --model deepseek-ai/DeepSeek-R1-0528 \
  --dataset-name random \
  --random-input-len 8000 \
  --random-output-len 1000 \
  --request-rate 10000 \
  --num-prompts 16 \
  --ignore-eos
```

### FP4 Benchmark

```bash
# Prompt-heavy benchmark (8k/1k)
vllm bench serve \
  --model nvidia/DeepSeek-R1-FP4 \
  --dataset-name random \
  --random-input-len 8000 \
  --random-output-len 1000 \
  --request-rate 10000 \
  --num-prompts 16 \
  --ignore-eos
```

### Benchmark Configurations

Test different workloads by adjusting input/output lengths:

- **Prompt-heavy**: 8000 input / 1000 output
- **Decode-heavy**: 1000 input / 8000 output  
- **Balanced**: 1000 input / 1000 output

Test different batch sizes by changing `--num-prompts`:

- Batch sizes: 1, 16, 32, 64, 128, 256, 512

### Expected Output

```shell
============ Serving Benchmark Result ============
Successful requests:                     1         
Benchmark duration (s):                  16.39     
Total input tokens:                      7902      
Total generated tokens:                  1000      
Request throughput (req/s):              0.06      
Output token throughput (tok/s):         61.00     
Total Token throughput (tok/s):          543.06    
---------------Time to First Token----------------
Mean TTFT (ms):                          560.00    
Median TTFT (ms):                        560.00    
P99 TTFT (ms):                           560.00    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          15.85     
Median TPOT (ms):                        15.85     
P99 TPOT (ms):                           15.85     
---------------Inter-token Latency----------------
Mean ITL (ms):                           15.85     
Median ITL (ms):                         15.85     
P99 ITL (ms):                            16.15     
==================================================
```

### AMD GPU Support

Please follow the steps here to install and run DeepSeek-R1 models on AMD MI300X GPU.
### Step 1: Prepare Docker Environment
Pull the latest vllm docker:
```shell
docker pull rocm/vllm-dev:nightly
```
Launch the ROCm vLLM docker: 
```shell
docker run -it \
  --ipc=host \
  --network=host \
  --privileged \
  --cap-add=CAP_SYS_ADMIN \
  --device=/dev/kfd \
  --device=/dev/dri \
  --device=/dev/mem \
  --group-add video \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -v $(pwd):/work \
  -e SHELL=/bin/bash \
  --name DeepSeek-R1 \
  rocm/vllm-dev:nightly
```
### Step 2: Log in to Hugging Face
Huggingface login
```shell
huggingface-cli login
```

### Step 3: Start the vLLM server 

Run the vllm online serving
Sample Command
```shell

SAFETENSORS_FAST_GPU=1 \
NCCL_MIN_NCHANNELS=112 \
VLLM_ROCM_USE_AITER=1 \
VLLM_ROCM_USE_AITER_RMSNORM=0 \
VLLM_ROCM_USE_AITER_MHA=0 \
VLLM_V1_USE_PREFILL_DECODE_ATTENTION=1 \
VLLM_USE_TRITON_FLASH_ATTN=0 \
VLLM_RPC_TIMEOUT=18000000 \
vllm serve "deepseek-ai/DeepSeek-R1" \
--tensor-parallel-size 8 \
--enable-expert-parallel \
--max-model-len 32768 \
--max-num-seqs 1024 \
--max-num-batched-tokens 32768 \
--disable-log-requests \
--block-size 1 \
--trust-remote-code
```

### Step 4: Run Benchmark
Open a new terminal and run the following command to execute the benchmark script inside the container.
```shell
docker exec -it DeepSeek-R1 vllm bench serve \
  --model "deepseek-ai/DeepSeek-R1" \
  --dataset-name random \
  --random-input-len 8000 \
  --random-output-len 1000 \
  --request-rate 10000 \
  --num-prompts 16 \
  --ignore-eos
```
