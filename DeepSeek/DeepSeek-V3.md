# DeepSeek-V3 (R1) Usage Guide

This guide describes how to run DeepSeek-V3 or DeepSeek-R1 with native FP8 or FP4.
In the guide, we use DeepSeek-R1 as an example, but the same applies to DeepSeek-V3 given they have the same model architecture.

## Installing vLLM

```bash
uv venv
source .venv/bin/activate
uv pip install -U vllm --torch-backend auto
```

## Running DeepSeek-R1 / DeepSeek-V3

DeepSeek-R1 and DeepSeek-V3 share the same architecture, so the same serving setup applies. Two common configurations are:

- **8xH200 with `fp8`**: Native FP8 with TP+EP or DP+EP.
- **4xB200 with `fp4`**: Native FP4 with FlashInfer enabled (TP+EP or DP+EP).

See sections below for detailed launch arguments for each configuration.

### 8xH200 (FP8)
There are two ways to parallelize the model over multiple GPUs: (1) Tensor-parallel or (2) Data-parallel. Tensor-parallel is usually better for low-latency / low-load scenarios, while data-parallel works better for high-load workloads.

<details>
<summary>Tensor Parallel + Expert Parallel (TP8+EP)</summary>

```bash
vllm serve deepseek-ai/DeepSeek-R1-0528 \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --enable-expert-parallel
```

</details>

<details>
<summary>Data Parallel + Expert Parallel (DP8+EP)</summary>

```bash
vllm serve deepseek-ai/DeepSeek-R1-0528 \
  --trust-remote-code \
  --data-parallel-size 8 \
  --enable-expert-parallel
```

</details>

### 4xB200
For Blackwell GPUs, enable FlashInfer before running:

- **For FP4**: `export VLLM_USE_FLASHINFER_MOE_FP4=1`  (recommended)
- **For FP8**: `export VLLM_USE_FLASHINFER_MOE_FP8=1`

<details>
<summary>Tensor Parallel + Expert Parallel (TP4+EP)</summary>

```bash
# The model is runnable on 4 or 8 GPUs, here we show usage of 4.
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve nvidia/DeepSeek-R1-FP4 \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --enable-expert-parallel
```

</details>

<details>
<summary>Data Parallel + Expert Parallel (DP4+EP)</summary>

```bash
# The model is runnable on 4 or 8 GPUs, here we show usage of 4.
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve nvidia/DeepSeek-R1-FP4 \
  --trust-remote-code \
  --data-parallel-size 4 \
  --enable-expert-parallel
```

</details>


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

## Disaggregrated Serving with Wide EP (Experimental GB200)

Experimental disaggregrated serving recipes for NVIDIA GB200 can be found beyond:
- https://github.com/vllm-project/vllm/issues/33583
- https://blog.vllm.ai/2026/02/03/dsr1-gb200-part1.html
- https://github.com/minosfuture/vllm/tree/pd_gb200_0114/runs/DS-R1/fp4

