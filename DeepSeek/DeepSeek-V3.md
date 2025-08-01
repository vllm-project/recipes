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

```bash
# Start server with FP8 model on 8 GPUs
vllm serve deepseek-ai/DeepSeek-R1-0528 \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --enable-expert-parallel
```

Additional flags:
* You can set `--max-model-len` to preserve memory. `--max-model-len=65536` is usually good for most scenarios.
* You can set `--max-num-batched-tokens` to balance throughput and latency, higher means higher throughput but higher latency. `--max-num-batched-tokens=32768` is usually good for prompt-heavy workloads. But you can reduce it to 16k and 8k to reduce activation memory usage and decrease latency.
* vLLM conservatively use 90% of GPU memory, you can set `--gpu-memory-utilization=0.95` to maximize KVCache.

## Running DeepSeek-R1 with FP4 on 4xB200

For Blackwell GPUs, add these environment variables before running:

```bash
export VLLM_ATTENTION_BACKEND=CUTLASS_MLA_VLLM_V1
export VLLM_USE_FLASHINFER_MOE_FP8=1
export VLLM_USE_FLASHINFER_MOE_FP4=1


# The model is runnable on 4 or 8 GPUs, here we show usage of 4.
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve nvidia/DeepSeek-R1-FP4 \
  --trust-remote-code \
  --tensor-parallel-size 4 \
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
