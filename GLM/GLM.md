# GLM-4.X LLM Usage Guide

GLM-4.X LLM include those model below:

+ GLM-4.7
+ GLM-4.6
+ GLM-4.5
+ GLM-4.5-Air

For GLM-V series, check [here](GLM-V.md)

This guide describes how to run GLM-4.X Series with native FP8 and BF16. FP8 models have minimal accuracy loss. 
Unless you need strict reproducibility for benchmarking or similar scenarios, we recommend using FP8 to run at a lower cost. These models have MTP layers. 

Here, we take GLM-4.5-Air as an example, and similarly, this applies to other models in the series.

## Installing vLLM

```bash
uv venv
source .venv/bin/activate
uv pip install -U vllm --torch-backend auto
```

## Running GLM-4.5-Air with FP8 or BF16

There are two ways to parallelize the model over multiple GPUs: (1) Tensor-parallel or (2) Data-parallel. Each one has its own advantages, where tensor-parallel is usually more beneficial for low-latency / low-load scenarios and data-parallel works better for cases where there is a lot of data with heavy-loads.

run tensor-parallel like this:

```bash

# Start server with FP8 model on 4 GPUs. the model can also changed to BF16 as zai-org/GLM-4.5-Air
vllm serve zai-org/GLM-4.5-Air-FP8 \
     --tensor-parallel-size 8 \
     --tool-call-parser glm45 \
     --reasoning-parser glm45 \
     --enable-auto-tool-choice
```

* You can set `--max-model-len` to preserve memory. `--max-model-len=65536` is usually good for most scenarios and max is 128k.
* You can set `--max-num-batched-tokens` to balance throughput and latency, higher means higher throughput but higher latency. `--max-num-batched-tokens=32768` is usually good for prompt-heavy workloads. But you can reduce it to 16k and 8k to reduce activation memory usage and decrease latency.
* vLLM conservatively use 90% of GPU memory, you can set `--gpu-memory-utilization=0.95` to maximize KVCache.
* Make sure to follow the command-line instructions to ensure the tool-calling functionality is properly enabled.

## Benchmarking

For benchmarking, disable prefix caching by adding `--no-enable-prefix-caching` to the server command.

### FP8 Benchmark

```bash
# Prompt-heavy benchmark (8k/1k)
vllm bench serve \
  --model zai-org/GLM-4.5-FP8 \
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
Successful requests:                     16        
Request rate configured (RPS):           10000.00  
Benchmark duration (s):                  24.56     
Total input tokens:                      128000    
Total generated tokens:                  16000     
Request throughput (req/s):              0.65      
Output token throughput (tok/s):         651.58    
Total Token throughput (tok/s):          5864.22   
---------------Time to First Token----------------
Mean TTFT (ms):                          2100.95   
Median TTFT (ms):                        2063.87   
P99 TTFT (ms):                           4284.97   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          22.35     
Median TPOT (ms):                        22.39     
P99 TPOT (ms):                           24.19     
---------------Inter-token Latency----------------
Mean ITL (ms):                           22.35     
Median ITL (ms):                         20.23     
P99 ITL (ms):                            37.17     
==================================================
```
