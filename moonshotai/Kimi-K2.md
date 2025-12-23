# moonshotai/Kimi-K2 Usage Guide

This guide describes how to run Kimi-K2 with native FP8. 

---
**Note:** This guide is partially referenced and adapted from the official [Kimi-K2-Instruct Deployment Guidance](https://huggingface.co/moonshotai/Kimi-K2-Instruct/blob/main/docs/deploy_guidance.md) provided by Moonshot AI. We would like to express our gratitude to the original authors.
---

## Installing vLLM

```bash
uv venv
source .venv/bin/activate
uv pip install -U vllm --torch-backend auto
```

## Running Kimi-K2 with FP8 on 16xH800

The smallest deployment unit for Kimi-K2 FP8 weights with 128k seqlen on mainstream H800 platform is a cluster with 16 GPUs with either Tensor Parallel (TP) or "data parallel + expert parallel" (DP+EP).
Running parameters for this environment are provided below. You may scale up to more nodes and increase expert-parallelism to enlarge the inference batch size and overall throughput.

### Tensor Parallelism + pipeline-parallelism
A sample launch command is:

```bash

# start ray on node 0 and node 1

# node 0:
vllm serve moonshotai/Kimi-K2-Instruct --trust-remote-code --tokenizer-mode auto --tensor-parallel-size 8 --pipeline-parallel-size 2 --dtype bfloat16 --quantization fp8 --max-model-len 2048 --max-num-seqs 1 --max-num-batched-tokens 1024 --enable-chunked-prefill --disable-log-requests --kv-cache-dtype fp8 -dcp 8
```

Key parameter notes:

* enable-auto-tool-choice: Required when enabling tool usage.
* tool-call-parser kimi_k2: Required when enabling tool usage.

### Data Parallelism + Expert Parallelism
You can install libraries like DeepEP and DeepGEMM as needed. Then run the command (example on H800):

```bash
# node 0
vllm serve moonshotai/Kimi-K2-Instruct --port 8000 --served-model-name kimi-k2 --trust-remote-code --data-parallel-size 16 --data-parallel-size-local 8 --data-parallel-address $MASTER_IP --data-parallel-rpc-port $PORT --enable-expert-parallel --max-num-batched-tokens 8192 --max-num-seqs 256 --gpu-memory-utilization 0.85 --enable-auto-tool-choice --tool-call-parser kimi_k2

# node 1
vllm serve moonshotai/Kimi-K2-Instruct --headless --data-parallel-start-rank 8 --port 8000 --served-model-name kimi-k2 --trust-remote-code --data-parallel-size 16 --data-parallel-size-local 8 --data-parallel-address $MASTER_IP --data-parallel-rpc-port $PORT --enable-expert-parallel --max-num-batched-tokens 8192 --max-num-seqs 256 --gpu-memory-utilization 0.85 --enable-auto-tool-choice --tool-call-parser kimi_k2
```

Additional flags:

* You can set `--max-model-len` to preserve memory. `--max-model-len=65536` is usually good for most scenarios.
* You can set `--max-num-batched-tokens` to balance throughput and latency, higher means higher throughput but higher latency. `--max-num-batched-tokens=32768` is usually good for prompt-heavy workloads. But you can reduce it to 16k and 8k to reduce activation memory usage and decrease latency.
* vLLM conservatively uses 90% of GPU memory, you can set `--gpu-memory-utilization=0.95` to maximize KVCache.


## Benchmarking

### FP8 Benchmark on 16xH800

```bash
vllm bench serve \
  --model moonshotai/Kimi-K2-Instruct \
  --dataset-name random \
  --random-input-len 1000 \
  --random-output-len 512 \
  --request-rate 1.0 \
  --num-prompts 8 \
  --ignore-eos \
  --trust-remote-code
```


### Benchmark Configurations


Test different batch sizes by changing `--num-prompts`:

- Batch sizes: 1, 16, 32, 64, 128, 256, 512

### Expected Output

```shell
============ Serving Benchmark Result ============
Successful requests:                     8         
Request rate configured (RPS):           1.00      
Benchmark duration (s):                  132.79    
Total input tokens:                      8000      
Total generated tokens:                  4096      
Request throughput (req/s):              0.06      
Output token throughput (tok/s):         30.84     
Total Token throughput (tok/s):          91.09     
---------------Time to First Token----------------
Mean TTFT (ms):                          58282.92  
Median TTFT (ms):                        57827.30  
P99 TTFT (ms):                           110831.45 
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          30.78     
Median TPOT (ms):                        31.49     
P99 TPOT (ms):                           33.76     
---------------Inter-token Latency----------------
Mean ITL (ms):                           30.78     
Median ITL (ms):                         22.37     
P99 ITL (ms):                            322.81    
==================================================
```

### FP8 Benchmark on 16xH200

```bash
vllm bench serve \
  --model moonshotai/Kimi-K2-Instruct \
  --dataset-name random \
  --random-input-len 8000 \
  --random-output-len 1000 \
  --request-rate 10000 \
  --num-prompts 16 \
  --ignore-eos \
  --trust-remote-code
```

### Expected Output

```shell
============ Serving Benchmark Result ============
Successful requests:                     16        
Benchmark duration (s):                  62.75     
Total input tokens:                      128000    
Total generated tokens:                  16000     
Request throughput (req/s):              0.25      
Output token throughput (tok/s):         254.99    
Total Token throughput (tok/s):          2294.88   
---------------Time to First Token----------------
Mean TTFT (ms):                          4278.46   
Median TTFT (ms):                        4285.54   
P99 TTFT (ms):                           7685.31   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          58.15     
Median TPOT (ms):                        58.16     
P99 TPOT (ms):                           61.35     
---------------Inter-token Latency----------------
Mean ITL (ms):                           58.15     
Median ITL (ms):                         54.59     
P99 ITL (ms):                            91.18     
==================================================
```

After adding '-dcp 8':
```bash
============ Serving Benchmark Result ============
Successful requests:                     16        
Request rate configured (RPS):           10000.00  
Benchmark duration (s):                  47.14     
Total input tokens:                      128000    
Total generated tokens:                  16000     
Request throughput (req/s):              0.34      
Output token throughput (tok/s):         339.38    
Peak output token throughput (tok/s):    384.00    
Peak concurrent requests:                16.00     
Total Token throughput (tok/s):          3054.46   
---------------Time to First Token----------------
Mean TTFT (ms):                          2007.87   
Median TTFT (ms):                        1932.03   
P99 TTFT (ms):                           4680.76   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          45.01     
Median TPOT (ms):                        45.10     
P99 TPOT (ms):                           46.51     
---------------Inter-token Latency----------------
Mean ITL (ms):                           45.01     
Median ITL (ms):                         42.01     
P99 ITL (ms):                            52.01     
==================================================
```


## AMD GPU Support 

Please follow the steps here to install and run kimi-K2 models on AMD MI300X GPU.
### Step 1: Prepare Docker Environment
Pull the latest vllm docker:
```shell
docker pull rocm/vllm-dev:nightly
```
Launch the ROCm vLLM docker: 
```shell
docker run -it --ipc=host --network=host --privileged --cap-add=CAP_SYS_ADMIN --device=/dev/kfd --device=/dev/dri --device=/dev/mem --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v $(pwd):/work -e SHELL=/bin/bash  --name kimi-K2 rocm/vllm-dev:nightly 
```
### Step 2: Log in to Hugging Face
Log in to your Hugging Face account:
```shell
huggingface-cli login
```

### Step 3: Start the vLLM server

Run the vllm online serving with this sample command:
```shell
SAFETENSORS_FAST_GPU=1 \
VLLM_USE_V1=1 \
VLLM_USE_TRITON_FLASH_ATTN=0 \
vllm serve moonshotai/Kimi-K2-Instruct \
  --tensor-parallel-size 8 \
  --no-enable-prefix-caching \
  --trust-remote-code


### Step 4: Run Benchmark
Open a new terminal and run the following command to execute the benchmark script inside the container.
```shell
docker exec -it kimi-K2 vllm bench serve \
  --model "moonshotai/Kimi-K2-Instruct" \
  --dataset-name random \
  --random-input-len 8192 \
  --random-output-len 1024 \
  --request-rate 10000 \
  --num-prompts 16 \
  --ignore-eos \
  --trust-remote-code 
```


