# Qwen2.5-VL Usage Guide

This guide describes how to run Qwen2.5-VL series with native BF16 on NVIDIA GPUs. 
Since BF16 is the commonly used precision type for Qwen2.5-VL training, using BF16 in inference ensures the best accuracy.


## Installing vLLM

```bash
uv venv
source .venv/bin/activate
uv pip install -U vllm --torch-backend auto
```

## Running Qwen2.5-VL-72B with BF16 on 4xA100

There are two ways to parallelize the model over multiple GPUs: (1) Tensor-parallel (TP) or (2) Data-parallel (DP). Each one has its own advantages, where tensor-parallel is usually more beneficial for low-latency / low-load scenarios and data-parallel works better for cases where there is a lot of data with heavy loads.

To launch the online inference server using TP=4:

```bash
# Start server with BF16 model on 4 GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3
vllm serve Qwen/Qwen2.5-VL-72B-Instruct  \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 4 \
  --data-parallel-size 1 \
```
* You can set `--max-model-len` to preserve memory. `--max-model-len=65536` is usually good for most scenarios.
* You can set `--tensor-parallel-size` and `--data-parallel-size` to adjust the parallel strategy. But TP should be larger than 2 for A100-80GB devices to avoid OOM.
* vLLM conservatively uses 90% of GPU memory. You can set `--gpu-memory-utilization=0.95` to maximize KVCache.


To run a smaller model, such as Qwen2.5-VL-7B, you can simply replace the model name `Qwen/Qwen2.5-VL-72B-Instruct` with `Qwen/Qwen2.5-VL-7B-Instruct`. 


## Benchmarking

For benchmarking, you first need to launch the server with prefix caching disabled by adding `--no-enable-prefix-caching` to the server command.

### Qwen2.5VL-72B Benchmark

Once the server is running, open another terminal and run the benchmark client:

```bash
vllm bench serve \
  --host 0.0.0.0 \
  --port 8000 \
  --model Qwen/Qwen2.5-VL-72B-Instruct \
  --dataset-name random \
  --random-input-len 8000 \
  --random-output-len 1000 \
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

For Qwen2.5VL-72B:

```shell
============ Serving Benchmark Result ============
Successful requests:                     16
Benchmark duration (s):                  68.85
Total input tokens:                      127945
Total generated tokens:                  16000
Request throughput (req/s):              0.23
Output token throughput (tok/s):         232.40
Total Token throughput (tok/s):          2090.83
---------------Time to First Token----------------
Mean TTFT (ms):                          13912.67
Median TTFT (ms):                        13939.98
P99 TTFT (ms):                           26067.96
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          53.91
Median TPOT (ms):                        53.93
P99 TPOT (ms):                           64.82
---------------Inter-token Latency----------------
Mean ITL (ms):                           53.91
Median ITL (ms):                         42.84
P99 ITL (ms):                            431.25
==================================================
```


For Qwen2.5VL-7B (all configurations remain the same except for model name):


```shell
============ Serving Benchmark Result ============
Successful requests:                     16
Benchmark duration (s):                  11.57
Total input tokens:                      127945
Total generated tokens:                  16000
Request throughput (req/s):              1.38
Output token throughput (tok/s):         1383.29
Total Token throughput (tok/s):          12444.88
---------------Time to First Token----------------
Mean TTFT (ms):                          1979.73
Median TTFT (ms):                        1981.13
P99 TTFT (ms):                           3691.55
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          9.39
Median TPOT (ms):                        9.40
P99 TPOT (ms):                           10.88
---------------Inter-token Latency----------------
Mean ITL (ms):                           9.39
Median ITL (ms):                         7.76
P99 ITL (ms):                            60.49
==================================================
```

