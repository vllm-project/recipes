# Ernie4.5 VL Model Usage Guide

This guide describes how to run [ERNIE-4.5-VL-28B-A3B-PT](https://huggingface.co/baidu/ERNIE-4.5-VL-28B-A3B-PT) and [ERNIE-4.5-VL-424B-A47B-PT](https://huggingface.co/baidu/ERNIE-4.5-VL-424B-A47B-PT) with native BF16. 


## Installing vLLM
Ernie4.5-VL support was recently added to vLLM main branch and is not yet available in any official release:
```bash
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install git+https://github.com/vllm-project/vllm.git
```

## Running Ernie4.5-VL

NOTE: torch.compile and cuda graph are not supported due to the heterogeneous expert architecture. (vision and text experts)
```bash
# 28B model 80G*1 GPU
vllm serve baidu/ERNIE-4.5-VL-28B-A3B-PT \
  --trust-remote-code
```
```bash
# 424B model 140G*8 GPU with native BF16
vllm serve baidu/ERNIE-4.5-VL-424B-A47B-PT \
  --trust-remote-code \
  --tensor-parallel-size 8
```

If you only want to test the functionality and only have 8×80G GPU, you can use the `--cpu-offload-gb` parameter to offload part of the weights to CPU memory, and additionally use FP8 online quantization to further reduce GPU memory.

```bash
# 424B model 80G*8 GPU with FP8 quantization and CPU offloading
vllm serve baidu/ERNIE-4.5-VL-424B-A47B-PT \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --quantization fp8 \
  --cpu-offload-gb 50
```


If your single node GPU memory is insufficient, native BF16 deployment may require multi nodes, multi node deployment reference [vLLM doc](https://docs.vllm.ai/en/latest/serving/parallelism_scaling.html?#multi-node-deployment) to start ray cluster. Then run vllm on the master node
```bash
# 424B model 80G*16 GPU with native BF16
vllm serve baidu/ERNIE-4.5-VL-424B-A47B-PT \
  --trust-remote-code \
  --tensor-parallel-size 16
```

## Benchmarking

For benchmarking, only the first `vllm bench serve` after service startup to ensure it is not affected by prefix cache

```bash
# Prompt-heavy benchmark (8k/1k)
vllm bench serve \
  --model baidu/ERNIE-4.5-VL-28B-A3B-PT \
  --host 127.0.0.1 \
  --port 8200 \
  --dataset-name random \
  --random-input-len 8000 \
  --random-output-len 1000 \
  --request-rate 10 \
  --num-prompts 16 \
  --ignore-eos \
  --trust-remote-code
```

### Benchmark Configurations

Test different workloads by adjusting input/output lengths:

- **Prompt-heavy**: 8000 input / 1000 output
- **Decode-heavy**: 1000 input / 8000 output
- **Balanced**: 1000 input / 1000 output

Test different batch sizes by changing `--num-prompts`, e.g., 1, 16, 32, 64, 128, 256, 512

### Expected Output

```shell
============ Serving Benchmark Result ============
Successful requests:                     16        
Request rate configured (RPS):           10.00     
Benchmark duration (s):                  61.27     
Total input tokens:                      128000    
Total generated tokens:                  16000     
Request throughput (req/s):              0.26      
Output token throughput (tok/s):         261.13    
Total Token throughput (tok/s):          2350.19   
---------------Time to First Token----------------
Mean TTFT (ms):                          15663.63  
Median TTFT (ms):                        21148.69  
P99 TTFT (ms):                           22147.85  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          38.74     
Median TPOT (ms):                        38.90     
P99 TPOT (ms):                           39.92     
---------------Inter-token Latency----------------
Mean ITL (ms):                           43.27     
Median ITL (ms):                         36.35     
P99 ITL (ms):                            236.49    
==================================================
```
