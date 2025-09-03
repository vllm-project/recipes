# Ernie4.5 Text Model Usage Guide

This guide describes how to run [ERNIE-4.5-21B-A3B-PT](https://huggingface.co/baidu/ERNIE-4.5-21B-A3B-PT) and [ERNIE-4.5-300B-A47B-PT](https://huggingface.co/baidu/ERNIE-4.5-300B-A47B-PT) with native BF16. 

## Installing vLLM
Note: transformers >= 4.54.0 and vllm >= 0.10.1

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install vllm --torch-backend=auto
```

## Running Ernie4.5

```bash
# 21B model 80G*1 GPU
vllm serve baidu/ERNIE-4.5-21B-A3B-PT
```

```bash
# 300B model 80G*8 GPU with vllm FP8 online quantification
vllm serve baidu/ERNIE-4.5-300B-A47B-PT \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization=0.95 \
  --quantization fp8
```

If your single node GPU memory is insufficient, native BF16 deployment may require multi nodes, multi node deployment reference [vLLM doc](https://docs.vllm.ai/en/latest/serving/parallelism_scaling.html?h=#multi-node-deployment) to start ray cluster. Then run vllm on the master node
```bash
# 300B model 80G*16 GPU with native BF16
vllm serve baidu/ERNIE-4.5-300B-A47B-PT \
  --tensor-parallel-size 16
```

## Running Ernie4.5 MTP

```bash
# 21B MTP model 80G*1 GPU
vllm serve baidu/ERNIE-4.5-21B-A3B-PT \
  --speculative-config '{"method": "ernie_mtp","model": "baidu/ERNIE-4.5-21B-A3B-PT","num_speculative_tokens": 1}'
```

```bash
# 300B MTP model 80G*8 GPU with vllm FP8 online quantification
vllm serve baidu/ERNIE-4.5-300B-A47B-PT \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization=0.95 \
  --quantization fp8 \
  --speculative-config '{"method": "ernie_mtp","model": "baidu/ERNIE-4.5-300B-A47B-PT","num_speculative_tokens": 1}'
```

If your single node GPU memory is insufficient, native BF16 deployment may require multi nodes, multi node deployment reference [vLLM doc](https://docs.vllm.ai/en/latest/serving/parallelism_scaling.html?#multi-node-deployment) to start ray cluster. Then run vllm on the master node
```bash
# 300B MTP model 80G*16 GPU with native BF16
vllm serve baidu/ERNIE-4.5-300B-A47B-PT \
  --tensor-parallel-size 16 \
  --speculative-config '{"method": "ernie_mtp","model": "baidu/ERNIE-4.5-300B-A47B-PT","num_speculative_tokens": 1}'
```


## Benchmarking

For benchmarking, only the first `vllm bench serve` after service startup to ensure it is not affected by prefix cache


```bash
# Prompt-heavy benchmark (8k/1k)
vllm bench serve \
  --model baidu/ERNIE-4.5-21B-A3B-PT \
  --host 127.0.0.1 \
  --port 8200 \
  --dataset-name random \
  --random-input-len 8000 \
  --random-output-len 1000 \
  --request-rate 10 \
  --num-prompts 16 \
  --ignore-eos
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
Benchmark duration (s):                  18.65     
Total input tokens:                      127952    
Total generated tokens:                  16000     
Request throughput (req/s):              0.86      
Output token throughput (tok/s):         857.78    
Total Token throughput (tok/s):          7717.46   
---------------Time to First Token----------------
Mean TTFT (ms):                          876.28    
Median TTFT (ms):                        910.42    
P99 TTFT (ms):                           1596.48   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          16.84     
Median TPOT (ms):                        16.86     
P99 TPOT (ms):                           18.11     
---------------Inter-token Latency----------------
Mean ITL (ms):                           16.84     
Median ITL (ms):                         15.49     
P99 ITL (ms):                            20.69     
==================================================
```

