# Qwen3-Next Usage Guide

[Qwen3-Next](https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd&from=research.latest-advancements-list) is an advanced large language model created by the Qwen team from Alibaba Cloud. It features several key improvements:

* A hybrid attention mechanism
* A highly sparse Mixture-of-Experts (MoE) structure
* Training-stability-friendly optimizations
* A multi-token prediction mechanism for faster inference

## Installing vLLM

```bash
uv venv
source .venv/bin/activate
uv pip install vllm --extra-index-url https://wheels.vllm.ai/nightly
```

## Launching Qwen3-Next with vLLM

You can use 4x H200/H20 or 4x A100/A800 GPUs to launch this model.

### Basic Multi-GPU Setup

```bash
vllm serve Qwen/Qwen3-Next-80B-A3B-Instruct \
  --tensor-parallel-size 4 \
  --served-model-name qwen3-next 

```
### Advanced Configuration with MTP

`Qwen3-Next` also supports Multi-Token Prediction (MTP in short), you can launch the model server with the following arguments to enable MTP.

```bash
vllm serve Qwen/Qwen3-Next-80B-A3B-Instruct  \
--tokenizer-mode auto  --gpu-memory-utilization 0.8 \
--speculative-config '{"method": "qwen3_next_mtp", "num_speculative_tokens": 2}' \
--tensor-parallel-size 4 --no-enable-chunked-prefill 
```

The `speculative-config` argument configures speculative decoding settings using a JSON format. The method "qwen3_next_mtp" specifies that the system should use Qwen3-Next's specialized multi-token prediction method. The `"num_speculative_tokens": 2` setting means the model will speculate 2 tokens ahead during generation.


## Performance Metrics

### Benchmarking

We use the following script to demonstrate how to benchmark `Qwen/Qwen3-Next-80B-A3B-Instruct`.

```bash
vllm bench serve \
  --backend vllm \
  --model Qwen/Qwen3-Next-80B-A3B-Instruct \
  --served-model-name qwen3-next \
  --endpoint /v1/completions \
  --dataset-name random \
  --random-input 2048 \
  --random-output 1024 \
  --max-concurrency 256
```

#### B200 Outputs

Server command:
```
vllm serve Qwen/Qwen3-Next-80B-A3B-Instruct --tensor-parallel-size 4 --served-model-name qwen3-next
```

Outputs
```
============ Serving Benchmark Result ============
Successful requests:                     1000      
Maximum request concurrency:             256       
Benchmark duration (s):                  117.94    
Total input tokens:                      2043736   
Total generated tokens:                  957462    
Request throughput (req/s):              8.48      
Output token throughput (tok/s):         8118.18   
Total Token throughput (tok/s):          25446.73  
---------------Time to First Token----------------
Mean TTFT (ms):                          1387.84   
Median TTFT (ms):                        419.05    
P99 TTFT (ms):                           8148.70   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          29.40     
Median TPOT (ms):                        30.14     
P99 TPOT (ms):                           45.72     
---------------Inter-token Latency----------------
Mean ITL (ms):                           28.49     
Median ITL (ms):                         20.46     
P99 ITL (ms):                            142.60    
==================================================
```

#### B200 MTP Outputs

Server command:
```
vllm serve Qwen/Qwen3-Next-80B-A3B-Instruct --tensor-parallel-size 4 --served-model-name qwen3-next --tokenizer-mode auto --speculative-config {"method": "qwen3_next_mtp", "num_speculative_tokens": 2} --no-enable-chunked-prefill
```

Outputs
```
============ Serving Benchmark Result ============
Successful requests:                     1000      
Maximum request concurrency:             256       
Benchmark duration (s):                  161.36    
Total input tokens:                      2043736   
Total generated tokens:                  952306    
Request throughput (req/s):              6.20      
Output token throughput (tok/s):         5901.85   
Total Token throughput (tok/s):          18567.77  
---------------Time to First Token----------------
Mean TTFT (ms):                          3963.48   
Median TTFT (ms):                        515.20    
P99 TTFT (ms):                           25537.02  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          39.03     
Median TPOT (ms):                        34.60     
P99 TPOT (ms):                           98.63     
---------------Inter-token Latency----------------
Mean ITL (ms):                           106.27    
Median ITL (ms):                         68.55     
P99 ITL (ms):                            392.13    
==================================================
```

## Usage Tips

### Tune MoE kernel

When starting the model service, you may encounter the following warning in the server log(Suppose the GPU is `NVIDIA_H20-3e`):

```shell
(VllmWorker TP2 pid=47571) WARNING 09-09 15:47:25 [fused_moe.py:727] Using default MoE config. Performance might be sub-optimal! Config file not found at ['/vllm_path/vllm/model_executor/layers/fused_moe/configs/E=512,N=128,device_name=NVIDIA_H20-3e.json']
```

You can use [benchmark_moe](https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_moe.py) to perform MoE Triton kernel tuning for your hardware. Once tuning is complete, a JSON file with a name like `E=512,N=128,device_name=NVIDIA_H20-3e.json` will be generated. You can specify the directory containing this file for your deployment hardware using the environment variable `VLLM_TUNED_CONFIG_FOLDER`, like:

```shell
VLLM_TUNED_CONFIG_FOLDER=your_moe_tuned_dir vllm serve Qwen/Qwen3-Next-80B-A3B-Instruct \
  --tensor-parallel-size 4 \
  --served-model-name qwen3-next 

```

You should see the following information printed in the server log. This indicates that the tuned MoE configuration has been loaded, which will improve the model service performance.

```shell
(VllmWorker TP2 pid=60498) INFO 09-09 16:23:07 [fused_moe.py:720] Using configuration from /your_moe_tuned_dir/E=512,N=128,device_name=NVIDIA_H20-3e.json for MoE layer.
```

### Data Parallel Deployment

vLLM supports multi-parallel groups. You can refer to [Data Parallel Deployment documentation](https://docs.vllm.ai/en/latest/serving/data_parallel_deployment.html) and try parallel combinations that are more suitable for this model.

### Function calling

vLLM also supports calling user-defined functions. Make sure to run your Qwen3-Next models with the following arguments.

```bash
vllm serve ... --tool-call-parser hermes --enable-auto-tool-choice
```

### Known limitations

- Qwen3-Next currently does not support automatic prefix caching.
