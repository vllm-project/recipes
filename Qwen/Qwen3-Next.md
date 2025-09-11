# Qwen3-Next Usage Guide

[Qwen3-Next](https://github.com/QwenLM/Qwen3-Coder) is an advanced large language model created by the Qwen team from Alibaba Cloud. vLLM already supports Qwen3-Next. You can install vLLM with `Qwen3-Next` support using the following method:

## Installing vLLM

```bash
conda create -n myenv python=3.12 -y
conda activate myenv
export VLLM_COMMIT=xxx # Use full commit hash from the main branch
pip install https://wheels.vllm.ai/${VLLM_COMMIT}/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl
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

`Qwen3-Next` also support MTP (Multi-Token Prediction), you can using the following script to enable MTP

```bash
vllm serve Qwen/Qwen3-Next-80B-A3B-Instruct  \
--tokenizer-mode auto  --gpu-memory-utilization 0.8 \
--speculative-config '{"method": "qwen3_next_mtp", "num_speculative_tokens": 2}' \
--tensor-parallel-size 4 --no-enable-chunked-prefill 
```

The `speculative-config` argument configures speculative decoding settings using a JSON format. The method "qwen3_next_mtp" specifies that the system should use Qwen3-Next's specialized multi-token prediction method. The `num_speculative_tokens` setting of 2 means the model will speculate 2 tokens ahead during generation.


## Performance Metrics

### Benchmarking

We use the following script to demonstrate how to benchmark `Qwen/Qwen3-Next-80B-A3B-Instruct`.

```bash
vllm bench serve \
  --backend vllm \
  --model qwen3-next \
  --endpoint /v1/completions \
  --dataset-name random \
  --random-input 2048 \
  --random-output 1024 \
  --max-concurrency 10 \
  --num-prompt 100 \
```
If successful, you will see output similar to the following output.

```shell
============ Serving Benchmark Result ============
Successful requests:                     100       
Maximum request concurrency:             10        
Benchmark duration (s):                  109.58    
Total input tokens:                      204357    
Total generated tokens:                  92788     
Request throughput (req/s):              0.91      
Output token throughput (tok/s):         846.72    
Total Token throughput (tok/s):          2711.55   
---------------Time to First Token----------------
Mean TTFT (ms):                          297.39    
Median TTFT (ms):                        217.65    
P99 TTFT (ms):                           752.94    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          11.23     
Median TPOT (ms):                        11.25     
P99 TPOT (ms):                           13.76     
---------------Inter-token Latency----------------
Mean ITL (ms):                           11.18     
Median ITL (ms):                         10.59     
P99 ITL (ms):                            12.38     
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

### Known limitations

- Qwen3-Next currently does not support prefix-cache