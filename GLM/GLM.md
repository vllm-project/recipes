# GLM-4.X LLM Usage Guide

GLM-4.X LLM include those model below:

+ GLM-4.7-Flash
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

```bash
# install the nightly build of vLLM for GLM-4.7
uv pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly

# install transformers from source
uv pip install git+https://github.com/huggingface/transformers.git
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

## Speculative Decoding with MTP

GLM-4.X models include built-in Multi-Token Prediction (MTP) layers that can be used for speculative decoding to accelerate generation throughput.

### Enabling MTP

To enable MTP speculative decoding, add the `--speculative-config` flags to your server command:

```bash

# Start server with FP8 model on 4xH200
vllm serve zai-org/GLM-4.7-FP8 \
     --tensor-parallel-size 4 \
     --speculative-config.method mtp \
     --speculative-config.num_speculative_tokens 1 \
     --tool-call-parser glm47 \
     --reasoning-parser glm45 \
     --enable-auto-tool-choice
```

### Recommended Settings

We recommend using `--speculative-config.num_speculative_tokens 1` for optimal performance. While higher values like 3 increase the mean acceptance length, the acceptance rate drops significantly, resulting in lower overall throughput.

### Performance Considerations

* MTP works best with high acceptance rates. With 1 speculative token, acceptance rates typically exceed 90%.
* MTP adds memory overhead for the draft model computations. Monitor GPU memory usage when enabling.
* The speedup is most noticeable for decode-heavy workloads where generation time dominates.
* MTP is compatible with other optimizations like prefix caching.

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


## AMD GPU Support

Please follow the steps here to install and run GLM models on AMD MI300X, MI325X, MI355X GPUs.

### Step 1: Prepare Environment
#### Option 1: Installation from pre-built wheels (For AMD ROCm: MI300x/MI325x/MI355x)
We recommend using the official package for AMD GPUs (MI300x/MI325x/MI355x). 
```bash
uv pip install vllm --extra-index-url https://wheels.vllm.ai/rocm
```
⚠️ The vLLM wheel for ROCm is compatible with Python 3.12, ROCm 7.0, and glibc >= 2.35. If your environment is incompatible, please use docker flow in [vLLM](https://vllm.ai/).

#### Option 2: Docker image
Pull the latest vllm docker:

```shell
docker pull vllm/vllm-openai-rocm:v0.15.1
```

Launch the ROCm vLLM docker: 
```shell
docker run -it --ipc=host --network=host --privileged --cap-add=CAP_SYS_ADMIN --device=/dev/kfd --device=/dev/dri --device=/dev/mem --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v $(pwd):/work -e SHELL=/bin/bash --name GLM vllm/vllm-openai-rocm:v0.15.1 
```

### Step 2: Log in to Hugging Face
Hugging Face login:

```shell
huggingface-cli login
```

### Step 3: Serving GLM Models
For the GLM model series, use `$model_name` as the parameter in the serving and benchmark commands. Valid values for `$model_name` are:
- zai-org/GLM-4.5
- zai-org/GLM-4.5-Air
- zai-org/GLM-4.6
- zai-org/GLM-4.7
- zai-org/GLM-4.7-Flash

Run the vllm online serving using a sample command as below:

```shell
SAFETENSORS_FAST_GPU=1 \
vllm serve $model_name \
    --tensor-parallel-size 8 \
    --gpu-memory-utilization 0.9 \
    --disable-log-requests \
    --no-enable-prefix-caching \
    --trust-remote-code
```


### Step 4: Run Benchmark
Open a new terminal and run the following command to execute the benchmark script inside the container.
```shell
docker exec -it GLM vllm bench serve \
  --model $model_name \
  --dataset-name random \
  --random-input-len 8000 \
  --random-output-len 1000 \
  --request-rate 10000 \
  --num-prompts 16 \
  --ignore-eos
```
