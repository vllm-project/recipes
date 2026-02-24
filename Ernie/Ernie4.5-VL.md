# Ernie4.5 VL Model Usage Guide

This guide describes how to run [ERNIE-4.5-VL-28B-A3B-PT](https://huggingface.co/baidu/ERNIE-4.5-VL-28B-A3B-PT) and [ERNIE-4.5-VL-424B-A47B-PT](https://huggingface.co/baidu/ERNIE-4.5-VL-424B-A47B-PT) with native BF16. 


## Installing vLLM
ERNIE-4.5-VL support was recently added to vLLM main branch and is not yet available in any official release:
```bash
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install -U vllm --torch-backend auto
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


## AMD GPU Support

Please follow the steps here to install and run ERNIE-4.5-VL model on AMD MI300X, MI325X, MI355X GPUs.

### Step 1: Prepare Environment
#### Option 1: Installation from pre-built wheels (For AMD ROCm: MI300x/MI325x/MI355x)
We recommend using the official package for AMD GPUs (MI300x/MI325x/MI355x). 
```bash
uv pip install vllm --extra-index-url https://wheels.vllm.ai/rocm
```
⚠️ The vLLM wheel for ROCm is compatible with Python 3.12, ROCm 7.0, and glibc >= 2.35. If your environment is incompatible, please use docker flow in [vLLM](https://vllm.ai/).

#### Option 2: Docker image
Pull the latest vllm docker:

```bash
docker pull vllm/vllm-openai-rocm:latest
```

Launch the ROCm vLLM docker: 

```bash
docker run -it \
  --ipc=host \
  --network=host \
  --privileged \
  --cap-add=CAP_SYS_ADMIN \
  --device=/dev/kfd \
  --device=/dev/dri \
  --device=/dev/mem \
  --group-add video \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -v $(pwd):/work \
  -e SHELL=/bin/bash \
  --name Ernie-4.5-VL \
  vllm/vllm-openai-rocm:latest
```

After running the command above, you are already inside the container. Proceed to Step 2 in that shell. If you detached from the container or started it in detached mode, attach to the container with:

```bash
docker attach Ernie-4.5-VL
```

### Step 2: Log in to Hugging Face
Hugging Face login:

```bash
huggingface-cli login
```

### Step 3: Start the vLLM server

Run the vllm online serving
Sample Command
```bash
VLLM_ROCM_USE_AITER=1 \
SAFETENSORS_FAST_GPU=1 \
vllm serve baidu/ERNIE-4.5-VL-28B-A3B-PT \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.9 \
    --disable-log-requests \
    --no-enable-prefix-caching \
    --trust-remote-code
```


### Step 4: Run Benchmark
Open a new terminal and run the following command to execute the benchmark script:

```bash
vllm bench serve \
  --model baidu/ERNIE-4.5-VL-28B-A3B-PT \
  --dataset-name random \
  --random-input-len 8000 \
  --random-output-len 1000 \
  --request-rate 10000 \
  --num-prompts 16 \
  --trust-remote-code \
  --ignore-eos
```

If you are using a Docker environment, open a new terminal and run the benchmark inside the container with:

```bash
docker exec -it Ernie-4.5-VL vllm bench serve \
  --model baidu/ERNIE-4.5-VL-28B-A3B-PT \
  --dataset-name random \
  --random-input-len 8000 \
  --random-output-len 1000 \
  --request-rate 10000 \
  --num-prompts 16 \
  --trust-remote-code \
  --ignore-eos
```