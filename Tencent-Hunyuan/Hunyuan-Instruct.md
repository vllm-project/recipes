### AMD GPU Support 

Please follow the steps here to install and run Hunyuan-A13B-Instruct models on AMD MI300X/MI325X/MI355X

### Step 1: Install vLLM
> Note: The vLLM wheel for ROCm requires Python 3.12, ROCm 7.0, and glibc >= 2.35. If your environment does not meet these requirements, please use the Docker-based setup as described in the [documentation](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/#pre-built-images). 
```bash
uv venv
source .venv/bin/activate
uv pip install vllm --extra-index-url https://wheels.vllm.ai/rocm
```

### Step 2: Log in to Hugging Face
Huggingface login
```shell
huggingface-cli login
```

### Step 3: Start the vLLM server

Run the vllm online serving:

```shell
export SAFETENSORS_FAST_GPU=1
export VLLM_USE_TRITON_FLASH_ATTN=0 
export VLLM_ROCM_USE_AITER=1
vllm serve tencent/Hunyuan-A13B-Instruct --tensor-parallel-size 2 --gpu-memory-utilization 0.9 --disable-log-requests --no-enable-prefix-caching --trust-remote-code 
```

### Step 4: Run Benchmark
Open a new terminal and run the following command to execute the benchmark script:
```shell
vllm bench serve \
  --model "tencent/Hunyuan-A13B-Instruct" \
  --dataset-name random \
  --random-input-len 8000 \
  --random-output-len 1000 \
  --request-rate 10000 \
  --num-prompts 16 \
  --ignore-eos
```
