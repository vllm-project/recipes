# Kimi-Linear Usage Guide

This guide describes how to run moonshotai/Kimi-Linear-48B-A3B-Instruct.

## Installing vLLM

```bash
uv venv
source .venv/bin/activate
uv pip install -U vllm --torch-backend auto
```

## Running Kimi-Linear

It's easy to run Kimi-Linear.
The following snippets assume you have 4 or 8 GPUs on a single node.

### 4-GPU tensor parallel
```bash
vllm serve moonshotai/Kimi-Linear-48B-A3B-Instruct \
  --port 8000 \
  --tensor-parallel-size 4 \
  --max-model-len 1048576 \
  --trust-remote-code
```

### 8-GPU tensor parallel
```bash
vllm serve moonshotai/Kimi-Linear-48B-A3B-Instruct \
  --port 8000 \
  --tensor-parallel-size 8 \
  --max-model-len 1048576 \
  --trust-remote-code
```

> If you see OOM, reduce `--max-model-len` (e.g. 65536) or increase `--gpu-memory-utilization` (≤ 0.95).

Once the server is up, test it with:
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"moonshotai/Kimi-Linear-48B-A3B-Instruct","messages":[{"role":"user","content":"Hello!"}]}'
```

## AMD GPU Support

Please follow the steps here to install and run Kimi-Linear-48B-A3B-Instruct models on AMD MI300X GPU.
### Step 1: Prepare Docker Environment
Pull the latest vllm docker:
```shell
docker pull rocm/vllm-dev:nightly
```
Launch the ROCm vLLM docker: 
```shell
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
  --name Kimi-Linear-48B-A3B-Instruct \
  rocm/vllm-dev:nightly
```
### Step 2: Log in to Hugging Face
Huggingface login
```shell
huggingface-cli login
```

### Step 3: Start the vLLM server

Run the vllm online serving
Sample Command
```shell
SAFETENSORS_FAST_GPU=1 \
VLLM_USE_V1=1 \
VLLM_USE_TRITON_FLASH_ATTN=0 vllm serve moonshotai/Kimi-Linear-48B-A3B-Instruct \
  --tensor-parallel-size 8 \
  --max-model-len 1048576 \
  --no-enable-prefix-caching \
  --trust-remote-code
```


### Step 4: Run Benchmark
Open a new terminal and run the following command to execute the benchmark script inside the container.
```shell
docker exec -it Kimi-Linear-48B-A3B-Instruct vllm bench serve \
  --model "moonshotai/Kimi-Linear-48B-A3B-Instruct" \
  --dataset-name random \
  --random-input-len 8192 \
  --random-output-len 1024 \
  --request-rate 10000 \
  --num-prompts 16 \
  --ignore-eos \
  --trust-remote-code 
```


  
