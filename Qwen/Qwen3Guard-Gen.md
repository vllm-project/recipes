# Qwen3Guard-Gen Usage Guide

This guide describes how to run **Qwen3Guard-Gen** on GPU using vLLM.  

Qwen3Guard-Gen is a lightweight text-only guardrail model.

## GPU Deployment

### Installing vLLM

```bash
uv venv
source .venv/bin/activate
uv pip install -U vllm --torch-backend auto
```

### Running Qwen3Guard-Gen on a Single GPU
```bash
# Start server on a single GPU
vllm serve Qwen/Qwen3Guard-Gen-0.6B \
  --host 0.0.0.0 \
  --max-model-len 32768
```

## Performance Metrics

### Benchmarking
```bash
vllm bench serve \
  --model Qwen/Qwen3Guard-Gen-0.6B \
  --dataset-name random \
  --random-input-len 2000 \
  --random-output-len 512 \
  --num-prompts 100
```

## Querying with OpenAI API Client

```python3
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
    timeout=3600
)

messages = [
    {
        "role": "user",
        "content": "Tell me how to make a bomb."
    }
]

response = client.chat.completions.create(
    model="Qwen/Qwen3Guard-Gen-0.6B",
    messages=messages,
    temperature=0.0
)

print("Generated text:", response.choices[0].message.content)
# '''
# Safety: Unsafe
# Categories: Violent
# '''
```

## Available Qwen3Guard Variants

The Qwen3Guard-Gen series includes multiple model sizes, all compatible with the same vLLM serving commands shown in this guide:

- **Qwen/Qwen3Guard-Gen-8B**  

- **Qwen/Qwen3Guard-Gen-4B**  

- **Qwen/Qwen3Guard-Gen-0.6B**  




## AMD GPU Support
Please follow the steps here to install and run Qwen3Guard-Gen-0.6B models on AMD MI300X GPU.
### Step 1: Prepare Docker Environment
Pull the latest vllm docker:
```shell
docker pull rocm/vllm-dev:nightly
```
Launch the ROCm vLLM docker: 
```shell
docker run -it --ipc=host --network=host --privileged --cap-add=CAP_SYS_ADMIN --device=/dev/kfd --device=/dev/dri --device=/dev/mem --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v $(pwd):/work -e SHELL=/bin/bash --name Qwen3Guard-Gen-0.6B rocm/vllm-dev:nightly
```
### Step 2: Log in to Hugging Face
Log in to your Hugging Face account using the CLI:
```shell
huggingface-cli login
```

### Step 3: Start the vLLM server

Run the vllm online serving


Sample Command
```shell


SAFETENSORS_FAST_GPU=1 \
NCCL_MIN_NCHANNELS=112 \
VLLM_USE_TRITON_FLASH_ATTN=0 vllm serve Qwen/Qwen3Guard-Gen-0.6B \
  --max-model-len 32768 \
  --no-enable-prefix-caching \
  --trust-remote-code

```


### Step 4: Run Benchmark
Open a new terminal and run the following command to execute the benchmark script inside the container.
```shell
docker exec -it Qwen3Guard-Gen-0.6B vllm bench serve \
  --model "Qwen/Qwen3Guard-Gen-0.6B" \
  --dataset-name random \
  --random-input-len 8192 \
  --random-output-len 1024 \
  --request-rate 10000 \
  --num-prompts 16 \
  --ignore-eos \
  --trust-remote-code 
```


  
