# Kimi Linear 48B A3B Instruct Usage Guide

[Kimi Linear 48B A3B Instruct](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct) is an LLM model. This guide covers how to efficiently deploy and serve the model using VLLM.

## Installing vLLM

You can either install vLLM from pip or use the pre-built Docker image.

### Pip Install

#### NVIDIA
```bash
uv venv
source .venv/bin/activate
uv pip install -U vllm --torch-backend=auto
```

#### AMD
> Note: The vLLM wheel for ROCm requires Python 3.12, ROCm 7.0, and glibc >= 2.35. If your environment does not meet these requirements, please use the Docker-based setup as described below. Supported GPUs: MI300X, MI325X, MI355X.
```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install vllm --extra-index-url https://wheels.vllm.ai/rocm
```

### Docker

#### NVIDIA

```bash
docker run --gpus all \
  -p 8000:8000 \
  --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai moonshotai/Kimi-Linear-48B-A3B-Instruct 
```

For Blackwell GPUs, use `vllm/vllm-openai:cu130-nightly`

#### AMD

```bash
docker run --device=/dev/kfd --device=/dev/dri \
  --security-opt seccomp=unconfined \
  --group-add video \
  --ipc=host \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai-rocm:latest \
  moonshotai/Kimi-Linear-48B-A3B-Instruct 
```

## Running Kimi Linear 48B A3B Instruct

The configurations below have been verified on AMD MI300X/MI355X GPUs.

### Basic Serving

```bash
vllm serve moonshotai/Kimi-Linear-48B-A3B-Instruct 
```

### MI300X/MI355X Deployment

For deployment on 1x MI300X/MI355X GPU:

```bash
vllm serve moonshotai/Kimi-Linear-48B-A3B-Instruct \
  --tensor-parallel-size 1
```


### Benchmarking

Once the server is running, open another terminal and run the benchmark client:

```bash
vllm bench serve \
  --backend openai-chat \
  --endpoint /v1/chat/completions \
  --model moonshotai/Kimi-Linear-48B-A3B-Instruct \
  --dataset-name random \
  --random-input-len 2048 \
  --random-output-len 512 \
  --num-prompts 1000 \
  --request-rate 20
```

### Consume the OpenAI API Compatible Server

```python
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
    timeout=3600
)

response = client.chat.completions.create(
    model="moonshotai/Kimi-Linear-48B-A3B-Instruct",
    messages=[
        {"role": "user", "content": "Hello! How are you?"}
    ],
    max_tokens=512
)
print(response.choices[0].message.content)
```
