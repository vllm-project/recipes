# Kimi-Linear Usage Guide

This guide describes how to run moonshotai/Kimi-Linear-48B-A3B-Instruct.

## Installing vLLM

### CUDA

```bash
uv venv
source .venv/bin/activate
# Install a stable version (avoid 0.12.0)
uv pip install vllm==0.11.2 --torch-backend auto
```
**Note**: Regarding Kimi-Linear, vLLM 0.12.0 has a known bug with `MLAModules.__init__() missing 1 required positional argument: 'indexer_rotary_emb'`. Please avoid this version.


### ROCm

You can choose either Option A (Docker) or Option B (install with uv).

#### Option A: Run on Host with uv
> Note: The vLLM wheel for ROCm requires Python 3.12, ROCm 7.0, and glibc >= 2.35. If your environment does not meet these requirements, please use the Docker-based setup as described in the [documentation](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/#pre-built-images).
```bash
uv venv
source .venv/bin/activate
uv pip install vllm --extra-index-url https://wheels.vllm.ai/rocm/
```

#### Option B: Run with Docker
Pull the latest vllm docker:
```shell
docker pull vllm/vllm-openai-rocm:latest
```
Launch the ROCm vLLM docker:
```shell
docker run -d -it \
  --ipc=host \
  --entrypoint /bin/bash \
  --network=host \
  --privileged \
  --cap-add=CAP_SYS_ADMIN \
  --device=/dev/kfd \
  --device=/dev/dri \
  --device=/dev/mem \
  --group-add video \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -v /:/work \
  -e SHELL=/bin/bash \
  -p 8000:8000 \
  --name Kimi-Linear-48B-A3B-Instruct \
  vllm/vllm-openai-rocm:latest
```

Log in to your Hugging Face account:
```shell
huggingface-cli login
```

## Running Kimi-Linear

### CUDA

It's easy to run Kimi-Linear.
The following snippets assume you have 4 or 8 GPUs on a single node.

#### 4-GPU tensor parallel
```bash
vllm serve moonshotai/Kimi-Linear-48B-A3B-Instruct \
  --port 8000 \
  --te\
  --max-model-len 1048576 \
  --trust-remote-code
```

#### 8-GPU tensor parallel
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

### ROCm

Run the vllm online serving with this sample command:
```shell
SAFETENSORS_FAST_GPU=1 \
vllm serve moonshotai/Kimi-Linear-48B-A3B-Instruct \
  --tensor-parallel-size 8 \
  --max-model-len 1048576 \
  --no-enable-prefix-caching \
  --trust-remote-code
```
