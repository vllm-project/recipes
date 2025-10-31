# Kimi-Linear Usage Guide

This guide describes how to run moonshotai/Kimi-Linear-48B-A3B-Instruct.

## Installing vLLM

```bash
uv venv
source .venv/bin/activate
uv pip install -U vllm --extra-index-url https://wheels.vllm.ai/nightly --prerelease=allow
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
