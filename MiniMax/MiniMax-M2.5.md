# MiniMax-M2.5 Usage Guide

This guide describes how to run [MiniMax-M2.5](https://huggingface.co/MiniMaxAI/MiniMax-M2.5) with vLLM.

## Installing vLLM

### Pip Install

If you encounter corrupted output when using vLLM to serve this model, you can upgrade to the nightly version (ensure it is a version after commit [cf3eacfe58fa9e745c2854782ada884a9f992cf7](https://github.com/vllm-project/vllm/commit/cf3eacfe58fa9e745c2854782ada884a9f992cf7)):

```bash
uv venv
source .venv/bin/activate
uv pip install -U vllm --extra-index-url https://wheels.vllm.ai/nightly
```

### Docker
```bash
docker run --gpus all \
  -p 8000:8000 \
  --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:nightly MiniMaxAI/MiniMax-M2.5 \
      --tensor-parallel-size 4 \
      --tool-call-parser minimax_m2 \
      --reasoning-parser minimax_m2_append_think \
      --enable-auto-tool-choice \
      --trust-remote-code
```

## Running MiniMax-M2.5

MiniMax-M2.5 can be run on different GPU configurations. The recommended setup uses 4x H200/H20 or 4x A100/A800 GPUs with tensor parallelism.

### B200 (FP8)

```bash
docker run --gpus all \
  -p 8000:8000 \
  --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:nightly MiniMaxAI/MiniMax-M2.5 \
      --tensor-parallel-size 4 \
      --tool-call-parser minimax_m2 \
      --reasoning-parser minimax_m2_append_think \
      --enable-auto-tool-choice \
      --trust-remote-code
```



## Benchmarking

We use the following script to demonstrate how to benchmark MiniMax-M2.5.

```bash
vllm bench serve \
  --backend vllm \
  --model MiniMaxAI/MiniMax-M2.5 \
  --endpoint /v1/completions \
  --dataset-name random \
  --random-input 2048 \
  --random-output 1024 \
  --max-concurrency 10 \
  --num-prompt 100
```

### Expected Output

```shell
============ Serving Benchmark Result ============
Successful requests:                     xxx
Failed requests:                         xxx
Maximum request concurrency:             xxx
Benchmark duration (s):                  xxx
Total input tokens:                      xxx
Total generated tokens:                  xxx
Request throughput (req/s):              xxx
Output token throughput (tok/s):         xxx
Peak output token throughput (tok/s):    xxx
Peak concurrent requests:                xxx
Total Token throughput (tok/s):          xxx
---------------Time to First Token----------------
Mean TTFT (ms):                          xxx
Median TTFT (ms):                        xxx
P99 TTFT (ms):                           xxx
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          xxx
Median TPOT (ms):                        xxx
P99 TPOT (ms):                           xxx
---------------Inter-token Latency----------------
Mean ITL (ms):                           xxx
Median ITL (ms):                         xxx
P99 ITL (ms):                            xxx
```

