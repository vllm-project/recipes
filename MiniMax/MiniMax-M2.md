# MiniMax-M2 Usage Guide

[MiniMax-M2](https://huggingface.co/MiniMaxAI/MiniMax-M2) is an advanced large language model created by [MiniMax](https://www.minimax.io/). It offers the following highlights:

* Superior Intelligence – Ranks #1 among open-source models globally across mathematics, science, coding, and tool use.
* Advanced Coding – Excels at multi-file edits, coding-run-fix loops, and test-validated repairs. Strong performance on SWE-Bench and Terminal-Bench tasks.
* Agent Performance – Plans and executes complex toolchains across shell, browser, and code environments. Maintains traceable evidence and recovers gracefully from errors.
* Efficient Design – 10B activated parameters (200B total) enables lower latency, cost, and higher throughput for interactive and batched workloads.

## Installing vLLM

```bash
uv venv
source .venv/bin/activate
uv pip install vllm --extra-index-url https://wheels.vllm.ai/nightly
```

## Launching MiniMax-M2 with vLLM

You can use 4x H200/H20 or 4x A100/A800 GPUs to launch this model.

run tensor-parallel like this:

```bash
vllm serve MiniMaxAI/MiniMax-M2 \
  --tensor-parallel-size 4 \
  --tool-call-parser minimax_m2 \
  --reasoning-parser minimax_m2_append_think  \
  --enable-auto-tool-choice
```

## Performance Metrics


### Benchmarking

We use the following script to demonstrate how to benchmark MiniMaxAI/MiniMax-M2`.

```bash
vllm bench serve \
  --backend vllm \
  --model MiniMaxAI/MiniMax-M2 \
  --served-model-name  minmax-m2 \
  --endpoint /v1/completions \
  --dataset-name random \
  --random-input 2048 \
  --random-output 1024 \
  --max-concurrency 10 \
  --num-prompt 100 
```



