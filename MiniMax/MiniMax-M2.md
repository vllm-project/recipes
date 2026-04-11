# MiniMax-M2 Series Usage Guide

[MiniMax-M2.5](https://huggingface.co/MiniMaxAI/MiniMax-M2.5), [MiniMax-M2.1](https://huggingface.co/MiniMaxAI/MiniMax-M2.1) and [MiniMax-M2](https://huggingface.co/MiniMaxAI/MiniMax-M2) are advanced large language models created by [MiniMax](https://www.minimax.io/). They offer the following highlights:

* Superior Intelligence – Ranks #1 among open-source models globally across mathematics, science, coding, and tool use.
* Advanced Coding – Excels at multi-file edits, coding-run-fix loops, and test-validated repairs. Strong performance on SWE-Bench and Terminal-Bench tasks.
* Agent Performance – Plans and executes complex toolchains across shell, browser, and code environments. Maintains traceable evidence and recovers gracefully from errors.
* Efficient Design – 10B activated parameters (230B total) enables lower latency, cost, and higher throughput for interactive and batched workloads.

## Supported Models

This guide applies to the following models. You only need to update the model name during deployment. The following examples use **MiniMax-M2.7**:

- [MiniMaxAI/MiniMax-M2.7](https://huggingface.co/MiniMaxAI/MiniMax-M2.7)
- [MiniMaxAI/MiniMax-M2.5](https://huggingface.co/MiniMaxAI/MiniMax-M2.5)
- [MiniMaxAI/MiniMax-M2.1](https://huggingface.co/MiniMaxAI/MiniMax-M2.1)
- [MiniMaxAI/MiniMax-M2](https://huggingface.co/MiniMaxAI/MiniMax-M2)

## Usage

## System Requirements

- OS: Linux

- Python: 3.10 - 3.13

- GPU requirements

    <details>
    <summary>NVIDIA GPU</summary>

      - compute capability 7.0 or higher

      - Memory requirements: 220 GB for weights, 240 GB per 1M context tokens

    </details>

    <details>
    <summary>AMD GPU</summary>

    The following are recommended configurations; actual requirements should be adjusted based on your use case:

    - **96G x4** GPU: Supports a total KV Cache capacity of 400K tokens.

    - **144G x8** GPU: Supports a total KV Cache capacity of up to 3M tokens.

    - **192G x2** AMD GPU (MI300X/MI325X): Supports a total KV Cache capacity of ~500K tokens.

    - **192G x4** AMD GPU (MI300X/MI325X): Supports a total KV Cache capacity of ~1.5M tokens.

    - **288G x2** AMD GPU (MI350X/MI355X): Supports a total KV Cache capacity of ~1.5M tokens.

    - **288G x4** AMD GPU (MI350X/MI355X): Supports a total KV Cache capacity of ~4M tokens.

    > **Note**: The values above represent the total aggregate hardware KV Cache capacity. The maximum context length per individual sequence remains **196K** tokens.
    </details>

### Using Docker

- We provide docker images specifically for the M2 series.

```bash
docker run --gpus all \
  -p 8000:8000 \
  --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:nightly MiniMaxAI/MiniMax-M2.7 \
      --tensor-parallel-size 4 \
      --tool-call-parser minimax_m2 \
      --reasoning-parser minimax_m2_append_think \
      --enable-auto-tool-choice \
      --compilation-config '{"mode":3,"pass_config":{"fuse_minimax_qk_norm":true}}' \
      --trust-remote-code
```

### Installing vLLM for AMD GPU (ROCm)

Install the vLLM ROCm wheel (requires Python 3.12 and ROCm 7.0+):

```bash
uv pip install vllm --extra-index-url https://wheels.vllm.ai/rocm/
```

Supported AMD GPUs: MI300X, MI325X, MI350X, MI355X.

### Installing vLLM from source

#### Install nightly version

- If you encounter corrupted output when using vLLM to serve these models, you can upgrade to the nightly version (ensure it is a version after commit [cf3eacfe58fa9e745c2854782ada884a9f992cf7](https://github.com/vllm-project/vllm/commit/cf3eacfe58fa9e745c2854782ada884a9f992cf7))

```bash
uv venv
source .venv/bin/activate
uv pip install -U vllm --extra-index-url https://wheels.vllm.ai/nightly
```

#### Install verified version

- We also verified the accuracy of `MiniMax-M2.7` in commit [0f3ce4c74b1875791d6604e006b6e905fde9f698](https://github.com/vllm-project/vllm/commit/0f3ce4c74b1875791d6604e006b6e905fde9f698)
  on AIME25, GPQA-D, and GSM8K.  You can use the following method to install this version.

```bash
uv venv
source .venv/bin/activate
export VLLM_COMMIT=0f3ce4c74b1875791d6604e006b6e905fde9f698 # use full commit hash from the main branch
uv pip install vllm \
    --torch-backend=auto \
    --extra-index-url https://wheels.vllm.ai/${VLLM_COMMIT} # add variant subdirectory here if needed
```



## Launching  M2 series with vLLM

### NVIDIA GPU

You can use 4x H200/H20/H100 or 4x A100/A800 GPUs to launch this model.

run tensor-parallel like this:

```bash
vllm serve MiniMaxAI/MiniMax-M2.7 \
  --tensor-parallel-size 4 \
  --tool-call-parser minimax_m2 \
  --reasoning-parser minimax_m2_append_think  \
  --compilation-config '{"mode":3,"pass_config":{"fuse_minimax_qk_norm":true}}' \
  --enable-auto-tool-choice
```

Note that pure TP8 is not supported. To run the model with >4 GPUs, please use DP+EP or TP+EP:

- DP8+EP
```bash
vllm serve MiniMaxAI/MiniMax-M2.7 \
  --data-parallel-size 8 \
  --enable-expert-parallel \
  --tool-call-parser minimax_m2 \
  --reasoning-parser minimax_m2_append_think  \
  --compilation-config '{"mode":3,"pass_config":{"fuse_minimax_qk_norm":true}}' \
  --enable-auto-tool-choice
```

- TP4+EP (recommended for H100)

> **Note**: On H100 GPUs, TP4+EP4 outperforms TP8+EP8 and is the recommended configuration.

```bash
vllm serve MiniMaxAI/MiniMax-M2.7 \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --tool-call-parser minimax_m2 \
  --reasoning-parser minimax_m2_append_think  \
  --compilation-config '{"mode":3,"pass_config":{"fuse_minimax_qk_norm":true}}' \
  --enable-auto-tool-choice
```

- TP8+EP

```bash
vllm serve MiniMaxAI/MiniMax-M2.7 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --tool-call-parser minimax_m2 \
  --reasoning-parser minimax_m2_append_think  \
  --compilation-config '{"mode":3,"pass_config":{"fuse_minimax_qk_norm":true}}' \
  --enable-auto-tool-choice
```

> **Note**: `fuse_minimax_qk_norm` fusion is a feature introduced in [#PR 37045](https://github.com/vllm-project/vllm/pull/37045). To use this feature, please ensure that the vllm you are using includes this PR.


### AMD GPU (ROCm)

You can use 2x or 4x MI300X/MI325X/MI350X/MI355X GPUs to launch this model with [AITER](https://github.com/ROCm/aiter) acceleration enabled:

- TP2 (2x MI300X/MI325X/MI350X/MI355X)
```bash
VLLM_ROCM_USE_AITER=1 vllm serve MiniMaxAI/MiniMax-M2.7 \
  --tensor-parallel-size 2 \
  --tool-call-parser minimax_m2 \
  --reasoning-parser minimax_m2_append_think \
  --enable-auto-tool-choice \
  --trust-remote-code
```

- TP4 (4x MI300X/MI325X/MI350X/MI355X)
```bash
VLLM_ROCM_USE_AITER=1 vllm serve MiniMaxAI/MiniMax-M2.7 \
  --tensor-parallel-size 4 \
  --tool-call-parser minimax_m2 \
  --reasoning-parser minimax_m2_append_think \
  --enable-auto-tool-choice \
  --trust-remote-code
```

> **Note**: The first launch with AITER may take several minutes as AITER JIT-compiles optimized kernels (CK-based FP8 MoE, RMSNorm, activation, etc.). Subsequent launches will use cached kernels.



## Reasoning parser

To run the model in responsesAPI that natively supports thinking, run it with the minimax_m2 reasoning parser:
```bash
vllm serve MiniMaxAI/MiniMax-M2.7 \
  --tensor-parallel-size 4 \
  --tool-call-parser minimax_m2 \
  --reasoning-parser minimax_m2 \
  --enable-auto-tool-choice
```



## Performance Metrics


### Benchmarking

We use the following script to demonstrate how to benchmark MiniMax-M2 models.

```bash
vllm bench serve \
  --backend vllm \
  --model MiniMaxAI/MiniMax-M2 \
  --endpoint /v1/completions \
  --dataset-name random \
  --random-input 2048 \
  --random-output 1024 \
  --max-concurrency 10 \
  --num-prompt 100 
```


If successful, you should see output similar to the following:

```
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

## Using Tips

### DeepGEMM Usage

vLLM has DeepGEMM enabled by default, you can install DeepGEMM using [install_deepgemm.sh](https://github.com/vllm-project/vllm/blob/main/tools/install_deepgemm.sh).

