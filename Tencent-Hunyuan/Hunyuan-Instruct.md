# Hunyuan-A13B Instruct Usage Guide

This guide provides instructions to install and run Hunyuan-A13B-Instruct on AMD GPUs.

## Install vLLM
> Note: The vLLM wheel for ROCm requires Python 3.12 and glibc >= 2.35. If your environment does not meet these requirements, please use the Docker-based setup as described in the [documentation](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/#pre-built-images). Supported GPUs: MI300X, MI325X, MI355X

```bash
uv venv
source .venv/bin/activate
uv pip install vllm --extra-index-url https://wheels.vllm.ai/rocm/
```

## Deploying Hunyuan-A13B Instruct

```bash 
export VLLM_ROCM_USE_AITER=1
vllm serve tencent/Hunyuan-A13B-Instruct --tensor-parallel-size 2 --trust-remote-code 
```

## Benchmarking

```bash
vllm bench serve \
  --model "tencent/Hunyuan-A13B-Instruct" \
  --dataset-name random \
  --random-input-len 8000 \
  --random-output-len 1000 \
  --request-rate 10000 \
  --num-prompts 16 \
  --ignore-eos
```
