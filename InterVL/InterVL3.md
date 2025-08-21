# InternVL3 Usage Guide

This guide describes how to run InternVL3 series on NVIDIA GPUs.

[InterVL3](https://huggingface.co/collections/OpenGVLab/internvl3-67f7f690be79c2fe9d74fe9d) is a powerful multimodal model that combines vision and language understanding capabilities. This recipe provides step-by-step instructions for running InterVL3 using vLLM, optimized for various hardware configurations.

## Deployment Steps

### Installing vLLM

```bash
uv venv
source .venv/bin/activate
uv pip install -U vllm --torch-backend auto
```

### Weights
[OpenGVLab/InternVL3-8B-hf](https://huggingface.co/OpenGVLab/InternVL3-8B-hf)

### Running InternVL3-8B-hf model on A100-SXM4-40GB GPUs (2 cards) in eager mode

Launch the online inference server using TP=2:
```bash
export CUDA_VISIBLE_DEVICES=0,1
vllm serve OpenGVLab/InternVL3-8B-hf --enforce-eager \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 2 \
  --data-parallel-size 1
```

## Configs and Parameters

`--enforce-eager` disables the CUDA Graph in PyTorch; otherwise, it will throw error `torch._dynamo.exc.Unsupported: Data-dependent branching` during testing. For more information about CUDA Graph, please check [Accelerating-pytorch-with-cuda-graphs](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/)

`--tensor-parallel-size` sets Tensor Parallel (TP).

`--data-parallel-size` sets Data-parallel (DP).



## Validation & Expected Behavior

### Basic Test
Open another terminal, and use the following commands:
```bash
# need to start vLLM service first
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "<|begin_of_text|><|system|>\nYou are a helpful AI assistant.\n<|user|>\nWhat is the capital of France?\n<|assistant|>",
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

The result would be like this:
```json
{
"id": "cmpl-1ed0df81b56448afa597215a8725c686",
"object": "text_completion",
"created": 1755739470,
"model": "OpenGVLab/InternVL3-8B-hf",
"choices":
  [{
  "index":0,
  "text":" The capital of France is Paris.",
  "logprobs":null,
  "finish_reason":"stop",
  "stop_reason":null,
  "prompt_logprobs":null
  }],
"service_tier":null,
"system_fingerprint":null,
"usage":
  {
  "prompt_tokens":35,
  "total_tokens":43,
  "completion_tokens":8,
  "prompt_tokens_details":null
  },
"kv_transfer_params":null}
```

### Benchmarking Performance

Take InternVL3-8B-hf as an example:

```bash
# need to start vLLM service first
vllm bench serve \
  --host 0.0.0.0 \
  --port 8000 \
  --model OpenGVLab/InternVL3-8B-hf \
  --dataset-name random \
  --random-input 2048 \
  --random-output 1024 \
  --max-concurrency 10 \
  --num-prompts 50 \
  --ignore-eos
```
If it works successfully, you will see the following output.

```
============ Serving Benchmark Result ============
Successful requests:                     497
Benchmark duration (s):                  229.42
Total input tokens:                      507680
Total generated tokens:                  62259
Request throughput (req/s):              2.17
Output token throughput (tok/s):         271.37
Total Token throughput (tok/s):          2484.22
---------------Time to First Token----------------
Mean TTFT (ms):                          102429.40
Median TTFT (ms):                        99644.38
P99 TTFT (ms):                           213820.81
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          664.26
Median TPOT (ms):                        776.39
P99 TPOT (ms):                           848.52
---------------Inter-token Latency----------------
Mean ITL (ms):                           661.73
Median ITL (ms):                         844.15
P99 ITL (ms):                            856.42
==================================================
```
