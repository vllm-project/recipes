# InternVL3 Usage Guide

This guide describes how to run InternVL3 series on NVIDIA GPUs.

[InternVL3](https://huggingface.co/collections/OpenGVLab/internvl3-67f7f690be79c2fe9d74fe9d) is a powerful multimodal model that combines vision and language understanding capabilities. This recipe provides step-by-step instructions for running InternVL3 using vLLM, optimized for various hardware configurations.

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

Take InternVL3-8B-hf as an example, using random multimodal dataset mentioned in [PR:Feature/benchmark/random mm data/images](https://github.com/vllm-project/vllm/pull/23119):

```bash
# need to start vLLM service first
vllm bench serve \
    --host 0.0.0.0 \
    --port 8000 \
    --model OpenGVLab/InternVL3-8B-hf \
    --dataset-name random-mm \
    --num-prompts 100 \
    --max-concurrency 10 \
    --random-prefix-len 25 \
    --random-input-len 300 \
    --random-output-len 40 \
    --random-range-ratio 0.2 \
    --random-mm-base-items-per-request 0 \
    --random-mm-num-mm-items-range-ratio 0 \
    --random-mm-limit-mm-per-prompt '{"image":3,"video":0}' \
    --random-mm-bucket-config '{(256, 256, 1): 0.25, (720, 1280, 1): 0.75}' \
    --request-rate inf \
    --ignore-eos \
    --endpoint-type openai-chat \
    --endpoint "/v1/chat/completions" \
    --seed 42 
```
If it works successfully, you will see the following output.

```
============ Serving Benchmark Result ============
Successful requests:                     100
Maximum request concurrency:             10
Benchmark duration (s):                  24.54
Total input tokens:                      32805
Total generated tokens:                  3982
Request throughput (req/s):              4.07
Output token throughput (tok/s):         162.25
Total Token throughput (tok/s):          1498.91
---------------Time to First Token----------------
Mean TTFT (ms):                          198.18
Median TTFT (ms):                        158.99
P99 TTFT (ms):                           524.05
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          55.56
Median TPOT (ms):                        56.04
P99 TPOT (ms):                           60.32
---------------Inter-token Latency----------------
Mean ITL (ms):                           54.22
Median ITL (ms):                         47.02
P99 ITL (ms):                            116.90
==================================================
```
