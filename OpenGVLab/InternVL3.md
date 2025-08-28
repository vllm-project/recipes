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

### Running InternVL3-8B-hf model on A100-SXM4-40GB GPUs (2 cards)

Launch the online inference server using TP=2:
```bash
export CUDA_VISIBLE_DEVICES=0,1
vllm serve OpenGVLab/InternVL3-8B-hf \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 2 \
  --data-parallel-size 1
```

## Configs and Parameters

* You can set `limit-mm-per-prompt` to limit how many multimodal data instances to allow for each prompt. This is useful if you want to control the incoming traffic of multimodal requests. E.g., `--limit-mm-per-prompt '{"image":2, "video":0}'`

* You can set `--tensor-parallel-size` and `--data-parallel-size` to adjust the parallel strategy.

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

#### InternVL3-8B-hf on Multimodal Random Dataset

Take InternVL3-8B-hf as an example, using the random multimodal dataset mentioned in [this vLLM PR](https://github.com/vllm-project/vllm/pull/23119):

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

#### InternVL3-8B-hf on VisionArena-Chat Dataset

```bash
# need to start vLLM service first
vllm bench serve \
    --host 0.0.0.0 \
    --port 8000 \
    --endpoint /v1/chat/completions \
    --endpoint-type openai-chat \
    --model OpenGVLab/InternVL3-8B-hf \
    --dataset-name hf \
    --dataset-path lmarena-ai/VisionArena-Chat \
    --num-prompts 1000 
```
If it works successfully, you will see the following output.

```
============ Serving Benchmark Result ============
Successful requests:                     1000
Benchmark duration (s):                  597.45
Total input tokens:                      109173
Total generated tokens:                  109352
Request throughput (req/s):              1.67
Output token throughput (tok/s):         183.03
Total Token throughput (tok/s):          365.76
---------------Time to First Token----------------
Mean TTFT (ms):                          280208.05
Median TTFT (ms):                        270322.52
P99 TTFT (ms):                           582602.60
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          519.16
Median TPOT (ms):                        539.03
P99 TPOT (ms):                           596.74
---------------Inter-token Latency----------------
Mean ITL (ms):                           593.88
Median ITL (ms):                         530.72
P99 ITL (ms):                            4129.92
==================================================
```
