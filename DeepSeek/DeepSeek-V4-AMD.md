# DeepSeek-V4 on AMD (ROCm) Usage Guide

This recipe mirrors the official DeepSeek-V4 recipe structure and is adapted for AMD ROCm based on [vllm-project/vllm#40871](https://github.com/vllm-project/vllm/pull/40871).

## Scope

This guide covers:

- DeepSeek-V4-Flash on MI355X (online serving)
- DeepSeek-V4-Pro on MI355X (offline + online serving)
- Reasoning mode usage
- Tool calling flags
- MTP speculative decoding (experimental recommendation)

## Environment and Version

At the time of writing, AMD DeepSeek-V4 support is under review upstream, so use the PR branch build:

```bash
# inside ROCm container
pip uninstall -y vllm
git clone https://github.com/vllm-project/vllm.git
cd vllm
git fetch origin pull/40871/head:pr_dsv4
git checkout pr_dsv4
python3 setup.py develop
```

Reference runtime used in PR validation:

- Docker image: `rocm/vllm-dev:nightly_main_20260423`
- Hardware: `MI355X`

## DeepSeek-V4-Flash (MI355X)

### Launch

```bash
max_num_seqs=16
max_num_batched_tokens=1024
tensor_parallel_size=4

export HF_HOME=/data/huggingface-cache
export VLLM_ROCM_USE_AITER=1
export VLLM_TORCH_PROFILER_DIR=/app/vllm_profile

MODEL=/home/models/DeepSeek-V4-Flash
vllm serve ${MODEL} \
  --host localhost \
  --port 8001 \
  --dtype auto \
  --tensor-parallel-size ${tensor_parallel_size} \
  --max-num-seqs ${max_num_seqs} \
  --max-num-batched-tokens ${max_num_batched_tokens} \
  --distributed-executor-backend mp \
  --trust-remote-code \
  --profiler-config '{"profiler":"torch","torch_profiler_dir":"./vllm_profile"}' \
  --gpu-memory-utilization 0.35 \
  --moe-backend triton_unfused \
  --tokenizer-mode deepseek_v4 \
  --async-scheduling \
  --enforce-eager
```

### Smoke test

```bash
curl -s http://localhost:8001/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write me a poem about AMD and DeepSeek.",
    "model": "/home/models/DeepSeek-V4-Flash",
    "max_tokens": 100,
    "temperature": 0.0
  }'
```

### Accuracy check (GSM8K, from PR)

```bash
MODEL=/home/models/DeepSeek-V4-Flash
lm_eval --model local-completions \
  --model_args model=$MODEL,base_url=http://0.0.0.0:8001/v1/completions,num_concurrent=4,max_retries=10,max_gen_toks=2048,timeout=60000 \
  --batch_size auto \
  --tasks gsm8k \
  --num_fewshot 8 \
  --output_path .
```

Reported result:

- `flexible-extract exact_match`: `0.9439`
- `strict-match exact_match`: `0.9431`

## DeepSeek-V4-Pro (MI355X)

### Offline validation

```python
import os
from vllm import LLM, SamplingParams

os.environ["VLLM_ROCM_USE_AITER"] = "1"
os.environ["VLLM_ROCM_USE_AITER_LINEAR"] = "1"

prompts = ["What is 2+2? Answer:", "The capital of France is "]
sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=20)

llm = LLM(
    model="/home/models/DeepSeek-V4-Pro",
    tensor_parallel_size=8,
    kv_cache_dtype="fp8",
    gpu_memory_utilization=0.6,
    async_scheduling=True,
    enforce_eager=True,
    disable_log_stats=False,
    tokenizer_mode="deepseek_v4",
    moe_backend="triton_unfused",
    reasoning_parser="deepseek_v4",
)

outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    print(output.prompt, output.outputs[0].text)
```

### Online serving

```bash
max_num_seqs=128
max_num_batched_tokens=8192
tensor_parallel_size=8

export HF_HOME=/data/huggingface-cache
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_LINEAR=1
rm -rf /root/.cache/vllm/torch_compile_cache

MODEL=/home/models/DeepSeek-V4-Pro
vllm serve ${MODEL} \
  --host localhost \
  --port 8001 \
  --dtype auto \
  --kv-cache-dtype fp8 \
  --tensor-parallel-size ${tensor_parallel_size} \
  --max-num-seqs ${max_num_seqs} \
  --max-num-batched-tokens ${max_num_batched_tokens} \
  --distributed-executor-backend mp \
  --trust-remote-code \
  --gpu-memory-utilization 0.6 \
  --moe-backend triton_unfused \
  --tokenizer-mode deepseek_v4 \
  --reasoning-parser deepseek_v4 \
  --async-scheduling \
  --enforce-eager
```

### Accuracy check (GSM8K, from PR)

```bash
MODEL=/home/models/DeepSeek-V4-Pro
lm_eval --model local-completions \
  --model_args model=$MODEL,base_url=http://0.0.0.0:8001/v1/completions,num_concurrent=2,max_retries=10,max_gen_toks=2048,timeout=60000 \
  --batch_size auto \
  --tasks gsm8k \
  --num_fewshot 8 \
  --output_path .
```

Reported result:

- `flexible-extract exact_match`: `0.9538`
- `strict-match exact_match`: `0.9545`

## Reasoning modes

DeepSeek-V4 exposes non-think / think-high / think-max via `chat_template_kwargs`.

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8001/v1", api_key="EMPTY")
model = "deepseek-ai/DeepSeek-V4-Pro"
messages = [{"role": "user", "content": "What is 17*19? Return only the final integer."}]

# Non-think
client.chat.completions.create(model=model, messages=messages)

# Think high
client.chat.completions.create(
    model=model,
    messages=messages,
    extra_body={"chat_template_kwargs": {"thinking": True, "reasoning_effort": "high"}},
)

# Think max (ensure sufficient max-model-len)
client.chat.completions.create(
    model=model,
    messages=messages,
    extra_body={"chat_template_kwargs": {"thinking": True, "reasoning_effort": "max"}},
)
```

## Tool calling

Add these arguments to your serve command:

```bash
--tokenizer-mode deepseek_v4 \
--tool-call-parser deepseek_v4 \
--enable-auto-tool-choice
```

## Speculative decoding (MTP)

DeepSeek-V4 has native MTP support. On AMD, start conservatively and tune:

```bash
--speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

If memory/throughput allows, test:

```bash
--speculative-config '{"method":"mtp","num_speculative_tokens":2}'
```

## ROCm-specific notes from PR #40871

- ROCm path includes DeepSeek-V4 FP8 compatibility updates and E8M0 scale handling.
- ROCm execution disables some multi-stream paths to avoid known hang scenarios.
- For DeepSeek-V4 routing mode, `triton_unfused` is preferred for accuracy, with AITER as fallback.

## Troubleshooting

1. **`NotImplementedError: "mul_cuda" not implemented for 'Float8_e8m0fnu'`**
   - Ensure you are using the PR build above (or a newer commit that includes ROCm E8M0 handling fixes).
2. **Model hangs during startup/load**
   - Keep `--enforce-eager` enabled.
   - Use `--moe-backend triton_unfused` on AMD.
3. **Tokenizer / reasoning mismatch**
   - Verify `--tokenizer-mode deepseek_v4` and `--reasoning-parser deepseek_v4` are both set.

