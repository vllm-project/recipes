# LFM2.5 Usage Guide

[LFM2.5](https://www.liquid.ai/) is [Liquid AI's](https://www.liquid.ai/) family of small,
efficient open-weight models built on the **LFM2 hybrid backbone** — short-range gated
convolution blocks interleaved with grouped-query attention. The hybrid design keeps the KV
cache small and decode fast, so LFM2.5 models punch above their weight while remaining cheap to
serve, down to edge / on-device GPUs. The family spans dense chat models (230M, 350M, 1.2B), a
mixture-of-experts model (8B-A1B), reasoning and Japanese variants, base checkpoints, and
vision-language models (VL 450M / 1.6B) — all served through vLLM's OpenAI-compatible API.

All LFM2.5 language and vision-language models are supported **natively** by vLLM (the
`Lfm2ForCausalLM`, `Lfm2MoeForCausalLM`, and `Lfm2VlForConditionalGeneration` architectures). Tool
calling is handled by the built-in `lfm2` tool parser, and `<think>` reasoning by the `qwen3`
reasoning parser.

## Supported Models

### Dense Chat Models

| Model | Parameters | Min NVIDIA GPU (BF16) | Context | Tools | HuggingFace |
|-------|-----------|------------------------|---------|-------|-------------|
| LFM2.5 230M | 230M | 1× (any) | 32K | ✓ | [LiquidAI/LFM2.5-230M](https://huggingface.co/LiquidAI/LFM2.5-230M) |
| LFM2.5 350M | 350M | 1× (any) | 32K | ✓ | [LiquidAI/LFM2.5-350M](https://huggingface.co/LiquidAI/LFM2.5-350M) |
| LFM2.5 1.2B Instruct | 1.2B | 1× (8 GB+) | 32K | ✓ | [LiquidAI/LFM2.5-1.2B-Instruct](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct) |
| LFM2.5 1.2B Thinking | 1.2B | 1× (8 GB+) | 32K | ✓ (+reasoning) | [LiquidAI/LFM2.5-1.2B-Thinking](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking) |
| LFM2.5 1.2B JP | 1.2B | 1× (8 GB+) | 32K | – | [LiquidAI/LFM2.5-1.2B-JP](https://huggingface.co/LiquidAI/LFM2.5-1.2B-JP) |
| LFM2.5 1.2B JP (202606) | 1.2B | 1× (8 GB+) | 32K | ✓ | [LiquidAI/LFM2.5-1.2B-JP-202606](https://huggingface.co/LiquidAI/LFM2.5-1.2B-JP-202606) |
| LFM2.5 1.2B Base | 1.2B | 1× (8 GB+) | 32K | – (completions) | [LiquidAI/LFM2.5-1.2B-Base](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Base) |

### Mixture-of-Experts (MoE) Model

| Model | Total / Active Params | Min NVIDIA GPU (BF16) | Context | HuggingFace |
|-------|----------------------|------------------------|---------|-------------|
| LFM2.5 8B-A1B | 8B / ~1B active | 1× (24 GB+) | 128K | [LiquidAI/LFM2.5-8B-A1B](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B) |

The 8B-A1B keeps every expert resident in VRAM, so size the GPU for the full ~8B of weights even
though only ~1B is active per token. It supports `<think>` reasoning and tool calling.

### Vision-Language Models

| Model | Parameters | Min NVIDIA GPU (BF16) | Context | HuggingFace |
|-------|-----------|------------------------|---------|-------------|
| LFM2.5 VL 450M | 450M | 1× (any) | 32K | [LiquidAI/LFM2.5-VL-450M](https://huggingface.co/LiquidAI/LFM2.5-VL-450M) |
| LFM2.5 VL 1.6B | 1.6B | 1× (8 GB+) | 32K | [LiquidAI/LFM2.5-VL-1.6B](https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B) |

The VL models pair the LFM2 hybrid language backbone with a SigLIP2 vision encoder
(`Lfm2VlForConditionalGeneration`).

## Installing vLLM

LFM2.5's dense, MoE, and VL architectures (`Lfm2ForCausalLM`, `Lfm2MoeForCausalLM`,
`Lfm2VlForConditionalGeneration`) run on **vLLM 0.23.0**, which `min_vllm_version` pins.

### pip (NVIDIA CUDA)

```bash
uv venv
source .venv/bin/activate
uv pip install -U vllm --torch-backend auto
```

### Docker

```bash
docker pull vllm/vllm-openai:latest        # CUDA 12.x
docker pull vllm/vllm-openai:latest-cu130  # CUDA 13.0 (Blackwell)
```

## Running LFM2.5

### Quick Start (Single GPU)

```bash
vllm serve LiquidAI/LFM2.5-1.2B-Instruct
```

Cap the context to fit a smaller GPU (models support up to 32K; 128K on 8B-A1B):

```bash
vllm serve LiquidAI/LFM2.5-1.2B-Instruct --max-model-len 32768
```

### Multi-GPU

Every LFM2.5 model fits on a single GPU (the 8B-A1B MoE on a 24 GB+ GPU), so tensor parallelism
gives no meaningful speedup on basically any GPU — there's nothing to split that helps. You can
still try it on a multi-GPU node, but expect no throughput gain:

```bash
vllm serve LiquidAI/LFM2.5-8B-A1B --tensor-parallel-size 2
```

### Docker Deployment

```bash
docker run -itd --name lfm2.5 \
    --ipc=host --network host --shm-size 16G --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    vllm/vllm-openai:latest \
        --model LiquidAI/LFM2.5-1.2B-Instruct \
        --host 0.0.0.0 --port 8000
```

## Recommended Sampling

LFM2.5 uses per-model sampling presets from the model cards. `top_k`, `min_p`, and
`repetition_penalty` are vLLM extra sampling params — pass them via `extra_body` on the OpenAI
client (or top-level in a raw `/v1/...` request).

| Model | temperature | top_k | min_p | repetition_penalty |
|-------|------------:|------:|------:|-------------------:|
| LFM2.5 230M | 0.1 | 50 | – | 1.05 |
| LFM2.5 350M | 0.1 | 50 | – | 1.05 |
| LFM2.5 1.2B Instruct | 0.1 | 50 | – | 1.05 |
| LFM2.5 1.2B Thinking | 0.05 | 50 | – | 1.05 |
| LFM2.5 1.2B JP | 0.3 | – | 0.15 | 1.05 |
| LFM2.5 1.2B JP (202606) | 0.1 | 50 | – | 1.05 |
| LFM2.5 1.2B Base | 0.3 | – | 0.15 | 1.05 |
| LFM2.5 8B-A1B | 0.2 | 80 | – | 1.05 |
| LFM2.5 VL 450M / 1.6B | 0.1 | – | 0.15 | 1.05 |

> ⚠️ Do **not** bake these into `vllm serve` — they are per-request client defaults, not server
> flags. Leave `max_tokens` unset — capping it truncates the reasoning models' chain-of-thought.

## Text Generation

### Online Serving (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
response = client.chat.completions.create(
    model="LiquidAI/LFM2.5-1.2B-Instruct",
    messages=[{"role": "user", "content": "What is C. elegans? Answer in one sentence."}],
    temperature=0.1,
    extra_body={"top_k": 50, "repetition_penalty": 1.05},
)
print(response.choices[0].message.content)
```

### Base Model (completions)

`LFM2.5-1.2B-Base` is not instruction-tuned and has no chat template — use the completions
endpoint:

```python
resp = client.completions.create(
    model="LiquidAI/LFM2.5-1.2B-Base",
    prompt="The three laws of thermodynamics are:",
    temperature=0.3,
    extra_body={"min_p": 0.15, "repetition_penalty": 1.05},
)
print(resp.choices[0].text)
```

## Reasoning Mode

`LFM2.5-8B-A1B` and `LFM2.5-1.2B-Thinking` emit an explicit `<think>…</think>` chain-of-thought.
Launch with the `qwen3` reasoning parser to split it into a separate `reasoning_content` field:

```bash
vllm serve LiquidAI/LFM2.5-1.2B-Thinking \
  --reasoning-parser qwen3
```

```python
response = client.chat.completions.create(
    model="LiquidAI/LFM2.5-1.2B-Thinking",
    messages=[{"role": "user", "content": "If a train travels 60 km in 45 minutes, what is its speed in km/h?"}],
    temperature=0.05,
    extra_body={"top_k": 50, "repetition_penalty": 1.05},
)
msg = response.choices[0].message
print("reasoning:", msg.reasoning_content)
print("answer:", msg.content)
```

> ℹ️ These models open the `<think>` channel for non-trivial problems; a trivial prompt may be
> answered directly, in which case `reasoning_content` is empty. That's expected behavior.

## Function Calling / Tool Use

LFM2.5 emits Pythonic tool calls wrapped in `<|tool_call_start|>…<|tool_call_end|>`. The built-in
`lfm2` tool parser converts these into standard OpenAI `tool_calls`:

```bash
vllm serve LiquidAI/LFM2.5-1.2B-Instruct \
  --enable-auto-tool-choice \
  --tool-call-parser lfm2
```

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
    },
}]

response = client.chat.completions.create(
    model="LiquidAI/LFM2.5-1.2B-Instruct",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=tools,
    temperature=0.1,
    extra_body={"top_k": 50, "repetition_penalty": 1.05},
)
print(response.choices[0].message.tool_calls)
```

Reasoning and tool calling can be combined for the reasoning models:

```bash
vllm serve LiquidAI/LFM2.5-8B-A1B \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser lfm2
```

## Image Understanding (VL)

The VL models accept image + text turns through the standard chat API:

```bash
vllm serve LiquidAI/LFM2.5-VL-1.6B
```

```python
response = client.chat.completions.create(
    model="LiquidAI/LFM2.5-VL-1.6B",
    messages=[{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"}},
            {"type": "text", "text": "What is in this image?"},
        ],
    }],
    temperature=0.1,
    extra_body={"min_p": 0.15, "repetition_penalty": 1.05},
)
print(response.choices[0].message.content)
```

Allow more than one image per request with `--limit-mm-per-prompt '{"image": 4}'`.

## Benchmarking

Disable prefix caching for consistent measurements:

```bash
vllm serve LiquidAI/LFM2.5-8B-A1B \
  --no-enable-prefix-caching

vllm bench serve \
  --model LiquidAI/LFM2.5-8B-A1B \
  --dataset-name random \
  --random-input-len 1000 \
  --random-output-len 1000 \
  --request-rate 10000 \
  --num-prompts 64 \
  --ignore-eos
```

## Speed-of-Light Tuning

The defaults are already strong for LFM2.5 — the model fits a single GPU (so tensor parallelism
gives no speedup) and the standard serve path is well optimized. A few flags give a small extra
gain:

| Flag | Effect | Notes |
|------|--------|-------|
| `VLLM_USE_OINK_OPS=1` | **~+2.6%** on **B200** | Routes RMSNorm to the Blackwell `oink` kernels (bundled in the `vllm-openai` image); output identical. **Auto-enabled on Blackwell** by these recipes; inert (native fallback) elsewhere. |
| `--optimization-level 3` | **~+2%** (all GPUs) | More aggressive compilation. Trades a longer one-time startup/compile for steady-state throughput — opt-in. |
| `VLLM_USE_FASTOKENS=1` | lower **TTFT** on tokenization-bound loads | Swaps the HF fast-tokenizer Rust BPE backend for the [`fastokens`](https://github.com/crusoecloud/fastokens) shim (`pip install fastokens`). Helps high-QPS / long-prompt workloads; the gain is in tokenization latency, so it doesn't show up in steady-state decode throughput. |

```bash
# -O3 (opt-in); on Blackwell, VLLM_USE_OINK_OPS=1 is applied for you by the recipe
vllm serve LiquidAI/LFM2.5-1.2B-Instruct --optimization-level 3

# tokenization-bound serving (after `pip install fastokens`)
VLLM_USE_FASTOKENS=1 vllm serve LiquidAI/LFM2.5-1.2B-Instruct
```

These knobs make little difference for single-GPU LFM2.5, but experiment if you like:
`VLLM_SSM_CONV_STATE_LAYOUT` (SD vs DS), `--mamba-backend` (triton vs flashinfer),
`--mamba-cache-mode` (none vs all), `--mm-processor-cache-type` (lru vs shm).

> **Coming:** the Lfm2VL encoder CUDA graph
> ([vllm-project/vllm#44930](https://github.com/vllm-project/vllm/pull/44930), ~10–20% lower e2e
> latency at low batch) is not in 0.23.0 — it will be added once it ships in a stable release.

## Server Flags Reference

| Flag | Description | When |
|------|-------------|------|
| `--reasoning-parser qwen3` | Split `<think>…</think>` into `reasoning_content` | 8B-A1B, 1.2B-Thinking |
| `--tool-call-parser lfm2` | Surface Pythonic tool calls as `tool_calls` | tool-capable models |
| `--enable-auto-tool-choice` | Auto-detect tool calls in output | with `--tool-call-parser` |
| `--max-model-len N` | Cap context (up to 32K; 128K on 8B-A1B) | small GPUs / fixed workload |
| `--limit-mm-per-prompt '{"image": N}'` | Max images per request | VL models |

## Deploy on Modal

[Modal](https://modal.com) runs this recipe on cloud GPUs with a single command — no
infrastructure to manage. The deployment script is [`lfm25-modal.py`](lfm25-modal.py) in this
directory: it serves an LFM2.5 model with vLLM behind an OpenAI-compatible endpoint, with the
model and GPU selectable via environment variables.

### Deploy

```bash
pip install modal
modal setup                  # one-time: authenticate with Modal
modal deploy lfm25-modal.py  # serves LiquidAI/LFM2.5-1.2B-Instruct on an L4 by default
```

### Test

```bash
modal run lfm25-modal.py
```

### Pick a model / GPU

```bash
MODEL=LiquidAI/LFM2.5-8B-A1B GPU=H100 modal run lfm25-modal.py
```

LFM2.5's small footprint means even a budget GPU is plenty — the 1.2B dense checkpoint runs across
NVIDIA **T4, L4, A10G, L40S, A100 (40/80 GB), H100, H200, and B200**; the 8B-A1B MoE and the VL
models run on **H100, H200, and B200**. Size up to a 24 GB+ GPU (L4 / A10G or larger) for the
8B-A1B MoE, which keeps all ~8B of experts resident in VRAM.

## References

- [Liquid AI](https://www.liquid.ai/)
- [LiquidAI on HuggingFace](https://huggingface.co/LiquidAI)
- [LFM2.5-1.2B-Instruct model card](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct)
- [vLLM supported models](https://docs.vllm.ai/en/latest/models/supported_models.html)
