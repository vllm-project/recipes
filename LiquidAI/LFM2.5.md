# LFM2.5 Usage Guide

[LFM2.5](https://www.liquid.ai/) is [Liquid AI's](https://www.liquid.ai/) family of small,
efficient open-weight models built on the **LFM2 hybrid backbone** — short-range gated
convolution blocks interleaved with grouped-query attention. The hybrid design keeps the KV
cache small and decode fast, so LFM2.5 models punch above their weight while remaining cheap to
serve, down to edge / on-device GPUs. The family spans dense chat models (350M, 1.2B), a
mixture-of-experts model (8B-A1B), reasoning and Japanese variants, base checkpoints, and
vision-language models (VL 450M / 1.6B) — all served through vLLM's OpenAI-compatible API.

All LFM2.5 language and vision-language models are supported **natively** by vLLM (the
`Lfm2ForCausalLM`, `Lfm2MoeForCausalLM`, and `Lfm2VlForConditionalGeneration` architectures), so
**no `--trust-remote-code` is required**. Tool calling is handled by the built-in `lfm2` tool
parser, and `<think>` reasoning by the `qwen3` reasoning parser.

> ℹ️ All commands below were verified end-to-end on **NVIDIA H100**. Other GPUs are expected to
> work but are listed only where validated.

## Supported Models

### Dense Chat Models

| Model | Parameters | Min NVIDIA GPU (BF16) | Context | Tools | HuggingFace |
|-------|-----------|------------------------|---------|-------|-------------|
| LFM2.5 350M | 350M | 1× (any) | 128K | ✓ | [LiquidAI/LFM2.5-350M](https://huggingface.co/LiquidAI/LFM2.5-350M) |
| LFM2.5 1.2B Instruct | 1.2B | 1× (8 GB+) | 128K | ✓ | [LiquidAI/LFM2.5-1.2B-Instruct](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct) |
| LFM2.5 1.2B Thinking | 1.2B | 1× (8 GB+) | 128K | ✓ (+reasoning) | [LiquidAI/LFM2.5-1.2B-Thinking](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking) |
| LFM2.5 1.2B JP | 1.2B | 1× (8 GB+) | 128K | – | [LiquidAI/LFM2.5-1.2B-JP](https://huggingface.co/LiquidAI/LFM2.5-1.2B-JP) |
| LFM2.5 1.2B JP (202606) | 1.2B | 1× (8 GB+) | 128K | ✓ | [LiquidAI/LFM2.5-1.2B-JP-202606](https://huggingface.co/LiquidAI/LFM2.5-1.2B-JP-202606) |
| LFM2.5 1.2B Base | 1.2B | 1× (8 GB+) | 128K | – (completions) | [LiquidAI/LFM2.5-1.2B-Base](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Base) |

### Mixture-of-Experts (MoE) Model

| Model | Total / Active Params | Min NVIDIA GPU (BF16) | Context | HuggingFace |
|-------|----------------------|------------------------|---------|-------------|
| LFM2.5 8B-A1B | 8B / ~1B active | 1× (24 GB+) | 128K | [LiquidAI/LFM2.5-8B-A1B](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B) |

The 8B-A1B keeps every expert resident in VRAM, so size the GPU for the full ~8B of weights even
though only ~1B is active per token. It supports `<think>` reasoning and tool calling.

### Vision-Language Models

| Model | Parameters | Min NVIDIA GPU (BF16) | Context | HuggingFace |
|-------|-----------|------------------------|---------|-------------|
| LFM2.5 VL 450M | 450M | 1× (any) | 128K | [LiquidAI/LFM2.5-VL-450M](https://huggingface.co/LiquidAI/LFM2.5-VL-450M) |
| LFM2.5 VL 1.6B | 1.6B | 1× (8 GB+) | 128K | [LiquidAI/LFM2.5-VL-1.6B](https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B) |

The VL models pair the LFM2 hybrid language backbone with a SigLIP2 vision encoder
(`Lfm2VlForConditionalGeneration`).

## Installing vLLM

LFM2.5 dense, MoE, and VL architectures all ship in vLLM **≥ 0.23.0** (stable).

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

Cap the context to fit a smaller GPU (models support up to 128K):

```bash
vllm serve LiquidAI/LFM2.5-1.2B-Instruct --max-model-len 32768
```

### 8B-A1B with Expert Parallelism (multi-GPU node)

```bash
vllm serve LiquidAI/LFM2.5-8B-A1B \
  --tensor-parallel-size 2 \
  --enable-expert-parallel
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
| LFM2.5 350M | 0.1 | 50 | – | 1.05 |
| LFM2.5 1.2B Instruct | 0.1 | 50 | – | 1.05 |
| LFM2.5 1.2B Thinking | 0.05 | 50 | – | 1.05 |
| LFM2.5 1.2B JP | 0.3 | – | 0.15 | 1.05 |
| LFM2.5 1.2B JP (202606) | 0.1 | 50 | – | 1.05 |
| LFM2.5 1.2B Base | 0.3 | – | 0.15 | 1.05 |
| LFM2.5 8B-A1B | 0.2 | 80 | – | 1.05 |
| LFM2.5 VL 450M / 1.6B | 0.1 | – | 0.15 | 1.05 |

> ⚠️ Do **not** bake these into `vllm serve` — they are per-request client defaults, not server
> flags. Capping `max_tokens` too low truncates the reasoning models' chain-of-thought.

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
    max_tokens=128,
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
            {"type": "image_url", "image_url": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"}},
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

## Server Flags Reference

| Flag | Description | When |
|------|-------------|------|
| `--reasoning-parser qwen3` | Split `<think>…</think>` into `reasoning_content` | 8B-A1B, 1.2B-Thinking |
| `--tool-call-parser lfm2` | Surface Pythonic tool calls as `tool_calls` | tool-capable models |
| `--enable-auto-tool-choice` | Auto-detect tool calls in output | with `--tool-call-parser` |
| `--enable-expert-parallel` | Split MoE experts across GPUs | 8B-A1B multi-GPU |
| `--max-model-len N` | Cap context (models support up to 128K) | small GPUs / fixed workload |
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

LFM2.5's small footprint means even a budget GPU is plenty. The 1.2B dense checkpoint was
validated end-to-end on this script across NVIDIA **T4, L4, A10G, L40S, A100 (40/80 GB), H100,
H200, and B200** (vLLM 0.23.0); the 8B-A1B MoE and the VL models were validated on **H100, H200,
and B200**. Size up to a 24 GB+ GPU (L4 / A10G or larger) for the 8B-A1B MoE, which keeps all
~8B of experts resident in VRAM.

## References

- [Liquid AI](https://www.liquid.ai/)
- [LiquidAI on HuggingFace](https://huggingface.co/LiquidAI)
- [LFM2.5-1.2B-Instruct model card](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct)
- [vLLM supported models](https://docs.vllm.ai/en/latest/models/supported_models.html)
