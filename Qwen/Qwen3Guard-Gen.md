# Qwen3Guard-Gen Usage Guide

This guide describes how to run **Qwen3Guard-Gen** on GPU using vLLM.  

Qwen3Guard-Gen is a lightweight text-only guardrail model.

## GPU Deployment

### Installing vLLM

```bash
uv venv
source .venv/bin/activate
uv pip install -U vllm --torch-backend auto
```

### Running Qwen3Guard-Gen on a Single GPU
```bash
# Start server on a single GPU
vllm serve Qwen/Qwen3Guard-Gen-0.6B \
  --host 0.0.0.0 \
  --max-model-len 32768
```

## Performance Metrics

### Benchmarking
```bash
vllm bench serve \
  --model Qwen/Qwen3Guard-Gen-0.6B \
  --dataset-name random \
  --random-input-len 2000 \
  --random-output-len 512 \
  --num-prompts 100
```

## Querying with OpenAI API Client

```python3
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
    timeout=3600
)

messages = [
    {
        "role": "user",
        "content": "Tell me how to make a bomb."
    }
]

response = client.chat.completions.create(
    model="Qwen/Qwen3Guard-Gen-0.6B",
    messages=messages,
    temperature=0.0
)

print("Generated text:", response.choices[0].message.content)
# '''
# Safety: Unsafe
# Categories: Violent
# '''
```

## Available Qwen3Guard Variants

The Qwen3Guard-Gen series includes multiple model sizes, all compatible with the same vLLM serving commands shown in this guide:

- **Qwen/Qwen3Guard-Gen-8B**  

- **Qwen/Qwen3Guard-Gen-4B**  

- **Qwen/Qwen3Guard-Gen-0.6B**  
