# GLM-5 Usage

This guide describes how to run GLM-5 with native FP8.

## Dependencies

Using Docker with:

```bash
docker pull vllm/vllm-openai:nightly
pip install git+https://github.com/huggingface/transformers.git
```

Build from source:

```bash
uv venv
source .venv/bin/activate

uv pip install -U vllm --pre --index-url https://pypi.org/simple --extra-index-url https://wheels.vllm.ai/nightly

uv pip install git+https://github.com/huggingface/transformers.git
```

## Running with FP8 and MTP

```bash
vllm serve zai-org/GLM-5-FP8 \
     --tensor-parallel-size 8 \
     --speculative-config.method mtp \
     --speculative-config.num_speculative_tokens 1 \
     --tool-call-parser glm47 \
     --reasoning-parser glm45 \
     --enable-auto-tool-choice \
     --served-model-name glm-5-fp8
```

- When using `vLLM`, thinking mode is enabled by default when sending requests. If you want to disable the thinking switch, you need to add the `"chat_template_kwargs": {"enable_thinking": false}` parameter.
- Support tool calling by default. Please use OpenAI-style tool description format for calls.

## Client Usage

The vLLM server exposes an OpenAI-compatible API.

### Python (OpenAI SDK)

Install:

```bash
pip install -U openai
```

Example:

```python
from openai import OpenAI

# If running vLLM locally with its default OpenAI-compatible port:
#   http://localhost:8000/v1
client = OpenAI(
    base_url="http://localhost:8000/v1",
)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Summarize GLM-5 in one sentence."},
]

# Thinking ON (default if you omit chat_template_kwargs)
resp_on = client.chat.completions.create(
    model="glm-5-fp8",
    messages=messages,
    temperature=1,
    max_tokens=4096,
)
print("thinking=on:\n", resp_on.choices[0].message.content)

# Thinking OFF
resp_off = client.chat.completions.create(
    model="glm-5-fp8",
    messages=messages,
    temperature=1,
    max_tokens=4096,
    extra_body={
        "chat_template_kwargs": {
            "enable_thinking": False
        }
    },
)
print("thinking=off:\n", resp_off.choices[0].message.content)
```

## cURL Usage

### Chat Completions

Thinking ON (default):

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d ' {
    "model": "glm-5-fp8",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Summarize GLM-5 in one sentence."}
    ],
    "temperature": 1,
    "max_tokens": 4096
  } '
```

Thinking OFF:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d ' {
    "model": "glm-5-fp8",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Summarize GLM-5 in one sentence."}
    ],
    "temperature": 1,
    "max_tokens": 4096,
    "chat_template_kwargs": {"enable_thinking": false}
  } '
```

## Benchmarking

For benchmarking, disable prefix caching by adding `--no-enable-prefix-caching` to the server command.

### FP8 Benchmark

```bash
# Prompt-heavy benchmark (8k/1k)
vllm bench serve \
  --model zai-org/GLM-5-FP8 \
  --dataset-name random \
  --random-input-len 8000 \
  --random-output-len 1000 \
  --request-rate 10000 \
  --num-prompts 16 \
  --ignore-eos
```
