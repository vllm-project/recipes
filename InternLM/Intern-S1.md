# Intern-S1 Usage Guide

[Intern-S1](https://github.com/InternLM/Intern-S1) is a vision-language model that is developed by Shanghai AI Laboratory.
Latest vLLM already supports Intern-S1. You can install it using the following method:

## Installing vLLM

```bash
uv venv
source .venv/bin/activate
uv pip install -U vllm \
    --torch-backend=auto \
    --extra-index-url https://wheels.vllm.ai/nightly
```

## Launching Intern-S1 with vLLM

### Serving BF16 Model on 8xH800 GPUs (80GB × 8)

```bash
vllm serve internlm/Intern-S1 \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --reasoning-parser qwen3
```

### Serving FP8 Model on 4xH800 GPUs (80GB × 4)

```bash
vllm serve internlm/Intern-S1-FP8 \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --reasoning-parser qwen3
```

## Advanced Usage

### Switching Between Thinking and Non-Thinking Modes

Configure through

```python
extra_body={
    "chat_template_kwargs": {"enable_thinking": False}
}
```

Sample code

```python
from openai import OpenAI
client = OpenAI(api_key='YOUR_API_KEY', base_url='http://0.0.0.0:8000/v1')
model_name = client.models.list().data[0].id

response = client.chat.completions.create(
    model=model_name,
    messages=[{
        'role':
        'user',
        'content': [{
            'type': 'text',
            'text': '9.11 and 9.8, which is greater?',
        }],
    }],
    temperature=0.8,
    top_p=0.8,
    extra_body={
        "chat_template_kwargs": {"enable_thinking": False}
    }
)
print(response)
```

## Using Tips
If you encounter `ValueError: No available memory for the cache blocks.`, try adding the `--gpu-memory-utilization 0.95` flag to your `vllm serve` command.

## Additional Resources

- [Intern-S1](https://github.com/InternLM/Intern-S1)
- [vLLM Documentation](https://docs.vllm.ai/)
