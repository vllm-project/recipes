# Intern-S1 Usage Guide

[Intern-S1](https://github.com/InternLM/Intern-S1) Intern-S1 is a vision-language model that is developed by Shanghai AI Laboratory.
Latest vLLM already supports Intern-S1. You can install vLLM using the following method:

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

**BF16 Model**

```bash
vllm serve internlm/Intern-S1 \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --reasoning-parser qwen3 \
  --tool-call-parser internlm
```

### Serving FP8 Model on 4xH800 GPUs (80GB × 4)

**FP8 Model**

```bash
vllm serve internlm/Intern-S1-FP8 \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --reasoning-parser qwen3 \
  --tool-call-parser internlm
```

## Additional Resources

- [Intern-S1](https://github.com/InternLM/Intern-S1)
- [vLLM Documentation](https://docs.vllm.ai/)
