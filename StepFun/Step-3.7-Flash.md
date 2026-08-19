# Step-3.7-Flash Guide

[Step-3.7-Flash](https://huggingface.co/stepfun-ai/Step-3.7-Flash) is a 198B-parameter sparse Mixture-of-Experts (MoE) vision-language model that combines a 196B-parameter language backbone with a 1.8B-parameter vision encoder for native image understanding. Engineered for high-frequency production workloads, it activates approximately 11B parameters per token and delivers a throughput of up to 400 tokens per second. Step 3.7 Flash supports a 256k context window and offers three selectable reasoning levels (low, medium, and high) so developers can easily balance speed, cost, and cognitive depth.
**Key highlights**:

- **Multimodal Understanding**: Native vision encoder for image understanding, supporting single and multi-image inputs alongside text.
- **Hybrid Attention Architecture**: Interleaves Sliding Window Attention (SWA) and Global Attention (GA) with a 3:1 ratio and an aggressive 512-token window, ensuring consistent performance across massive datasets while significantly reducing computational overhead.
- **Sparse Mixture-of-Experts**: Only 11B active parameters out of 198B total parameters.
- **Multi-Layer Multi-Token Prediction (MTP)**: Equipped with 3-way Multi-Token Prediction (MTP-3) for complex, multi-step reasoning chains with immediate responsiveness.

## Installing vLLM

```bash
uv venv
source .venv/bin/activate
uv pip install vllm --torch-backend auto
```

## Serving with vLLM

### Official Provided Formats

`Step-3.7-Flash` provides three precision options, You can choose the appropriate model based on your needs.

- [stepfun-ai/Step-3.7-Flash](https://huggingface.co/stepfun-ai/Step-3.7-Flash)
- [stepfun-ai/Step-3.7-Flash-FP8](https://huggingface.co/stepfun-ai/Step-3.7-Flash-FP8)
- [stepfun-ai/Step-3.7-Flash-NVFP4](https://huggingface.co/stepfun-ai/Step-3.7-Flash-NVFP4)

### Deployment

#### For FP8 model

```bash
vllm serve stepfun-ai/Step-3.7-Flash-FP8 \
  --served-model-name step3p7-flash \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --disable-cascade-attn \
  --reasoning-parser step3p5 \
  --enable-auto-tool-choice \
  --tool-call-parser step3p5 \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 3}' \
  --trust-remote-code
```

#### For BF16 model

```bash
vllm serve stepfun-ai/Step-3.7-Flash \
  --served-model-name step3p7-flash-bf16 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --disable-cascade-attn \
  --reasoning-parser step3p5 \
  --enable-auto-tool-choice \
  --tool-call-parser step3p5 \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 3}' \
  --trust-remote-code
```

#### For NVFP4 model

Compared to standard precisions, running the FP4 quantized version requires modelopt activation and FP8 KV Cache alignment.

```bash
vllm serve stepfun-ai/Step-3.7-Flash-NVFP4  \
  --served-model-name step3p7 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.9 \
  --enable-expert-parallel \
  --trust-remote-code \
  --quantization modelopt \
  --kv-cache-dtype fp8 \
  --reasoning-parser step3p5 \
  --enable-auto-tool-choice \
  --tool-call-parser step3p5 \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 3}' \
  --async-scheduling
```
