# TranslateGemma Usage Guide

[TranslateGemma](https://huggingface.co/collections/google/translategemma) is a family of lightweight, state-of-the-art open translation models from Google, based on the Gemma 3 family of models.
TranslateGemma models are designed to handle translation tasks across 55 languages. Their relatively small size makes it possible to deploy them in environments with limited resources such as laptops, desktops or your own cloud infrastructure, democratizing access to state of the art translation models and helping foster innovation for everyone.

## Models
Original Models:

- [google/translategemma-27b-it](https://huggingface.co/google/translategemma-27b-it)
- [google/translategemma-4b-it](https://huggingface.co/google/translategemma-4b-it)

Optimized vLLM Models:

- [Infomaniak-AI/vllm-translategemma-27b-it](https://huggingface.co/Infomaniak-AI/vllm-translategemma-27b-it)
- [Infomaniak-AI/vllm-translategemma-4b-it](https://huggingface.co/Infomaniak-AI/vllm-translategemma-4b-it)

### Why use vLLM-optimized models?

The original Google models have compatibility issues with standard inference engines like vLLM. The optimized versions from Infomaniak-AI (see [detailed changes](https://huggingface.co/Infomaniak-AI/vllm-translategemma-27b-it#changes-from-original-model)) resolve these issues:

- **vLLM Compatibility**: The original models require custom JSON parameters (`source_lang_code` and `target_lang_code`) that are not supported by the standard vLLM/OpenAI chat interface. The optimized version uses string delimiters instead.
- **RoPE Simplification**: The original models use a complex RoPE configuration for sliding attention. The optimized versions use a standard linear RoPE format (`factor: 8.0`) that vLLM can correctly parse.
- **EOS Token Fix**: Corrects the EOS token from `<end_of_turn>` to `<eos>` to ensure proper sequence termination.

## Installing vLLM

### pip (AMD ROCm: MI300X, MI325X, MI350X, MI355X)

 **Note:** The vLLM nightly wheel for ROCm requires Python 3.12, ROCm 7.2.1, glibc ≥ 2.35 (Ubuntu 22.04+)

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install vllm --pre \
--extra-index-url https://wheels.vllm.ai/rocm/nightly/rocm721 --upgrade
```
### Docker (NVIDIA)
```bash
docker pull vllm/vllm-openai:v0.14.1-cu130
```
### Docker (AMD ROCm: MI300X, MI325X, MI350X, MI355X)
```bash
docker pull vllm/vllm-openai-rocm:latest
```
## Running TranslateGemma

The following configuration has been verified for the 4B/27B model

### NVIDIA
```bash
docker run -itd --name google-translategemma-27b-it \
    --ipc=host \
    --network host \
    --shm-size 16G \
    --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    vllm/vllm-openai:v0.14.1-cu130 \
        Infomaniak-AI/vllm-translategemma-27b-it \
        --served-model-name translategemma-27b-it \
        --gpu-memory-utilization 0.8 \
        --host 0.0.0.0 \
        --port 8000
```
### AMD
```bash
docker run -d -it \
    --ipc=host \
    --network=host \
    --privileged \
    --cap-add=CAP_SYS_ADMIN \
    --device=/dev/kfd \
    --device=/dev/dri \
    --device=/dev/mem \
    --group-add video \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --shm-size 32G \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    vllm/vllm-openai-rocm:latest \
        Infomaniak-AI/vllm-translategemma-27b-it \
        --served-model-name translategemma-27b-it \
        --gpu-memory-utilization 0.8 \
		--trust-remote-code
```
## Consume the OpenAI API Compatible Server

Tips:

- **Prompt Delimiters**: Language metadata is encoded directly into the content string using specific delimiters: `<<<source>>>{src_lang}<<<target>>>{tgt_lang}<<<text>>>{text}`
- **Language Codes**: Supports ISO 639-1 Alpha-2 codes (e.g., `en`, `zh`) and regional variants (e.g., `en_US`, `zh_CN`).
- **Context Limit**: The model is optimized for a context window of approximately **2K tokens**.

### Curl Example (Translation)
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "translategemma-27b-it",
    "messages": [
        {
          "role": "user",
          "content": "<<<source>>>en<<<target>>>zh<<<text>>>We distribute two models for language identification, which can recognize 176 languages."
        }
      ]
    }'
```

