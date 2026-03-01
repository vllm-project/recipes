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

### Docker
```bash
docker pull vllm/vllm-openai:v0.14.1-cu130
```

## Running TranslateGemma

The following configuration has been verified for the 4B/27B model

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
          --optimization-level 0 \
          --host 0.0.0.0 \
          --port 8000
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

