# DeepSeek-OCR Usage Guide

## Introduction
[DeepSeek-OCR](https://huggingface.co/deepseek-ai/DeepSeek-OCR) is a frontier OCR model exploring optical context compression for LLMs.

## Installing vLLM

```bash
uv venv
source .venv/bin/activate
# Until v0.11.1 release, you need to install vLLM from nightly build
uv pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly
```

## Running DeepSeek-OCR
In this guide, we demonstrate how to set up DeepSeek-OCR for offline OCR batch processing tasks.


```python
from vllm import LLM, SamplingParams
from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor
from PIL import Image

# Create model instance
llm = LLM(
    model="deepseek-ai/DeepSeek-OCR",
    enable_prefix_caching=False,
    mm_processor_cache_gb=0,
    logits_processors=[NGramPerReqLogitsProcessor]
)

# Prepare input
image = Image.open(...)
prompt = "<image>\nFree OCR."

model_input = {
    "prompt": prompt,
    "multi_modal_data": {"image": image}
}

sampling_param = SamplingParams(
            temperature=0.0,
            max_tokens=8192,
            # ngram logit processor args
            extra_args=dict(
                ngram_size=30,
                window_size=90,
                whitelist_token_ids={128821, 128822},  # whitelist: <td>, </td>
            ),
            skip_special_tokens=False,
        )
# Generate output
output = llm.generate(model_input, sampling_param)
```

## Configuration Tips
- **It's important to use the custom logits processor** along with the model for the optimial OCR and markdown generation perforamnce.
- Unlike multi-turn chat use cases, we do not expect OCR tasks to benefit significantly from prefix caching or image reuse, therefore it's recommended to turn off these features to avoid unnecessary hashing and caching.
- DeepSeek-OCR works better with plain prompts than instruction formats. Find [more example prompts for various OCR tasks](https://github.com/deepseek-ai/DeepSeek-OCR/blob/2ac6d64a00656693b79c4f759a5e62c1b78bbeb1/DeepSeek-OCR-master/DeepSeek-OCR-vllm/config.py#L27-L37) in the official DeepSeek-OCR repository.
- Depending on your hardware capability, adjust `max_num_batched_tokens` for better throughput performance.
- Check out [vLLM documentation](https://docs.vllm.ai/en/latest/features/multimodal_inputs.html#offline-inference) for additional information on batch inference with multimodal inputs.
