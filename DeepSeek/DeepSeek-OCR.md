# DeepSeek-OCR Usage Guide

## Introduction
[DeepSeek-OCR](https://huggingface.co/deepseek-ai/DeepSeek-OCR) is a frontier OCR model exploring optical context compression for LLMs.

## Installing vLLM

```bash
uv venv
source .venv/bin/activate
uv pip install -U vllm --torch-backend autoh
```

## Running DeepSeek-OCR
### Offline OCR tasks
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

# Prepare batched input with your image file
image_1 = Image.open("path/to/your/image_1.png").convert("RGB")
image_2 = Image.open("path/to/your/image_2.png").convert("RGB")
prompt = "<image>\nFree OCR."

model_input = [
    {
        "prompt": prompt,
        "multi_modal_data": {"image": image_1}
    },
    {
        "prompt": prompt,
        "multi_modal_data": {"image": image_2}
    }
]

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
model_outputs = llm.generate(model_input, sampling_param)

# Print output
for output in model_outputs:
    print(output.outputs[0].text)
```

### Online OCR serving
In this guide, we demonstrate how to set up DeepSeek-OCR for online OCR serving with OpenAI compatible API server.

```bash
vllm serve deepseek-ai/DeepSeek-OCR --logits_processors vllm.model_executor.models.deepseek_ocr:NGramPerReqLogitsProcessor --no-enable-prefix-caching --mm-processor-cache-gb 0
```

```python3
import time
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
    timeout=3600
)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://ofasys-multimodal-wlcb-3-toshanghai.oss-accelerate.aliyuncs.com/wpf272043/keepme/image/receipt.png"
                }
            },
            {
                "type": "text",
                "text": "Free OCR."
            }
        ]
    }
]

start = time.time()
response = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-OCR",
    messages=messages,
    max_tokens=2048,
    temperature=0.0,
    extra_body={
        "skip_special_tokens": False,
        # args used to control custom logits processor
        "vllm_xargs": {
            "ngram_size": 30,
            "window_size": 90,
            # whitelist: <td>, </td>
            "whitelist_token_ids": [128821, 128822],
        },
    },
)
print(f"Response costs: {time.time() - start:.2f}s")
print(f"Generated text: {response.choices[0].message.content}")
```

## Configuration Tips
- **It's important to use the custom logits processor** along with the model for the optimal OCR and markdown generation performance.
- Unlike multi-turn chat use cases, we do not expect OCR tasks to benefit significantly from prefix caching or image reuse, therefore it's recommended to turn off these features to avoid unnecessary hashing and caching.
- DeepSeek-OCR works better with plain prompts than instruction formats. Find [more example prompts for various OCR tasks](https://github.com/deepseek-ai/DeepSeek-OCR/blob/2ac6d64a00656693b79c4f759a5e62c1b78bbeb1/DeepSeek-OCR-master/DeepSeek-OCR-vllm/config.py#L27-L37) in the official DeepSeek-OCR repository.
- Depending on your hardware capability, adjust `max_num_batched_tokens` for better throughput performance.
- Check out [vLLM documentation](https://docs.vllm.ai/en/latest/features/multimodal_inputs.html#offline-inference) for additional information on batch inference with multimodal inputs.
