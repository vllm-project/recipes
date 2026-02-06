# DeepSeek-OCR Usage Guide

## Introduction
[DeepSeek-OCR](https://huggingface.co/deepseek-ai/DeepSeek-OCR) is a frontier OCR model exploring optical context compression for LLMs.

## Installing vLLM

```bash
uv venv
source .venv/bin/activate
# Until v0.11.1 release, you need to install vLLM from nightly build
uv pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly --extra-index-url https://download.pytorch.org/whl/cu129 --index-strategy unsafe-best-match
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


## AMD GPU Support
Recommended approaches by hardware type are:

MI300X/MI325X/MI355X 

Please follow the steps here to install and run DeepSeek-OCR models on AMD MI300X/MI325X/MI355X GPU.

### Step 1: Installing vLLM (AMD ROCm Backend: MI300X, MI325X, MI355X) 
 > Note: The vLLM wheel for ROCm requires Python 3.12, ROCm 7.0, and glibc >= 2.35. If your environment does not meet these requirements, please use the Docker-based setup as described in the [documentation](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/#pre-built-images).  
 ```bash 
 uv venv 
 source .venv/bin/activate 
 uv pip install vllm --extra-index-url https://wheels.vllm.ai/rocm/
 ```


### Step 2: Start the vLLM server

Run the vllm online serving


```bash
SAFETENSORS_FAST_GPU=1 \
VLLM_USE_TRITON_FLASH_ATTN=0 \
VLLM_ROCM_USE_AITER=1 \
vllm serve deepseek-ai/DeepSeek-OCR \
  --logits_processors vllm.model_executor.models.deepseek_ocr:NGramPerReqLogitsProcessor \
  --no-enable-prefix-caching \
  --mm-processor-cache-gb 0
```


### Step 3: Run Benchmark
Open a new terminal and run the following command to execute the benchmark script.
```bash
vllm bench serve \
  --model "deepseek-ai/DeepSeek-OCR" \
  --dataset-name random \
  --random-input-len 4096 \
  --random-output-len 512 \
  --request-rate 1 \
  --num-prompts 4 \
  --ignore-eos
```

