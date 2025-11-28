# PaddleOCR-VL Usage Guide

## Introduction
[PaddleOCR-VL](https://huggingface.co/PaddlePaddle/PaddleOCR-VL) is a SOTA and resource-efficient model tailored for document parsing. Its core component is PaddleOCR-VL-0.9B, a compact yet powerful vision-language model (VLM) that integrates a NaViT-style dynamic resolution visual encoder with the ERNIE-4.5-0.3B language model to enable accurate element recognition.

## Installing vLLM

```bash
uv venv
source .venv/bin/activate
# Until v0.11.1 release, you need to install vLLM from nightly build
uv pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly --extra-index-url https://download.pytorch.org/whl/cu129 --index-strategy unsafe-best-match
```

## Deploying PaddleOCR-VL

```bash
vllm serve PaddlePaddle/PaddleOCR-VL \
    --trust-remote-code \
    --max-num-batched-tokens 16384 \
    --no-enable-prefix-caching \
    --mm-processor-cache-gb 0
```

## Querying with OpenAI API Client
```python3
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
    timeout=3600
)

# Task-specific base prompts
TASKS = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
    "chart": "Chart Recognition:",
}

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
                "text": TASKS["ocr"]
            }
        ]
    }
]

response = client.chat.completions.create(
    model="PaddlePaddle/PaddleOCR-VL",
    messages=messages,
    temperature=0.0,
)
print(f"Generated text: {response.choices[0].message.content}")
```
## Offline inference using vLLM combined with PP-DocLayoutV2
In the examples above, we have demonstrated the inference of PaddleOCR-VL using vLLM. Typically, we also need to integrate the PP-DocLayoutV2 model to fully unleash the capabilities of the PaddleOCR-VL model, making it more aligned with the examples provided by PaddlePaddle officially.
### Install [PaddlePaddle](https://www.paddlepaddle.org.cn/install/quick) and [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
```shell
python -m pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
python -m pip install -U "paddleocr[doc-parser]"
python -m pip install https://paddle-whl.bj.bcebos.com/nightly/cu126/safetensors/safetensors-0.6.2.dev0-cp38-abi3-linux_x86_64.whl
```

Using vLLM as the backend, combined with PP-DocLayoutV2 for offline inference.

```python
from paddleocr import PaddleOCRVL

doclayout_model_path = "/path/to/your/PP-DocLayoutV2/"

pipeline = PaddleOCRVL(vl_rec_backend="vllm-server", 
                       vl_rec_server_url="http://localhost:8000/v1", 
                       layout_detection_model_name="PP-DocLayoutV2",  
                       layout_detection_model_dir=doclayout_model_path)

output = pipeline.predict("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/paddleocr_vl_demo.png")

for res in output:
    res.save_to_json(save_path="output")
    res.save_to_markdown(save_path="output")
```

## Configuration Tips
- Unlike multi-turn chat use cases, we do not expect OCR tasks to benefit significantly from prefix caching or image reuse, therefore it's recommended to turn off these features to avoid unnecessary hashing and caching.
- Depending on your hardware capability, adjust `max_num_batched_tokens` for better throughput performance.
- Check out the official [PaddleOCR-VL documentation](https://github.com/PaddlePaddle/PaddleOCR) for more details and examples of using the model for various document parsing tasks.
