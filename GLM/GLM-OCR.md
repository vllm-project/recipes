# GLM-OCR Usage Guide

This guide describe how to run GLM-OCR with MTP support in vllm.

## Installing vllm

```bash
uv venv
source .venv/bin/activate
uv pip install -U vllm --torch-backend auto
```

```bash
# install the nightly build of vLLM for GLM-4.7
uv pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly

# install transformers from source
uv pip install git+https://github.com/huggingface/transformers.git
```

## Running GLM-OCR with MTP

GLM-OCR model include built-in Multi-Token Prediction (MTP) layers that can be used for speculative decoding to accelerate generation throughput.

Add the `--speculative-config` flags to server command to enable MTP speculative decoding:

```bash
vllm serve zai-org/GLM-OCR \
     --speculative-config.method mtp \
     --speculative-config.num_speculative_tokens 1
```

## CURL Usage

```bash
curl -s http://localhost:8000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{
          "model": "zai-org/GLM-OCR",
          "messages": [
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
                              "text": "Text Recognition:"
                         }
                    ]
               }
          ],
          "max_tokens": 2048,
          "temperature": 0.0
     }'
```

## OpenAI SDK Client Usage

```python
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
                                        "text": "Text Recognition:"
                              }
                    ]
          }
]

start = time.time()
response = client.chat.completions.create(
          model="zai-org/GLM-OCR",
          messages=messages,
          max_tokens=2048,
          temperature=0.0
)
print(f"Response costs: {time.time() - start:.2f}s")
print("Generated text:")
print(response.choices[0].message.content)
```
