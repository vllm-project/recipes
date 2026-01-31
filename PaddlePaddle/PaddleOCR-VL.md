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

!!! tip
    Use separate virtual environments for `vllm` and `paddlepaddle` to prevent dependency conflicts. If you encounter the error `The model PaddleOCR-VL-0.9B does not exist.`, add `--served-model-name PaddleOCR-VL-0.9B` to your vLLM launch command.

### Install [PaddlePaddle](https://www.paddlepaddle.org.cn/install/quick) and [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
```shell
uv pip install paddlepaddle-gpu==3.2.1 --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu126/
uv pip install -U "paddleocr[doc-parser]"
uv pip install safetensors
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

for i, res in enumerate(output):
    res.save_to_json(save_path=f"output_{i}.json")
    res.save_to_markdown(save_path=f"output_{i}.md")
```

## Configuration Tips
- Unlike multi-turn chat use cases, we do not expect OCR tasks to benefit significantly from prefix caching or image reuse, therefore it's recommended to turn off these features to avoid unnecessary hashing and caching.
- Depending on your hardware capability, adjust `max_num_batched_tokens` for better throughput performance.
- Check out the official [PaddleOCR-VL documentation](https://github.com/PaddlePaddle/PaddleOCR) for more details and examples of using the model for various document parsing tasks.




## AMD GPU Support 
Recommended approaches by hardware type are:

MI300X/MI325X/MI355X

Please follow the steps here to install and run PaddleOCR-VL models on AMD  GPU.
### Step 1: Prepare Docker Environment
Pull the latest vllm docker:
```shell
docker pull vllm/vllm-openai-rocm:v0.14.1
```
Launch the ROCm vLLM docker: 
```shell

docker run -d -it --entrypoint /bin/bash --ipc=host --network=host --privileged --cap-add=CAP_SYS_ADMIN --device=/dev/kfd --device=/dev/dri --device=/dev/mem --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v /:/work   -v ~/.cache/huggingface:/root/.cache/huggingface -p 8000:8000 --name PaddleOCR-VL vllm/vllm-openai-rocm:v0.14.1

```
### Step 2: Log in to Hugging Face
Huggingface login
```shell
hf auth login
```

### Step 3: Start the vLLM server

Run the vllm online serving
Sample Command
```shell

SAFETENSORS_FAST_GPU=1 \
VLLM_USE_V1=1 \
VLLM_USE_TRITON_FLASH_ATTN=0 vllm serve PaddlePaddle/PaddleOCR-VL \
  --max-num-batched-tokens 16384 \
  --no-enable-prefix-caching \
  --mm-processor-cache-gb 0 \
  --trust-remote-code

```


### Step 4: Run Benchmark
Open a new terminal and run the following command to execute the benchmark script inside the container.
```shell
docker exec -it PaddleOCR-VL vllm bench serve \
  --model "PaddlePaddle/PaddleOCR-VL" \
  --dataset-name random \
  --random-input-len 8192 \
  --random-output-len 1024 \
  --request-rate 10000 \
  --num-prompts 16 \
  --ignore-eos \
  --trust-remote-code 
```

