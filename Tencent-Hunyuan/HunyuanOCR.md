# HunyuanOCR Usage Guide

## Introduction
[HunyuanOCR](https://huggingface.co/tencent/HunyuanOCR) stands as a leading end-to-end OCR expert VLM powered by Hunyuan's native multimodal architecture. In this guide, we demonstrate how to set up HunyuanOCR for online OCR serving with OpenAI compatible API server.

## Installing vLLM

```bash
uv venv
source .venv/bin/activate
# Until v0.11.3 release, you need to install vLLM from nightly build
uv pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly
```

## Deploying HunyuanOCR

```bash
vllm serve tencent/HunyuanOCR \
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

messages = [
    {"role": "system", "content": ""},
    {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/chat-ui/tools-dark.png"
                }
            },
            {
                "type": "text",
                "text": (
                    "Extract all information from the main body of the document image "
                    "and represent it in markdown format, ignoring headers and footers."
                    "Tables should be expressed in HTML format, formulas in the document "
                    "should be represented using LaTeX format, and the parsing should be "
                    "organized according to the reading order."
                )
            }
        ]
    }
]

response = client.chat.completions.create(
    model="tencent/HunyuanOCR",
    messages=messages,
    temperature=0.0,
    extra_body={
        "top_k": 1,
        "repetition_penalty": 1.0
    },
)
print(f"Generated text: {response.choices[0].message.content}")
```

## Configuration Tips
- Use greedy sampling (i.e., temperature=0.0) or sampling with low temperature for the optimal OCR performance.
- Unlike multi-turn chat use cases, we do not expect OCR tasks to benefit significantly from prefix caching or image reuse, therefore it's recommended to turn off these features to avoid unnecessary hashing and caching.
- Depending on your hardware capability, adjust `max_num_batched_tokens` for better throughput performance.
- Check out the official [HunyuanOCR documentation](https://huggingface.co/tencent/HunyuanOCR#%F0%9F%92%AC-application-oriented-prompts) for more application-oriented prompts for various document parsing tasks.



### AMD GPU Support

Please follow the steps here to install and run HunyuanOCR models on AMD MI300X GPU.
### Step 1: Prepare Docker Environment
Pull the latest vllm docker:
```shell
docker pull rocm/vllm-dev:nightly
```
Launch the ROCm vLLM docker: 
```shell
docker run -it --ipc=host --network=host --privileged --cap-add=CAP_SYS_ADMIN --device=/dev/kfd --device=/dev/dri --device=/dev/mem --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v $(pwd):/work -e SHELL=/bin/bash  --name HunyuanOCR rocm/vllm-dev:nightly 
```
### Step 2: Log in to Hugging Face
Huggingface login
```shell
huggingface-cli login
```

### Step 3: Start the vLLM server

Run the vllm online serving
Sample Command
```shell

SAFETENSORS_FAST_GPU=1 VLLM_USE_TRITON_FLASH_ATTN=0 vllm serve tencent/HunyuanOCR --tensor-parallel-size 2 --gpu-memory-utilization 0.9 --disable-log-requests --no-enable-prefix-caching --trust-remote-code 

SAFETENSORS_FAST_GPU=1 \
VLLM_USE_TRITON_FLASH_ATTN=0 vllm serve tencent/HunyuanOCR \
    --no-enable-prefix-caching \
    --mm-processor-cache-gb 0 \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code 
	
```


### Step 4: Run Benchmark
Open a new terminal and run the following command to execute the benchmark script inside the container.
```shell
docker exec -it HunyuanOCR vllm bench serve \
  --model "tencent/HunyuanOCR" \
  --dataset-name random \
  --random-input-len 8000 \
  --random-output-len 1000 \
  --request-rate 10000 \
  --num-prompts 16 \
  --ignore-eos
```

