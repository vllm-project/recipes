# GLM-5 and GLM-5.1 Series Usage

GLM-5 is a significantly scaled-up language model (744B parameters, 28.5T tokens) with novel asynchronous RL infrastructure that achieves best-in-class open-source performance on reasoning, coding, and agentic tasks, rivaling frontier models. GLM is available in 2 precision formats: [zai-org/GLM-5](https://huggingface.co/zai-org/GLM-5) and [zai-org/GLM-5-FP8](https://huggingface.co/zai-org/GLM-5-FP8), with [GLM-5.1](https://huggingface.co/zai-org/GLM-5.1) as a refreshed version of GLM-5.

This guide describes how to run GLM-5 or GLM-5.1 with native FP8.

## Dependencies

### Using Docker

```bash
docker run --gpus all \
  -p 8000:8000 \
  --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:glm51 zai-org/GLM-5.1-FP8 \
    --tensor-parallel-size 8 \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --enable-auto-tool-choice \
    --chat-template-content-format=string \
    --served-model-name glm-5.1-fp8
```

Please use the `vllm/vllm-openai:glm51-cu130` Docker image if your CUDA version is 13 or higher.
>Note: When encounter Tool Call Parse issue with MTP enabled, please turn to vllm main branch to serve GLM-5.1.

### Installing vLLM from source

```bash
uv venv
source .venv/bin/activate
uv pip install "vllm==0.19.0" --torch-backend=auto
uv pip install "transformers>=5.4.0"
```

- For FP8 model, you must install DeepGEMM using [install_deepgemm.sh](https://github.com/vllm-project/vllm/blob/v0.16.0rc0/tools/install_deepgemm.sh).

!!! attention
    Instead of nightly releases, please use the 0.19.0 stable release of vLLM for the best model performance.


## Model Deployment

### Serving FP8 Model on 8xH200 (or H20) GPUs (141GB × 8)


```bash
vllm serve zai-org/GLM-5.1-FP8 \
     --tensor-parallel-size 8 \
     --speculative-config.method mtp \
     --speculative-config.num_speculative_tokens 3 \
     --tool-call-parser glm47 \
     --reasoning-parser glm45 \
     --enable-auto-tool-choice \
    --chat-template-content-format=string \
     --served-model-name glm-5.1-fp8
```

- When using vLLM, **thinking mode is enabled by default when sending requests**. If you want to disable the thinking switch, you need to add the `"chat_template_kwargs": {"enable_thinking": false}` parameter.
- Support tool calling by default. Please use OpenAI-style tool description format for calls.

### OpenAI Client Example

First, install the OpenAI Python client:

```bash
uv pip install -U openai
```

You can use the OpenAI client as follows to  verify the think mode.

```python
from openai import OpenAI

# If running vLLM locally with its default OpenAI-compatible port:
#   http://localhost:8000/v1
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Summarize GLM-5 in one sentence."},
]

# Thinking ON (default if you omit chat_template_kwargs)
resp_on = client.chat.completions.create(
    model="glm-5.1-fp8",
    messages=messages,
    temperature=1,
    max_tokens=4096,
)
print("thinking=on, think content:\n", resp_on.choices[0].message.reasoning)

# Thinking OFF
resp_off = client.chat.completions.create(
    model="glm-5.1-fp8",
    messages=messages,
    temperature=1,
    max_tokens=4096,
    extra_body={
        "chat_template_kwargs": {
            "enable_thinking": False
        }
    },
)
# The content of reasoning should be None.
print("thinking=off:\n", resp_off.choices[0].message.reasoning)
```

### cURL Usage

- Thinking ON (default):

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d ' {
    "model": "glm-5.1-fp8",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Summarize GLM-5 in one sentence."}
    ],
    "temperature": 1,
    "max_tokens": 4096
  } '
```

- Thinking OFF:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d ' {
    "model": "glm-5.1-fp8",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Summarize GLM-5 in one sentence."}
    ],
    "temperature": 1,
    "max_tokens": 4096,
    "chat_template_kwargs": {"enable_thinking": false}
  } '
```

## Benchmarking

For benchmarking, disable prefix caching by adding `--no-enable-prefix-caching` to the server command.

### FP8 Benchmark

- The following uses H200*8 as an example to demonstrate how to run the benchmark.

```bash
# Prompt-heavy benchmark (8k/1k)
vllm bench serve \
  --model zai-org/GLM-5.1-FP8 \
  --dataset-name random \
  --random-input 8000 \
  --random-output 1024 \
  --request-rate 10 \
  --num-prompts 32 \
  --ignore-eos
```

If successful, you will see the following output.

In practice, the actual generation speed is usually higher than what is shown here, because the model supports MTP. In pure performance benchmarks, the MTP acceptance rate is often relatively low, so the measured throughput may underestimate the model’s real-world speed.

```shell
============ Serving Benchmark Result ============
Successful requests:                     32        
Failed requests:                         0         
Request rate configured (RPS):           10.00     
Benchmark duration (s):                  62.23     
Total input tokens:                      256000    
Total generated tokens:                  32768     
Request throughput (req/s):              0.51      
Output token throughput (tok/s):         526.57    
Peak output token throughput (tok/s):    800.00    
Peak concurrent requests:                32.00     
Total token throughput (tok/s):          4640.43   
---------------Time to First Token----------------
Mean TTFT (ms):                          13395.44  
Median TTFT (ms):                        14494.06  
P99 TTFT (ms):                           22952.29  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          35.39     
Median TPOT (ms):                        34.58     
P99 TPOT (ms):                           49.72     
---------------Inter-token Latency----------------
Mean ITL (ms):                           57.98     
Median ITL (ms):                         41.88     
P99 ITL (ms):                            578.32    
---------------Speculative Decoding---------------
Acceptance rate (%):                     21.33     
Acceptance length:                       1.64      
Drafts:                                  19982     
Draft tokens:                            59946     
Accepted tokens:                         12784     
Per-position acceptance (%):
  Position 0:                            36.59     
  Position 1:                            20.39     
  Position 2:                            7.00      
==================================================
```
