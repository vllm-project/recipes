# GLM-5 Usage

GLM-5 is a significantly scaled-up language model (744B parameters, 28.5T tokens) with novel asynchronous RL infrastructure that achieves best-in-class open-source performance on reasoning, coding, and agentic tasks, rivaling frontier models. GLM is available in 2 precision formats: [zai-org/GLM-5](https://huggingface.co/zai-org/GLM-5) and [zai-org/GLM-5-FP8](https://huggingface.co/zai-org/GLM-5-FP8). This guide describes how to run GLM-5 with native FP8.

## Dependencies

### Using Docker

```bash
docker run --gpus all \
  -p 8000:8000 \
  --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --entrypoint bash \
  vllm/vllm-openai:nightly-0b20469c627e94060d1015170b186d19de1db583 \
  -lc "
    apt update -y -qq && \
    apt install -y git -qq && \
    pip install git+https://github.com/huggingface/transformers.git && \
    vllm serve zai-org/GLM-5-FP8 \
      --tensor-parallel-size 8 \
      --tool-call-parser glm47 \
      --reasoning-parser glm45 \
      --enable-auto-tool-choice \
      --served-model-name glm-5-fp8 \
      --trust-remote-code
  "
```

### Installing vLLM from source

```bash
uv venv
source .venv/bin/activate

uv pip install -U vllm --pre --index-url https://pypi.org/simple --extra-index-url https://wheels.vllm.ai/nightly

uv pip install git+https://github.com/huggingface/transformers.git
```

<<<<<<< HEAD
- For FP8 model, you must install DeepGEMM using [install_deepgemm.sh](https://github.com/vllm-project/vllm/blob/v0.16.0rc0/tools/install_deepgemm.sh).
=======
- For FP8 model, you can install DeepGEMM using [install_deepgemm.sh](https://github.com/vllm-project/vllm/blob/main/tools/install_deepgemm.sh).
>>>>>>> origin/main


## Model Deployment

### Serving FP8 Model on 8xH200 (or H20) GPUs (141GB × 8)


```bash
vllm serve zai-org/GLM-5-FP8 \
     --tensor-parallel-size 8 \
     --speculative-config.method mtp \
     --speculative-config.num_speculative_tokens 1 \
     --tool-call-parser glm47 \
     --reasoning-parser glm45 \
     --enable-auto-tool-choice \
     --served-model-name glm-5-fp8
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
    model="glm-5-fp8",
    messages=messages,
    temperature=1,
    max_tokens=4096,
)
print("thinking=on, think content:\n", resp_on.choices[0].message.reasoning)

# Thinking OFF
resp_off = client.chat.completions.create(
    model="glm-5-fp8",
    messages=messages,
    temperature=1,
    max_tokens=4096,
    extra_body={
        "chat_template_kwargs": {
            "enable_thinking": False
        }
    },
)
# The content of reasoning should be None
print("thinking=off:\n", resp_off.choices[0].message.reasoning)
```

### cURL Usage

- Thinking ON (default):

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d ' {
    "model": "glm-5-fp8",
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
    "model": "glm-5-fp8",
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
  --model zai-org/GLM-5-FP8 \
  --dataset-name random \
  --random-input 8000 \
  --random-output 1024 \
  --request-rate 10 \
  --num-prompts 32 \
  --trust-remote-code
  --ignore-eos
```

If successful, you will see the following output.

```shell
============ Serving Benchmark Result ============
Successful requests:                     32        
Failed requests:                         0         
Request rate configured (RPS):           10.00     
Benchmark duration (s):                  71.46     
Total input tokens:                      256000    
Total generated tokens:                  32768     
Request throughput (req/s):              0.45      
Output token throughput (tok/s):         458.55    
Peak output token throughput (tok/s):    832.00    
Peak concurrent requests:                32.00     
Total token throughput (tok/s):          4040.98   
---------------Time to First Token----------------
Mean TTFT (ms):                          13529.94  
Median TTFT (ms):                        13689.81  
P99 TTFT (ms):                           25567.26  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          54.52     
Median TPOT (ms):                        54.54     
P99 TPOT (ms):                           67.51     
---------------Inter-token Latency----------------
Mean ITL (ms):                           54.52     
Median ITL (ms):                         42.22     
P99 ITL (ms):                            914.84    
==================================================

```



