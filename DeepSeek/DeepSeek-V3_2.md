# DeepSeek-V3.2 Usage Guide

## Introduction

[DeepSeek-V3.2](https://huggingface.co/deepseek-ai/DeepSeek-V3.2)  is a model that balances computational efficiency with strong reasoning and agent capabilities through three technical innovations:
- **DeepSeek Sparse Attention (DSA):** An efficient attention mechanism that reduces computational complexity while maintaining performance, optimized for long-context scenarios.
- **Scalable Reinforcement Learning Framework:**: The model achieves GPT-5-level performance through robust RL protocols and scaled post-training compute. The high-compute variant, DeepSeek-V3.2-Speciale, surpasses GPT-5 and matches Gemini-3.0-Pro in reasoning, achieving gold-medal level performance in the 2025 IMO and IOI competitions.
- **Large-Scale Agentic Task Synthesis Pipeline:** A novel data synthesis pipeline that generates training data at scale, integrating reasoning into tool-use scenarios and improving model compliance and generalization in complex interactive environments.


## Installing vLLM

```bash
uv venv
source .venv/bin/activate
uv pip install vllm --extra-index-url https://wheels.vllm.ai/nightly
```

## Launching DeepSeek-V3.2


- The chat-template changes in the DeepSeek-V3.2 are quite significant. vLLM adapts to this through `--tokenizer-mode deepseek_v32`.


```bash

  VLLM_MOE_USE_DEEP_GEMM=0 vllm serve serve deepseek-ai/DeepSeek-V3.2 \
   --tensor-parallel-size 8 \
   --tokenizer-mode deepseek_v32 \
   --tool-call-parser deepseek_v32 \
   --reasoning-parser deepseek_v3
   
```


## Accuracy Benchmarking


### GSM8K

- Script

```bash
lm_eval --model local-completions --model_args "model=deepseek-ai/DeepSeek-V3.2,base_url=http://0.0.0.0:8000/v1/completions,max_length=8192,tokenized_requests=False,tokenizer_backend=None,num_concurrent=32" --tasks gsm8k --num_fewshot 5
```

- Result

``` bash
local-completions (model=deepseek-ai/DeepSeek-V3.2,base_url=http://0.0.0.0:8000/v1/completions,max_length=8192,tokenized_requests=False,tokenizer_backend=None,num_concurrent=32), gen_kwargs: (None), limit: None, num_fewshot: 5, batch_size: 1
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9560|±  |0.0056|
|     |       |strict-match    |     5|exact_match|↑  |0.9553|±  |0.0057|
```



### AIME25

- Script

```bash
lm_eval --model local-chat-completions --model_args "model=deepseek-ai/DeepSeek-V3.2,base_url=http://0.0.0.0:8000/v1/chat/completions,tokenized_requests=False,tokenizer_backend=None,num_concurrent=20,timeout=5000,max_length=72768" --tasks aime25 --apply_chat_template --gen_kwargs '{"temperature":1.0,"max_gen_toks":72768,"top_p":0.95,"chat_template_kwargs":{"thinking":true}}' --log_samples --output_path "aime25_ds32"    
```


- Result

``` bash
local-chat-completions (model=deepseek-ai/DeepSeek-V3.2,base_url=http://0.0.0.0:8000/v1/chat/completions,tokenized_requests=False,tokenizer_backend=None,num_concurrent=20,timeout=5000,max_length=72768), gen_kwargs: ({'temperature': 1.0, 'max_gen_toks': 72768, 'top_p': 0.95, 'chat_template_kwargs': {'thinking': True}}), limit: None, num_fewshot: None, batch_size: 1
|Tasks |Version|Filter|n-shot|  Metric   |   |Value |   |Stderr|
|------|------:|------|-----:|-----------|---|-----:|---|-----:|
|aime25|      0|none  |     0|exact_match|↑  |0.9333|±  |0.0463|
```



## Benchmarking

We used the following script to benchmark `deepseek-ai/DeepSeek-V3.2` on 8*H20.

```bash
vllm bench serve \
  --model deepseek-ai/DeepSeek-V3.2 \
  --dataset-name random \
  --random-input 2048 \
  --random-output 1024 \
  --request-rate 10 \
  --num-prompt 100  \ 
  --trust-remote-code
```



### TP8 Benchmark Output

```shell
============ Serving Benchmark Result ============
Successful requests:                     100       
Failed requests:                         0         
Request rate configured (RPS):           10.00     
Benchmark duration (s):                  129.34    
Total input tokens:                      204800    
Total generated tokens:                  102400    
Request throughput (req/s):              0.77      
Output token throughput (tok/s):         791.73    
Peak output token throughput (tok/s):    1300.00   
Peak concurrent requests:                100.00    
Total Token throughput (tok/s):          2375.18   
---------------Time to First Token----------------
Mean TTFT (ms):                          21147.20  
Median TTFT (ms):                        21197.97  
P99 TTFT (ms):                           41133.00  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          99.71     
Median TPOT (ms):                        99.25     
P99 TPOT (ms):                           124.28    
---------------Inter-token Latency----------------
Mean ITL (ms):                           99.71     
Median ITL (ms):                         76.89     
P99 ITL (ms):                            2032.37   
==================================================


```

### Performance tips

You can refer to [DeepSeek-V3_2-Exp recipe](recipes/DeepSeek/DeepSeek-V3_2-Exp.md) and [Data Parallel Deployment documentation](https://docs.vllm.ai/en/latest/serving/data_parallel_deployment.html) to conduct related experiments and benchmark testing to select the parallel group suitable for your scenerio



## Tool Calling Example


DeepSeek 3.2's thinking mode now supports tool calling,see: [DeepSeek API Doc](https://api-docs.deepseek.com/zh-cn/guides/thinking_mode#%E5%B7%A5%E5%85%B7%E8%B0%83%E7%94%A8). The model can perform multiple rounds of reasoning and tool calls before outputting the final answer.

``` python

import os
import json
from openai import OpenAI

# The definition of the tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_date",
            "description": "Get the current date",
            "parameters": { "type": "object", "properties": {} },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather of a location, the user should supply the location and date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": { "type": "string", "description": "The city name" },
                    "date": { "type": "string", "description": "The date in format YYYY-mm-dd" },
                },
                "required": ["location", "date"]
            },
        }
    },
]

# The mocked version of the tool calls
def get_date_mock():
    return "2025-12-01"

def get_weather_mock(location, date):
    return "Cloudy 7~13°C"

TOOL_CALL_MAP = {
    "get_date": get_date_mock,
    "get_weather": get_weather_mock
}

def clear_reasoning_content(messages):
    for message in messages:
        if hasattr(message, 'reasoning'):
            message.reasoning = None

def run_turn(turn, messages):
    sub_turn = 1
    while True:
        response = client.chat.completions.create(
            model='deepseek-chat',
            messages=messages,
            tools=tools,
            extra_body = {"chat_template_kwargs": {"thinking": True}}
        )

        reasoning_content = response.choices[0].message.reasoning

        messages.append(
            {
                "role": "assistant",
                "tool_calls": response.choices[0].message.tool_calls,
                "reasoning": reasoning_content, # append reasoning
            }
        )

        content = response.choices[0].message.content
        tool_calls = response.choices[0].message.tool_calls
        print(f"Turn {turn}.{sub_turn}\n{reasoning_content=}\n{content=}\n{tool_calls=}")
        # If there is no tool calls, then the model should get a final answer and we need to stop the loop
        if tool_calls is None:
            break
        for tool in tool_calls:
            tool_function = TOOL_CALL_MAP[tool.function.name]
            tool_result = tool_function(**json.loads(tool.function.arguments))
            print(f"tool result for {tool.function.name}: {tool_result}\n")
            messages.append({
                "role": "tool",
                "tool_call_id": tool.id,
                "content": tool_result,
            })
        sub_turn += 1

client = OpenAI(
    api_key=os.environ.get('DEEPSEEK_API_KEY'),
    base_url=os.environ.get('DEEPSEEK_BASE_URL'),
)

# The user starts a question
turn = 1
messages = [{
    "role": "user",
    "content": "How's the weather in Hangzhou Tomorrow"
}]
run_turn(turn, messages)

# The user starts a new question
turn = 2
messages.append({
    "role": "user",
    "content": "How's the weather in Hangzhou Tomorrow"
})
# We recommended to clear the reasoning_content in history messages so as to save network bandwidth
clear_reasoning_content(messages)
run_turn(turn, messages)

```