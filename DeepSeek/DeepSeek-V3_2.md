# DeepSeek-V3.2 Usage Guide

## Introduction

[DeepSeek-V3.2](https://huggingface.co/deepseek-ai/DeepSeek-V3.2)  is a model that balances computational efficiency with strong reasoning and agent capabilities through three technical innovations:
- **DeepSeek Sparse Attention (DSA):** An efficient attention mechanism that reduces computational complexity while maintaining performance, optimized for long-context scenarios.
- **Scalable Reinforcement Learning Framework:** The model achieves GPT-5-level performance through robust RL protocols and scaled post-training compute. The high-compute variant, DeepSeek-V3.2-Speciale, surpasses GPT-5 and matches Gemini-3.0-Pro in reasoning, achieving gold-medal level performance in the 2025 IMO and IOI competitions.
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

  VLLM_MOE_USE_DEEP_GEMM=0 vllm serve deepseek-ai/DeepSeek-V3.2 \
   --tensor-parallel-size 8 \
   --tokenizer-mode deepseek_v32 \
   --tool-call-parser deepseek_v32 \
   --enable-auto-tool-choice \
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

### Usage tips

- You can refer to [DeepSeek-V3_2-Exp recipe](../DeepSeek/DeepSeek-V3_2-Exp.md) and [Data Parallel Deployment documentation](https://docs.vllm.ai/en/latest/serving/data_parallel_deployment.html) to conduct related experiments and benchmark testing to select the parallel group suitable for your scenerio.

- Regarding `thinking mode` and `non-thinking mode`, you can refer to [DeepSeek-V3_1recipe](../DeepSeek/DeepSeek-V3_1.md).

## Tool Calling Example


DeepSeek 3.2's thinking mode now supports tool calling, see: [DeepSeek API Doc](https://api-docs.deepseek.com/zh-cn/guides/thinking_mode#%E5%B7%A5%E5%85%B7%E8%B0%83%E7%94%A8). The model can perform multiple rounds of reasoning and tool calls before outputting the final answer. The code example below is directly copied from the DeepSeek official examples. For vLLM, the main modifications are:

 - To enable thinking mode in vLLM, use **extra_body = {"chat_template_kwargs": {"thinking": True}}**. In the DeepSeek official API, the method to enable thinking mode is **extra_body = {"thinking": {"type": "enabled"}}**.
  
 - For the `think`· field, vLLM recommends using **reasoning**, the DeepSeek official API uses **reasoning_content**. 

 - In vLLM, if there are no tool_calls, then tool_calls is  an empty list (`[]`), In contrast, the DeepSeek official API returns `None`.
  
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
        
        # DeepSeek official API
        # if hasattr(message, 'reasoning_content'):
        #     message.reasoning_content = None

        #  vLLM Server
        if hasattr(message, 'reasoning'):
            message.reasoning = None

def run_turn(turn, messages):
    sub_turn = 1
    while True:
        response = client.chat.completions.create(
            model='deepseek-chat',
            messages=messages,
            tools=tools,
            # extra_body={ "thinking": { "type": "enabled" } } # DeepSeek official API
            extra_body = {"chat_template_kwargs": {"thinking": True}} # vLLM Server
        )
        messages.append(response.choices[0].message)
        reasoning_content = response.choices[0].message.reasoning_content
        content = response.choices[0].message.content
        tool_calls = response.choices[0].message.tool_calls
        print(f"Turn {turn}.{sub_turn}\n{reasoning_content=}\n{content=}\n{tool_calls=}")
        # If there is no tool calls, then the model should get a final answer and we need to stop the loop
        # In DeepSeek API, if there are no tool_calls, then tool_calls is None.
        #if tool_calls is None:
        # In vLLM, if there are no tool_calls, then tool_calls is [].
        if not tool_calls:
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

# You can running vLLM server using the following command
#  VLLM_MOE_USE_DEEP_GEMM=0 vllm serve serve deepseek-ai/DeepSeek-V3.2 \
#   --tensor-parallel-size 8 \
#   --tokenizer-mode deepseek_v32 \
#   --tool-call-parser deepseek_v32 \
#   --enable-auto-tool-choice \
#   --reasoning-parser deepseek_v3

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

### vLLM  Server Print

```text
Turn 1.1
reasoning_content="I need to help the user with weather in Hangzhou tomorrow. First, I need to get the current date to determine tomorrow's date. Then I can use the weather function. Let me start by getting the current date."
content=None
tool_calls=[ChatCompletionMessageFunctionToolCall(id='chatcmpl-tool-a2de4337498c482c', function=Function(arguments='{}', name='get_date'), type='function')]
tool result for get_date: 2025-12-01

Turn 1.2
reasoning_content='Today is December 1, 2025. Tomorrow would be December 2, 2025. Now I can get the weather for Hangzhou for that date.'
content=None
tool_calls=[ChatCompletionMessageFunctionToolCall(id='chatcmpl-tool-b11e7a47d3b689ea', function=Function(arguments='{"location": "Hangzhou", "date": "2025-12-02"}', name='get_weather'), type='function')]
tool result for get_weather: Cloudy 7~13°C

Turn 1.3
reasoning_content="I have the weather information: Cloudy with temperatures between 7°C and 13°C. I should provide this to the user in a clear and friendly manner. I'll mention that this is for tomorrow, December 2, 2025. Let me craft the response."
content='The weather in Hangzhou **tomorrow, Tuesday, December 2, 2025**, will be **Cloudy** with temperatures ranging from **7°C to 13°C**.'
tool_calls=[]
Turn 2.1
reasoning_content='The user is asking about the weather in Hangzhou tomorrow again. I already answered this question in the previous exchange, but I should check if "tomorrow" still refers to the same date or if there\'s a new context. The current date is December 1, 2025, so tomorrow would be December 2, 2025. I already provided that information. However, maybe the user is asking again because they want to confirm or maybe they didn\'t see the previous answer? Looking at the conversation, I provided the weather for tomorrow (December 2, 2025). The user\'s latest question is identical to the first one. I should probably respond with the same information, but perhaps acknowledge that I already provided this information. However, since the conversation continues, maybe they want additional details or something else? The weather tool only gives basic info: "Cloudy 7~13°C". I could present it again. But maybe the user expects a different format or more details? I could just repeat the answer. Let me respond politely with the same information.'
content="The weather in Hangzhou **tomorrow, Tuesday, December 2, 2025**, will be **Cloudy** with temperatures ranging from **7°C to 13°C**. \n\nThis is the same forecast I provided earlier - it looks like tomorrow's weather will be consistently cloudy with cool temperatures."
tool_calls=[]

```


### DeepSeek Offical API Print

``` text
Turn 1.1
reasoning_content="The user is asking about the weather in Hangzhou tomorrow. I need to get the current date to determine what tomorrow's date is, then use that to get the weather forecast. Let me first get the current date."
content=''
tool_calls=[ChatCompletionMessageFunctionToolCall(id='call_00_OOAEfTpXddWI9rgC75bfYQJY', function=Function(arguments='{}', name='get_date'), type='function', index=0)]
tool result for get_date: 2025-12-01

Turn 1.2
reasoning_content='Today is December 1, 2025. Tomorrow would be December 2, 2025. So I need to get the weather for Hangzhou on 2025-12-02. Now I can call get_weather with location Hangzhou and date 2025-12-02.'
content=''
tool_calls=[ChatCompletionMessageFunctionToolCall(id='call_00_3P0Xqw5MrVhklmQ4QSACbDq6', function=Function(arguments='{"location": "Hangzhou", "date": "2025-12-02"}', name='get_weather'), type='function', index=0)]
tool result for get_weather: Cloudy 7~13°C

Turn 1.3
reasoning_content='Now I have the weather information: Cloudy with temperatures between 7 and 13 degrees Celsius. I should provide this to the user in a friendly manner. I can mention that tomorrow is December 2nd, and give the forecast. Let me craft the response.'
content='Tomorrow (December 2, 2025) in Hangzhou, the weather will be **cloudy** with temperatures ranging from **7°C to 13°C**.'
tool_calls=None
Turn 2.1
reasoning_content='The user is asking about the weather in Hangzhou tomorrow. I already answered this question in the previous interaction. However, I should check if "tomorrow" is still the same date. The current date is 2025-12-01. Tomorrow would be 2025-12-02. I already provided the weather for that date: Cloudy 7~13°C. \n\nBut wait, the user might be asking again, perhaps not noticing the previous answer. Or maybe they want a different presentation. I should answer again, but maybe with a slightly different phrasing. Also, I should confirm that "tomorrow" is indeed 2025-12-02.\n\nI could just repeat the information. But perhaps I should check if the date has changed? The current date is still 2025-12-01. So tomorrow is still 2025-12-02. I already have the weather data.\n\nI\'ll respond with the weather information again.'
content='Based on the previous query, tomorrow (December 2, 2025) in Hangzhou will be **cloudy** with temperatures between **7°C and 13°C**.'
tool_calls=None
```

