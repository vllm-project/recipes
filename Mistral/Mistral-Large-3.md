# Mistral-Large-3 Instruct Usage Guide

This guide describes how to run Mistral-Large-3-675B-Instruct-2512 with NVFP4 and FP8 weights.

Here are the links to the different formats:
- FP8 (up to `256k` context length): [mistralai/Mistral-Large-3-675B-Instruct-2512](https://huggingface.co/mistralai/Mistral-Large-3-675B-Instruct-2512)
- NVFP4 (for `< 64k` context length): [mistralai/Mistral-Large-3-675B-Instruct-2512-NVFP4](https://huggingface.co/mistralai/Mistral-Large-3-675B-Instruct-2512-NVFP4)


## Installing vLLM

```bash
uv venv
source .venv/bin/activate
uv pip install -U vllm --torch-backend auto
```

## Running the model

## Running Mistral-Large-3-Instruct FP8 on 8xH200

The Mistral-Large-3-Instruct FP8 format can be used on one 8xH200 node. We recommend to use this format if you plan to fine-tune as it can be more precise than NVFP4 in some situations.


A simple launch command is:

```bash

vllm serve mistralai/Mistral-Large-3-675B-Instruct-2512 \
  --tensor-parallel-size 8 \
  --enable-auto-tool-choice --tool-call-parser mistral
```

Key parameter notes:

* enable-auto-tool-choice: Required when enabling tool usage.
* tool-call-parser mistral: Required when enabling tool usage.


Additional flags:

* You can set `--max-model-len` to preserve memory. By default it is set to `262144` which is quite large but not necessary for most scenarios.
* You can set `--max-num-batched-tokens` to balance throughput and latency, higher means higher throughput but higher latency.

## Running Mistral-Large-3-Instruct NVFP4 on 4xB200

We recommend to use this format if you plan to deploy Mistral-Large-3 as it achieves performance similar to FP8 for less memory. However please note that for large context (`> 64k`) we observed a subsequent drop of performance. In such cases, please use the FP8 weights. Otherwise on B200 (Blackwell 200) we observe a significant speed-up and a minor regression on vision datasets probably due to the calibration that was performed mainly on text data.

To take full advantage of the NVFP4 format, consider using B200 GPUs that has a dedicated architecture to maximize its performance. For those who do not have access to this GPU generation, vLLM has the awesome feature to fall back to Marlin FP4 which allows you to still run the quantized model in older generations (A100, H100). You won't notice a speed-up in comparison with FP8 quantization but still benefit a memory gain.


A simple launch command is:

```bash

vllm serve mistralai/Mistral-Large-3-675B-Instruct-2512-NVFP4 \
  --tensor-parallel-size 4 \
  --enable-auto-tool-choice --tool-call-parser mistral
```

Key parameter notes:

* enable-auto-tool-choice: Required when enabling tool usage.
* tool-call-parser mistral: Required when enabling tool usage.


Additional flags:

* You can set `--max-model-len` to preserve memory. By default it is set to `262144` which is quite large but not necessary for most scenarios.
* You can set `--max-num-batched-tokens` to balance throughput and latency, higher means higher throughput but higher latency.


## Usage of the model

Here we assume that the model `mistralai/Mistral-Large-3-675B-Instruct-2512` is served and you can ping it to the domain `localhost` with the port `8000` which is the default for vLLM.

### Vision Reasoning

Let's see if ML3 model knows when to pick a fight !

```python
from datetime import datetime, timedelta

from openai import OpenAI
from huggingface_hub import hf_hub_download

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

TEMP = 0.15
MAX_TOK = 262144

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id


def load_system_prompt(repo_id: str, filename: str) -> str:
    file_path = hf_hub_download(repo_id=repo_id, filename=filename)
    with open(file_path, "r") as file:
        system_prompt = file.read()
    today = datetime.today().strftime("%Y-%m-%d")
    yesterday = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    model_name = repo_id.split("/")[-1]
    return system_prompt.format(name=model_name, today=today, yesterday=yesterday)


SYSTEM_PROMPT = load_system_prompt(model, "SYSTEM_PROMPT.txt")
image_url = "https://static.wikia.nocookie.net/essentialsdocs/images/7/70/Battle.png/revision/latest?cb=20220523172438"

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "What action do you think I should take in this situation? List all the possible actions and explain why you think they are good or bad.",
            },
            {"type": "image_url", "image_url": {"url": image_url}},
        ],
    },
]


response = client.chat.completions.create(
    model=model,
    messages=messages,
    temperature=TEMP,
    max_tokens=MAX_TOK,
)

print(response.choices[0].message.content)
```

### Function Calling

Let's solve some equations thanks to our simple Python calculator tool.

```python
import json
from openai import OpenAI
from huggingface_hub import hf_hub_download

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

TEMP = 0.15
MAX_TOK = 262144

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id


def load_system_prompt(repo_id: str, filename: str) -> str:
    file_path = hf_hub_download(repo_id=repo_id, filename=filename)
    with open(file_path, "r") as file:
        system_prompt = file.read()
    return system_prompt


SYSTEM_PROMPT = load_system_prompt(model, "SYSTEM_PROMPT.txt")

image_url = "https://math-coaching.com/img/fiche/46/expressions-mathematiques.jpg"


def my_calculator(expression: str) -> str:
    # WARNING: Using eval() with untrusted input is a security risk. For production, use a safer expression evaluator.
    return str(eval(expression))


tools = [
    {
        "type": "function",
        "function": {
            "name": "my_calculator",
            "description": "A calculator that can evaluate a mathematical equation and compute its results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate.",
                    },
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rewrite",
            "description": "Rewrite a given text for improved clarity",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The input text to rewrite",
                    }
                },
            },
        },
    },
]

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Thanks to your calculator, compute the results for the equations that involve numbers displayed in the image.",
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": image_url,
                },
            },
        ],
    },
]

response = client.chat.completions.create(
    model=model,
    messages=messages,
    temperature=TEMP,
    max_tokens=MAX_TOK,
    tools=tools,
    tool_choice="auto",
)

tool_calls = response.choices[0].message.tool_calls

results = []
for tool_call in tool_calls:
    function_name = tool_call.function.name
    function_args = tool_call.function.arguments
    if function_name == "my_calculator":
        result = my_calculator(**json.loads(function_args))
        results.append(result)

messages.append({"role": "assistant", "tool_calls": tool_calls})
for tool_call, result in zip(tool_calls, results):
    messages.append(
        {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": tool_call.function.name,
            "content": result,
        }
    )


response = client.chat.completions.create(
    model=model,
    messages=messages,
    temperature=TEMP,
    max_tokens=MAX_TOK,
)

print(response.choices[0].message.content)
```

### Text only request

Mistral-Large-3 can follow your instructions down to the letter.

```python
from openai import OpenAI
from huggingface_hub import hf_hub_download

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

TEMP = 0.15
MAX_TOK = 262144

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id


def load_system_prompt(repo_id: str, filename: str) -> str:
    file_path = hf_hub_download(repo_id=repo_id, filename=filename)
    with open(file_path, "r") as file:
        system_prompt = file.read()
    return system_prompt


SYSTEM_PROMPT = load_system_prompt(model, "SYSTEM_PROMPT.txt")

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {
        "role": "user",
        "content": "Write me a sentence where every word starts with the next letter in the alphabet - start with 'a' and end with 'z'.",
    },
]

response = client.chat.completions.create(
    model=model,
    messages=messages,
    temperature=TEMP,
    max_tokens=MAX_TOK,
)

assistant_message = response.choices[0].message.content
print(assistant_message)
```
