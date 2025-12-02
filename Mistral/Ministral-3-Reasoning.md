# Ministral-3 Reasoning Usage Guide

This guide describes how to run Ministral-3 Reasoning that comes with BF16 weights and 3 different sizes:

- 3B: tied embeddings share the embedding and output layers.
- 8B and 14B: each with different layer for embeddings and outputs.

Each of this variant comes with vision support and a large context with a maximum size of 256k.

By using smaller models, expect faster inference with the price of lower performance. Depending on your needs, choose the best trade-off between cost and performance.

## Installing vLLM

```bash
uv venv
source .venv/bin/activate
uv pip install -U vllm --torch-backend auto
```

## Running Ministral-3 Reasoning 3B or 8B on 1xH200

Due to their size, `Ministral-3-3B-Reasoning-2512` and `Ministral-3-8B-Reasoning-2512` can run on a single 1xH200 GPU.

However, for those who do not have access to this GPU generation, vLLM falls back to Marlin FP4 which allows you to still run the model quantized in NVFP4. You won't notice a speed-up in comparison with FP8 quantization but still benefit the memory gain.

Regarding performance on GB200 we observe a significant speed-up and a minor regression on vision datasets probably due to the calibration that was performed mainly on text data.

A simple launch command is:

```bash
# For 3B use `vllm serve mistralai/Ministral-3-3B-Reasoning-2512`
vllm serve mistralai/Ministral-3-8B-Reasoning-2512 \
  --tokenizer_mode mistral --config_format mistral --load_format mistral \
  --enable-auto-tool-choice --tool-call-parser mistral \
  --reasoning-parser mistral
```

Key parameter notes:

* enable-auto-tool-choice: Required when enabling tool usage.
* tool-call-parser mistral: Required when enabling tool usage.
* reasoning-parser mistral: Required when enabling reasoning.


Additional flags:

* You can set `--max-model-len` to preserve memory. By default it is set to `262144` which is quite large but not necessary for most scenarios.
* You can set `--max-num-batched-tokens` to balance throughput and latency, higher means higher throughput but higher latency.


##  Running Ministral-3 Reasoning 14B on 2xH200

To fully exploit the `Ministral-3-14B-Reasoning-2512` we recommend using 2xH200 GPUs for deployment due to its large context. However if you don't need a large context, you can fall back to a single GPU.

A simple launch command is:

```bash

vllm serve mistralai/Ministral-3-14B-Reasoning-2512 \
  --tensor-parallel-size 2 \
  --tokenizer_mode mistral --config_format mistral --load_format mistral \
  --enable-auto-tool-choice --tool-call-parser mistral \
  --reasoning-parser mistral
```

Key parameter notes:

* enable-auto-tool-choice: Required when enabling tool usage.
* tool-call-parser mistral: Required when enabling tool usage.
* reasoning-parser mistral: Required when enabling reasoning.


Additional flags:

* You can set `--max-model-len` to preserve memory. By default it is set to `262144` which is quite large but not necessary for most scenarios.
* You can set `--max-num-batched-tokens` to balance throughput and latency, higher means higher throughput but higher latency.


## Usage of the model

Here we assume that the model `mistralai/Ministral-3-14B-Reasoning-2512` is served and you can ping it to the domain `localhost` with the port `8000` which is the default for vLLM.

### Vision Reasoning

Let's see if the Ministral-3 model knows when to pick a fight !

```python
from typing import Any

from openai import OpenAI
from huggingface_hub import hf_hub_download

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

TEMP = 0.7
TOP_P = 0.95
MAX_TOK = 262144
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id


def load_system_prompt(repo_id: str, filename: str) -> dict[str, Any]:
    file_path = hf_hub_download(repo_id=repo_id, filename=filename)
    with open(file_path, "r") as file:
        system_prompt = file.read()

    index_begin_think = system_prompt.find("[THINK]")
    index_end_think = system_prompt.find("[/THINK]")

    return {
        "role": "system",
        "content": [
            {"type": "text", "text": system_prompt[:index_begin_think]},
            {
                "type": "thinking",
                "thinking": system_prompt[
                    index_begin_think + len("[THINK]") : index_end_think
                ],
                "closed": True,
            },
            {
                "type": "text",
                "text": system_prompt[index_end_think + len("[/THINK]") :],
            },
        ],
    }


SYSTEM_PROMPT = load_system_prompt(model, "SYSTEM_PROMPT.txt")

image_url = "https://static.wikia.nocookie.net/essentialsdocs/images/7/70/Battle.png/revision/latest?cb=20220523172438"

messages = [
    SYSTEM_PROMPT,
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


stream = client.chat.completions.create(
    model=model,
    messages=messages,
    stream=True,
    temperature=TEMP,
    top_p=TOP_P,
    max_tokens=MAX_TOK,
)

print("client: Start streaming chat completions...:\n")
printed_reasoning_content = False
answer = []

for chunk in stream:
    reasoning_content = None
    content = None
    # Check the content is reasoning_content or content
    if hasattr(chunk.choices[0].delta, "reasoning_content"):
        reasoning_content = chunk.choices[0].delta.reasoning_content
    if hasattr(chunk.choices[0].delta, "content"):
        content = chunk.choices[0].delta.content

    if reasoning_content is not None:
        if not printed_reasoning_content:
            printed_reasoning_content = True
            print("Start reasoning:\n", end="", flush=True)
        print(reasoning_content, end="", flush=True)
    elif content is not None:
        # Extract and print the content
        if not reasoning_content and printed_reasoning_content:
            answer.extend(content)
        print(content, end="", flush=True)

if answer:
    print("\n\n=============\nAnswer\n=============\n")
    print("".join(answer))
else:
    print("\n\n=============\nNo Answer\n=============\n")
    print(
        "No answer was generated by the model, probably because the maximum number of tokens was reached."
    )
```

Now we'll make it compute some maths !

```python
from typing import Any

from openai import OpenAI
from huggingface_hub import hf_hub_download

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

TEMP = 0.7
TOP_P = 0.95
MAX_TOK = 262144
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id


def load_system_prompt(repo_id: str, filename: str) -> dict[str, Any]:
    file_path = hf_hub_download(repo_id=repo_id, filename=filename)
    with open(file_path, "r") as file:
        system_prompt = file.read()

    index_begin_think = system_prompt.find("[THINK]")
    index_end_think = system_prompt.find("[/THINK]")

    return {
        "role": "system",
        "content": [
            {"type": "text", "text": system_prompt[:index_begin_think]},
            {
                "type": "thinking",
                "thinking": system_prompt[
                    index_begin_think + len("[THINK]") : index_end_think
                ],
                "closed": True,
            },
            {
                "type": "text",
                "text": system_prompt[index_end_think + len("[/THINK]") :],
            },
        ],
    }


SYSTEM_PROMPT = load_system_prompt(model, "SYSTEM_PROMPT.txt")

image_url = "https://i.ytimg.com/vi/5Y3xLHeyKZU/hqdefault.jpg"

messages = [
    SYSTEM_PROMPT,
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Solve the equations. If they contain only numbers, use your calculator, else only think. Answer in the language of the image.",
            },
            {"type": "image_url", "image_url": {"url": image_url}},
        ],
    },
]

stream = client.chat.completions.create(
    model=model,
    messages=messages,
    stream=True,
    temperature=TEMP,
    top_p=TOP_P,
    max_tokens=MAX_TOK,
)

print("client: Start streaming chat completions...:\n")
printed_reasoning_content = False
answer = []

for chunk in stream:
    reasoning_content = None
    content = None
    # Check the content is reasoning_content or content
    if hasattr(chunk.choices[0].delta, "reasoning_content"):
        reasoning_content = chunk.choices[0].delta.reasoning_content
    if hasattr(chunk.choices[0].delta, "content"):
        content = chunk.choices[0].delta.content

    if reasoning_content is not None:
        if not printed_reasoning_content:
            printed_reasoning_content = True
            print("Start reasoning:\n", end="", flush=True)
        print(reasoning_content, end="", flush=True)
    if content is not None:
        # Extract and print the content
        if not reasoning_content and printed_reasoning_content:
            answer.extend(content)
        print(content, end="", flush=True)

if answer:
    print("\n\n=============\nAnswer\n=============\n")
    print("".join(answer))
else:
    print("\n\n=============\nNo Answer\n=============\n")
    print(
        "No answer was generated by the model, probably because the maximum number of tokens was reached."
    )
```

### Text only request

Let's do more maths and leave it up to the model to figure out how to achieve a result.

```python
from typing import Any
from openai import OpenAI
from huggingface_hub import hf_hub_download

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

TEMP = 0.7
TOP_P = 0.95
MAX_TOK = 262144
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id


def load_system_prompt(repo_id: str, filename: str) -> dict[str, Any]:
    file_path = hf_hub_download(repo_id=repo_id, filename=filename)
    with open(file_path, "r") as file:
        system_prompt = file.read()

    index_begin_think = system_prompt.find("[THINK]")
    index_end_think = system_prompt.find("[/THINK]")

    return {
        "role": "system",
        "content": [
            {"type": "text", "text": system_prompt[:index_begin_think]},
            {
                "type": "thinking",
                "thinking": system_prompt[
                    index_begin_think + len("[THINK]") : index_end_think
                ],
                "closed": True,
            },
            {
                "type": "text",
                "text": system_prompt[index_end_think + len("[/THINK]") :],
            },
        ],
    }


SYSTEM_PROMPT = load_system_prompt(model, "SYSTEM_PROMPT.txt")

query = "Use each number in 2,5,6,3 exactly once, along with any combination of +, -, ×, ÷ (and parentheses for grouping), to make the number 24."

messages = [
    SYSTEM_PROMPT,
    {"role": "user", "content": query}
]
stream = client.chat.completions.create(
  model=model,
  messages=messages,
  stream=True,
  temperature=TEMP,
  top_p=TOP_P,
  max_tokens=MAX_TOK,
)

print("client: Start streaming chat completions...:\n")
printed_reasoning_content = False
answer = []

for chunk in stream:
    reasoning_content = None
    content = None
    # Check the content is reasoning_content or content
    if hasattr(chunk.choices[0].delta, "reasoning_content"):
        reasoning_content = chunk.choices[0].delta.reasoning_content
    if hasattr(chunk.choices[0].delta, "content"):
        content = chunk.choices[0].delta.content

    if reasoning_content is not None:
        if not printed_reasoning_content:
            printed_reasoning_content = True
            print("Start reasoning:\n", end="", flush=True)
        print(reasoning_content, end="", flush=True)
    if content is not None:
        # Extract and print the content
        if not reasoning_content and printed_reasoning_content:
            answer.extend(content)
        print(content, end="", flush=True)

if answer:
    print("\n\n=============\nAnswer\n=============\n")
    print("".join(answer))
else:
    print("\n\n=============\nNo Answer\n=============\n")
    print("No answer was generated by the model, probably because the maximum number of tokens was reached.")
```
