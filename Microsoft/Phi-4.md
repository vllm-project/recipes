# Phi-4 Usage Guide

This guide describes how to run **Microsoft Phi-4-mini-instruct** on GPU using vLLM.  

Phi-4-multimodal-instruct is a lightweight open multimodal foundation model that leverages the language, vision, and speech research and datasets used for Phi-3.5 and 4.0 models. The model processes text, image, and audio inputs, generating text outputs, and comes with 128K token context length

## GPU Deployment

### Installing vLLM

```bash
uv venv
source .venv/bin/activate

uv pip install -U vllm --torch-backend auto
```

### Running Phi-4-mini-instruct on a Single GPU
```bash
# Start server on a single GPU
vllm serve microsoft/Phi-4-mini-instruct \
  --host 0.0.0.0 \
  --max-model-len 4000
```

## Performance Metrics

### Benchmarking
```bash
vllm bench serve \
  --model microsoft/Phi-4-mini-instruct \
  --dataset-name random \
  --random-input-len 2000 \
  --random-output-len 512 \
  --num-prompts 100
```

## Querying with OpenAI API Client

```python
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
    timeout=3600
)

messages = [
    {
        "role": "user",
        "content": "write short story"
    }
]

response = client.chat.completions.create(
    model="microsoft/Phi-4-mini-instruct",
    messages=messages,
    temperature=0.0
)

print("Generated text:", response.choices[0].message.content)
```

### Multimodal Example (Image + Text)

```python
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
    timeout=3600
)

# Multimodal input: text + image
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "What is shown in this image? Describe it in detail."
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://example.com/image.jpg"
                }
            }
        ]
    }
]

response = client.chat.completions.create(
    model="microsoft/Phi-4-multimodal-instruct",
    messages=messages,
    temperature=0.0
)

print("Generated text:", response.choices[0].message.content)
```

## Available Phi-4 Variants

The Phi-4 series includes multiple model variants, all compatible with the same vLLM serving commands shown in this guide:

- **microsoft/Phi-4-mini-reasoning**  
  Optimized for reasoning tasks

- **microsoft/Phi-4-reasoning**  
  Advanced reasoning capabilities

- **microsoft/Phi-4-multimodal-instruct**  
  Multimodal instruction-following model

- **microsoft/Phi-4-mini-instruct**  
  Instruction-tuned variant optimized for conversational tasks
