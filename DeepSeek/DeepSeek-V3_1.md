# DeepSeek-V3.1 Usage Guide


## Introduction
[DeepSeek-V3.1](https://huggingface.co/deepseek-ai/DeepSeek-V3.1) is a hybrid model that supports both thinking mode and non-thinking mode. This guide describes how to dynamically switch between `think` and `non-think` mode in vllm.


## Installing vLLM

```bash
uv venv
source .venv/bin/activate
uv pip install -U vllm --torch-backend auto
```


## Launching DeepSeek-V3.1

### Serving on 8xH200 (or H20) GPUs (141GB × 8)


```bash
vllm serve deepseek-ai/DeepSeek-V3.1 \
  --enable-expert-parallel \
  --tensor-parallel-size 8 \
  --served-model-name ds31 
```

## Using the Model

### OpenAI Client Example

You can use the OpenAI client as follows. You can control whether to enable think mode by using `extra_body={"chat_template_kwargs": {"thinking": False}}`, where `True` enables think mode and `False` disables think mode (non-thinking mode).

```python
  from openai import OpenAI

  openai_api_key = "EMPTY"
  openai_api_base = "http://localhost:8000/v1"

  client = OpenAI(
      api_key=openai_api_key,
      base_url=openai_api_base,
  )

  models = client.models.list()
  model = models.data[0].id

  messages = [{"role": "user", "content": "9.11 and 9.8, which is greater?"}]
  extra_body={"chat_template_kwargs": {"thinking": False}}
  response = client.chat.completions.create(model=model, messages=messages,extra_body=extra_body)
  content = response.choices[0].message.content
  print("content:", content)
```

### curl Example

You can run the following `curl` command:

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "ds31",
        "messages": [
            {
                "role": "user",
                "content": "9.11 and 9.8, which is greater?"
            }
        ],
        "chat_template_kwargs": {
            "thinking": true
        }
    }'
```
