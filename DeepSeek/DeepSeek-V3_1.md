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

### Function calling

vLLM also supports calling user-defined functions. Make sure to run your DeepSeek-V3.1 models with the following arguments.

```bash
vllm serve ... 
    --enable-auto-tool-choice 
    --tool-call-parser deepseek_v31 
    --chat-template examples/tool_chat_template_deepseekv31.jinja
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

messages = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Who are you?"},
    {"role": "assistant", "content": "<think>Hmm</think>I am DeepSeek"},
    {"role": "user", "content": "9.11 and 9.8, which is greater?"},
]
extra_body = {"chat_template_kwargs": {"thinking": False}}
response = client.chat.completions.create(
    model=model, messages=messages, extra_body=extra_body
)
content = response.choices[0].message.content
print("content:\n", content)

```
### Example Outputs

#### thinking=True
- As shown below, the output results contain `</think>`
```text
 Hmm, the user is asking which number is greater between 9.11 and 9.8. This seems straightforward, but I should be careful because decimals can sometimes confuse people. 

I recall that comparing decimals involves looking at each digit from left to right. Both numbers have the same whole number part (9), so I need to compare the decimal parts. 0.11 is greater than 0.8 because 0.11 is equivalent to 0.110 and 0.8 is 0.800, so 110 thousandths is greater than 800 thousandths? Wait no, that’s wrong. 

Actually, 0.8 is the same as 0.80, and 0.11 is less than 0.80. So 9.11 is actually less than 9.8. I should double-check that. Yes, 9.8 is larger because 0.8 > 0.11. 

I’ll explain it clearly by comparing the tenths place: 9.8 has 8 tenths, while 9.11 has 1 tenth and 1 hundredth, so 8 tenths is indeed larger. 

The answer is 9.8 is greater. I’ll state it confidently and offer further help if needed.</think>9.8 is greater than 9.11.  

To compare them:  
- 9.8 is equivalent to 9.80  
- 9.80 has 8 tenths, while 9.11 has only 1 tenth  
- Since 8 tenths (0.8) is greater than 1 tenth (0.1), 9.8 > 9.11  

Let me know if you need further clarification! 😊
```
#### thinking=False

```text
 The number **9.11** is greater than **9.8**.  

To compare them:  
- 9.11 = 9 + 11/100  
- 9.8 = 9 + 80/100  

Since 11/100 (0.11) is less than 80/100 (0.80), 9.11 is actually smaller than 9.8. Wait, let me correct that:  

Actually, **9.8 is greater than 9.11**.  

- 9.8 can be thought of as 9.80  
- Comparing 9.80 and 9.11: 80 hundredths is greater than 11 hundredths.  

So, **9.8 > 9.11**.  

Apologies for the initial confusion! 😅
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
