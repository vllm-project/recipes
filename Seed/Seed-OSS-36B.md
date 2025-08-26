# Seed-OSS-36B Usage Guide

This guide describes how to run Seed-OSS-36B models with vLLM and native BF16 precision. Seed-OSS features unique "thinking budget" functionality for controlled reasoning and supports up to 512K context length.

## Installing vLLM

Seed-OSS support was recently added to vLLM main branch and is not yet available in any official release:

```bash
uv venv
source .venv/bin/activate
uv pip install git+https://github.com/vllm-project/vllm.git
```

You may need to download the latest version of the transformer for compatibility:

```bash
uv pip install git+https://github.com/huggingface/transformers.git@56d68c6706ee052b445e1e476056ed92ac5eb383
```

## Running Seed-OSS-36B with BF16

There are two ways to parallelize the model over multiple GPUs: (1) Tensor-parallel or (2) Data-parallel. Each one has its own advantages, where tensor-parallel is usually more beneficial for low-latency / low-load scenarios and data-parallel works better for cases where there is a lot of data with heavy-loads.

Run tensor-parallel like this:

```bash
vllm serve ByteDance-Seed/Seed-OSS-36B-Instruct \
    --host localhost \
    --port 8000 \
    --tensor-parallel-size 8 \
    --enable-auto-tool-choice \
    --tool-call-parser seed_oss \
```

* You can set `--max-model-len` to preserve memory. `--max-model-len=65536` is usually good for most scenarios and max is 512k.
* You can set `--max-num-batched-tokens` to balance throughput and latency, higher means higher throughput but higher latency. `--max-num-batched-tokens=32768` is usually good for prompt-heavy workloads. But you can reduce it to 16k and 8k to reduce activation memory usage and decrease latency.
* vLLM conservatively use 90% of GPU memory, you can set `--gpu-memory-utilization=0.95` to maximize KVCache.
* Make sure to follow the command-line instructions to ensure the tool-calling functionality is properly enabled.

## Thinking Budget Feature

Users can flexibly specify the model's thinking budget. For simpler tasks (such as IFEval), the model's chain of thought (CoT) is shorter, and the score exhibits fluctuations as the thinking budget increases. For more challenging tasks (such as AIME and LiveCodeBench), the model's CoT is longer, and the score improves with an increase in the thinking budget.

If no thinking budget is set (default mode), Seed-OSS will initiate thinking with unlimited length. If a thinking budget is specified, users are advised to prioritize values that are integer multiples of 512 (e.g., 512, 1K, 2K, 4K, 8K, or 16K), as the model has been extensively trained on these intervals. Models are instructed to output a direct response when the thinking budget is 0, and we recommend setting any budget below 512 to this value.

## Usage Examples

### OpenAI Client Usage

You can use the OpenAI client as follows. You can pass `thinking_budget` through `extra_body` to control the thinking budget:

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
    {"role": "user", "content": "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"}
]
extra_body = {"chat_template_kwargs": {"thinking_budget": 512}}
response = client.chat.completions.create(
    model=model, messages=messages, extra_body=extra_body
)
content = response.choices[0].message.content
print("content:\n", content)
```

### Example Outputs

**thinking_budget = 512**:
```
content:
 <seed:think>
Got it, let's try to figure out this problem step by step. First, the question is about Janet's ducks laying eggs, and we need to find out how much money she makes at the farmers' market each day. 
<seed:cot_budget_reflect>I have used 138 tokens, and there are 374 tokens remaining for use.</seed:cot_budget_reflect>
 Let's start by listing out the information given. 

First, her ducks lay 16 eggs per day. That's the total number of eggs she has each day, right? Then, she does a few things with these eggs: she eats three for breakfast every morning, bakes muffins with four every day, and sells the remainder at the farmers' market. Each of those sold eggs is $2, so we need to find the remainder first and then multiply by 2 to get the daily earnings.
<seed:cot_budget_reflect>I have used 260 tokens, and there are 252 tokens remaining for use.</seed:cot_budget_reflect>

Let me write that down. Total eggs: 16. Eggs used: eaten (3) plus muffins (4). So first, let's add up how many eggs she uses each day. 3 + 4 = 7 eggs used. Then the remainder is total eggs minus used eggs, so 16 - 7 = 9 eggs left to sell. Wait, is that right? Let me check again. 16 total, subtract 3 eaten, that's 13 left, then subtract 4 for muffins, that's 13 - 4 = 9.
<seed:cot_budget_reflect>I have used 395 tokens, and there are 117 tokens remaining for use.</seed:cot_budget_reflect>
 Yep, that's 9 eggs. Then she sells each for $2, so 9 times 2 is $18. That seems straightforward. Let me make sure I didn't miss anything. The problem says "daily," so we don't have to worry about anything over multiple days. Just one day: 16 eggs, use 3+4=7, sell 9, 9*2=18.
<seed:cot_budget_reflect>I have exhausted my token budget, and now I will start answering the question.</seed:cot_budget_reflect>
</seed:think>To determine how much Janet makes at the farmers' market daily, follow these steps:

### Step 1: Calculate total eggs laid daily  
Janet’s ducks lay **16 eggs per day**.

### Step 2: Calculate eggs used daily  
- She eats 3 eggs for breakfast.  
- She uses 4 eggs for muffins.  
Total eggs used = \(3 + 4 = 7\) eggs.

### Step 3: Find the number of eggs sold  
Remaining eggs = Total eggs - Eggs used = \(16 - 7 = 9\) eggs.

### Step 4: Calculate daily earnings  
She sells each egg for $2, so total earnings = \(9 \times 2 = 18\) dollars.

**Answer:** 18
```

**thinking_budget = 0**:
```
content:
 The current thinking budget is 0, so I will directly start answering the question.</seed:cot_budget_reflect>
</seed:think>To determine how much Janet makes daily at the farmers' market, follow these steps:

### Step 1: Calculate total eggs laid  
Janet’s ducks lay **16 eggs per day**.

### Step 2: Calculate eggs used  
- She eats 3 eggs for breakfast.  
- She uses 4 eggs for muffins.  
- Total eggs used: \(3 + 4 = 7\) eggs.  

### Step 3: Find remaining eggs for sale  
Subtract used eggs from total eggs:  
\(16 - 7 = 9\) eggs.  

### Step 4: Calculate daily earnings  
She sells each remaining egg for $2:  
\(9 \times 2 = 18\) dollars.  

**Answer:** 18
```

### curl Usage

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ByteDance-Seed/Seed-OSS-36B-Instruct",
    "messages": [{"role": "user", "content": "Explain quantum computing"}],
    "chat_template_kwargs": {
    	"thinking_budget": 512
    }
  }'
```

## Benchmarking

We used the following script to benchmark `ByteDance-Seed/Seed-OSS-36B-Instruct` on RTX 3090 GPU:

```
vllm bench serve \
  --backend vllm \
  --model ByteDance-Seed/Seed-OSS-36B-Instruct \
  --endpoint /v1/completions \
  --host localhost \
  --port 8000 \
  --dataset-name random \
  --random-input 800 \
  --random-output 100 \
  --request-rate 2 \
  --num-prompt 100 \
```

Sample output:

```
============ Serving Benchmark Result ============
Successful requests:                     100       
Request rate configured (RPS):           2.00      
Benchmark duration (s):                  54.08     
Total input tokens:                      79934     
Total generated tokens:                  10000     
Request throughput (req/s):              1.85      
Output token throughput (tok/s):         184.92    
Total Token throughput (tok/s):          1663.06   
---------------Time to First Token----------------
Mean TTFT (ms):                          97.96     
Median TTFT (ms):                        99.71     
P99 TTFT (ms):                           128.60    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          44.39     
Median TPOT (ms):                        43.74     
P99 TPOT (ms):                           49.19     
---------------Inter-token Latency----------------
Mean ITL (ms):                           44.39     
Median ITL (ms):                         46.18     
P99 ITL (ms):                            64.52     
==================================================
```