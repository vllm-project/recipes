# Jina Reranker vLLM Deployment Recipe

This guide contains deployment instructions for the [jinaai/jina-reranker-m0](https://huggingface.co/jinaai/jina-reranker-m0) using vLLM. This is a multilingual, multimodal reranker model designed to rank visual documents across multiple languages. It processes both textual and visual content, including pages with mixed text, figures, tables, and various layouts across over 29 languages.

This guide uses 2x NVIDIA T4 GPUs or 2x NVIDIA L4 GPUs to launch this model.


## Installation

Install vLLM and required dependencies:

```bash
uv pip install vllm
```

## Using vLLM docker image (For AMD users)

```bash
alias drun='sudo docker run -it --network=host --device=/dev/kfd --device=/dev/dri --group-add=video --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --shm-size 32G -v /data:/data -v $HOME:/myhome -w /myhome'
drun rocm/vllm-dev:nightly
``` 

## Online Deployment

Deploy the model as a production-ready API server using vLLM.

### 1. Deploy Model Server

```bash
# https://docs.vllm.ai/en/latest/cli/serve.html
vllm serve jinaai/jina-reranker-m0 \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor_parallel_size 2 \
    --gpu-memory-utilization 0.75 \
    --max_num_seqs 32
```

### 2. Rerank API

The Rerank API returns a ranked list of documents ordered by their relevance to the query.

#### Request Format

```bash
curl -X POST http://localhost:8000/v1/rerank \
-H "accept: application/json" \
-H "Content-Type: application/json" \
-d '{
    "model": "jinaai/jina-reranker-m0",
    "query": "What are the health benefits of green tea?",
    "documents": [
        "Green tea contains antioxidants called catechins that may help reduce inflammation and protect cells from damage.",
        "El precio del café ha aumentado un 20% este año debido a problemas en la cadena de suministro.",
        "Studies show that drinking green tea regularly can improve brain function and boost metabolism.",
        "Basketball is one of the most popular sports in the United States.",
        "绿茶富含儿茶素等抗氧化剂，可以降低心脏病风险，还有助于控制体重。",
        "Le thé vert est riche en antioxydants et peut améliorer la fonction cérébrale."
    ],
    "top_n": 3,
    "return_documents": true
}'
```

#### Response Format

```json
{
    "id": "rerank-f0a2c978b4fb4d61b0a54fd1c05e335f",
    "model": "jinaai/jina-reranker-m0",
    "usage": {
        "total_tokens": 225
    },
    "results": [
        {
            "index": 4,
            "document": {
                "text": "绿茶富含儿茶素等抗氧化剂，可以降低心脏病风险，还有助于控制体重。",
                "multi_modal": null
            },
            "relevance_score": 0.9823843836784363
        },
        {
            "index": 0,
            "document": {
                "text": "Green tea contains antioxidants called catechins that may help reduce inflammation and protect cells from damage.",
                "multi_modal": null
            },
            "relevance_score": 0.9777672290802002
        },
        {
            "index": 2,
            "document": {
                "text": "Studies show that drinking green tea regularly can improve brain function and boost metabolism.",
                "multi_modal": null
            },
            "relevance_score": 0.9752224683761597
        }
    ]
}
```

### 3. Score API

The Score API computes similarity scores between a query and multiple documents without ranking them.

#### Text-to-Text Scoring

```bash
curl -X POST http://localhost:8000/v1/score \
-H "accept: application/json" \
-H "Content-Type: application/json" \
-d '{
    "model": "jinaai/jina-reranker-m0",
    "text_1": [
        "What is the capital of Brazil?",
        "What is the capital of France?"
    ],
    "text_2": [
        "The capital of Brazil is Brasilia.",
        "The capital of France is Paris."
    ]
}'
```

#### Request Parameters
- `model`: Model identifier (required)
- `text_1`: Query text (required)
- `text_2`: Document(s) to score against. Can be a single string or an array of strings (required)

#### Response Format

```json
{
    "id":"score-30d069df61924c4292579640c0d97bcc",
    "object":"list",
    "created":1761686670,
    "model":"jinaai/jina-reranker-m0",
    "data":[
        {
            "index":0,
            "object":"score",
            "score":0.9878721237182617
        },
        {
            "index":1,
            "object":"score",
            "score":0.9879010915756226
        }
    ],
    "usage":{
        "prompt_tokens":47,
        "total_tokens":47,
        "completion_tokens":0,
        "prompt_tokens_details":null
    }
}
```

#### Multimodal Scoring

The Score API supports multimodal inputs, allowing you to score text against images or vice versa:

```bash
curl -X POST http://localhost:8000/v1/score \
-H "accept: application/json" \
-H "Content-Type: application/json" \
-d '{
    "model": "jinaai/jina-reranker-m0",
    "text_1": "A cat",
    "text_2": {
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": "cat_img.jpg"
                }
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": "dog_img.jpg"
                }
            }
        ]
    }
}'
```


## Offline Deployment

Use the model directly in your Python code without running a server.

```python
from vllm import LLM

MODEL = "jinaai/jina-reranker-m0"

# Initialize the LLM engine
llm = LLM(
    model=MODEL,
    tensor_parallel_size=2,
    gpu_memory_utilization=0.75,
    max_model_len=1024,
    max_num_seqs=32,
    kv_cache_dtype="fp8",
    dtype="bfloat16",
)

# Prepare query and documents
query = "fast recipes for weeknight dinners"
documents = [
    "A 65-minute pasta with garlic and olive oil.",
    "Slow braised short ribs that cook for 5 hours.",
    "Stir-fry veggies with pre-cooked rice.",
]

# Compute scores
res = llm.score(query, documents)

# Extract and print scores
for item in res:
    print(item.outputs.score)
```


## Resources

- [jina-reranker-m0 on Hugging Face](https://huggingface.co/jinaai/jina-reranker-m0)
- [vLLM Documentation](https://docs.vllm.ai/)
- [vLLM Score API Documentation](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#score-api)
- [vLLM CLI Serve Reference](https://docs.vllm.ai/en/latest/cli/serve.html)
