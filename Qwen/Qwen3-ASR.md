# Qwen3-ASR Usage Guide
Qwen3-ASR is a speech-to-text model that achieves accurate and robust speech recogition performance, supporting 11 languages and multiple accents. Qwen3-ASR supports users to prompt the model with texture context in any format to obtain costumized ASR results, and is also good at singing voice recognition. This guide demonstrates how to deploy Qwen3-ASR efficiently with vLLM.

## Installing vllm

```bash
uv venv
source .venv/bin/activate
uv pip install -U vllm --pre \
    --extra-index-url https://wheels.vllm.ai/nightly/cu129 \
    --extra-index-url https://download.pytorch.org/whl/cu129 \
    --index-strategy unsafe-best-match
uv pip install "vllm[audio]" # For additional audio dependencies
```

## Launching Qwen3-ASR with vLLM
### Online Serving
You can easily deploy Qwen3-ASR with vLLM by running the following command
```bash
vllm serve Qwen/Qwen3-ASR-1.7B
```
After the model server is successfully deployed, you can interact with it in multiple ways.

#### Using OpenAI SDK
```python
import base64
import httpx
from openai import OpenAI

# Initialize client
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)

# Create multimodal chat completion request
response = client.chat.completions.create(
    model="Qwen/Qwen3-ASR-1.7B",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "audio_url",
                    "audio_url": {
                        {"url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav"}
                    }
                }
            ]
        }
    ],
)

print(response.choices[0].message.content)
```
This model is also supported on vLLM with OpenAI transcription API.
```python
import httpx
from openai import OpenAI

# Initialize client
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)
audio_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav"
audio_file = httpx.get(audio_url).content

transcription = client.audio.transcriptions.create(
    model="Qwen/Qwen3-ASR-1.7B",
    file=audio_file,
)

print(transcription.text)
```

#### Using cURL
```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
    "messages": [
    {"role": "user", "content": [
        {"type": "audio_url", "audio_url": {"url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav"}}
    ]}
    ]
    }'
```

### Offline Inference
See the following example on using vLLM to run offline infernece with Qwen3-ASR
```python
from vllm import LLM, SamplingParams
from vllm.assets.audio import AudioAsset
import base64
import requests

# Initialize the LLM
llm = LLM(
    model="Qwen/Qwen3-ASR-1.7B"
)

# Load audio
audio_asset = AudioAsset("winning_call")

# Create conversation with audio content
conversation = [
    {
        "role": "user",
        "content": [
            {
                "type": "audio_url",
                "audio_url": {"url": audio_asset.url}
            }
        ]
    }
]

sampling_params = SamplingParams(temperature=0.01, max_tokens=256)

# Run inference using .chat()
outputs = llm.chat(conversation, sampling_params=sampling_params)
print(outputs[0].outputs[0].text)
```