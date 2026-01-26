# GLM-ASR Usage Guide

This guide describes how to run GLM-ASR-Nano-2512 for automatic speech recognition.

## Model Introduction

GLM-ASR-Nano-2512 is a robust, open-source speech recognition model with 1.5B parameters (2B model size). Designed for real-world complexity, it outperforms OpenAI Whisper V3 on multiple benchmarks while maintaining a compact size.

### Key Capabilities

- **Exceptional Dialect Support**: Beyond standard Mandarin and English, the model is highly optimized for Cantonese (粤语) and other dialects, effectively bridging the gap in dialectal speech recognition.
- **Low-Volume Speech Robustness**: Specifically trained for "Whisper/Quiet Speech" scenarios. It captures and accurately transcribes extremely low-volume audio that traditional models often miss.
- **SOTA Performance**: Achieves the lowest average error rate (4.10) among comparable open-source models, showing significant advantages in Chinese benchmarks (Wenet Meeting, Aishell-1, etc.).

## Installing Dependencies

```bash
uv venv
source .venv/bin/activate

# Install transformers from source (required)
uv pip install git+https://github.com/huggingface/transformers.git

uv pip install -U vllm --torch-backend auto # vllm>=0.12.0 is required
```

## Running with vLLM

### Start Server

```bash
vllm serve zai-org/GLM-ASR-Nano-2512
```

### Client Usage

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

# Load audio file and encode to base64
audio_url = "https://github.com/zai-org/GLM-ASR/raw/main/examples/example_en.wav"
audio_data = base64.b64encode(httpx.get(audio_url).content).decode("utf-8")

# Create transcription request
response = client.chat.completions.create(
    model="zai-org/GLM-ASR-Nano-2512",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": audio_data,
                        "format": "wav"
                    }
                }
            ]
        }
    ],
    max_tokens=500
)

print(response.choices[0].message.content)
```

#### Using cURL

```bash
# First encode audio to base64
AUDIO_BASE64=$(curl -sL "https://github.com/zai-org/GLM-ASR/raw/main/examples/example_en.wav" | base64)

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"zai-org/GLM-ASR-Nano-2512\",
    \"messages\": [
      {
        \"role\": \"user\",
        \"content\": [
          {
            \"type\": \"input_audio\",
            \"input_audio\": {
              \"data\": \"$AUDIO_BASE64\",
              \"format\": \"wav\"
            }
          }
        ]
      }
    ],
    \"max_tokens\": 500
  }"
```

#### Using Local Audio File

```python
import base64
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)

# Load local audio file
with open("your_audio.mp3", "rb") as f:
    audio_data = base64.b64encode(f.read()).decode("utf-8")

response = client.chat.completions.create(
    model="zai-org/GLM-ASR-Nano-2512",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": audio_data,
                        "format": "mp3"  # or "wav", "flac", etc.
                    }
                }
            ]
        }
    ],
    max_tokens=500
)

print(response.choices[0].message.content)
```

#### Using Transcribe Endpoint

```python
import httpx
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)

# Transcribe audio from URL
audio_url = "https://github.com/zai-org/GLM-ASR/raw/main/examples/example_en.wav"
audio_file = httpx.get(audio_url).content

response = client.audio.transcriptions.create(
    model="zai-org/GLM-ASR-Nano-2512",
    file=("audio.wav", audio_file),
)

print(response.text)
```

#### Transcribe with cURL

```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer EMPTY" \
  -F "model=zai-org/GLM-ASR-Nano-2512" \
  -F "file=@your_audio.wav"
```

## Notes

- **Transformers Version**: This model requires `transformers >= 5.0.0` for optimal compatibility.

## Additional Resources

- [Model Card](https://huggingface.co/zai-org/GLM-ASR-Nano-2512)
- [GitHub Repository](https://github.com/zai-org/GLM-ASR)
