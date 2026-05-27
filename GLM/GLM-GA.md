# GLM-GA Usage Guide

This guide describes how to run GLM-GA for image and video understanding with vLLM.

GLM-GA is a dense vision-language model (~10B parameters) based on the GLM-4.6V-Flash architecture. It uses dedicated `GlmgaImageProcessor` and `GlmgaVideoProcessor` sub-processors. The key difference from GLM-4.6V is in video processing: GLM-GA samples at a fixed 2 fps and supports up to 640 frames, enabling long-video understanding.

## Installing vLLM

```bash
uv venv
source .venv/bin/activate
uv pip install -U vllm --torch-backend auto
uv pip install git+https://github.com/huggingface/transformers.git    # Installed from main branch
```

## Running GLM-GA on a single H100/H200

```bash
VLLM_VIDEO_LOADER_BACKEND=glm4_6v
vllm serve zai-org/GLM-GA \
     --tool-call-parser glm47 \
     --reasoning-parser glm45 \
     --enable-auto-tool-choice \
     --allowed-local-media-path / \
     --mm-processor-cache-type shm
```

* vLLM conservatively uses 90% of GPU memory; set `--gpu-memory-utilization=0.95` to maximize KV cache.

## Client Usage

### Image Understanding

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
resp = client.chat.completions.create(
    model="zai-org/GLM-GA",
    messages=[{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
            {"type": "text", "text": "Describe the image."}
        ]
    }],
    max_tokens=512,
)
print(resp.choices[0].message.content)
```

### Video Understanding

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
resp = client.chat.completions.create(
    model="zai-org/GLM-GA",
    messages=[{
        "role": "user",
        "content": [
            {"type": "video_url", "video_url": {"url": "https://example.com/video.mp4"}},
            {"type": "text", "text": "Summarize what happens in this video."}
        ]
    }],
    max_tokens=1024,
)
print(resp.choices[0].message.content)
```

## Video Processing Details

GLM-GA uses a dedicated `GlmgaVideoProcessor` that differs from GLM-4.6V:

| Feature | GLM-4.6V | GLM-GA |
|---------|----------|--------|
| FPS | Dynamic (3/1/0.5 by duration) | Fixed 2 fps |
| Max frames | 640 | 640 |
| Max pixels (video) | 47M | 87M |
| Frame upsampling | Duration-based | `math.floor` aligned with HF |

## Troubleshooting

- **Long-context memory:** At 128K context, tune `--max-num-batched-tokens` and `--gpu-memory-utilization` to prevent OOM.
- **Video loading errors:** Ensure OpenCV or PyAV is installed for video decoding.
