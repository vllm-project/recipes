# GLM-Image Usage Guide

This guide describes how to run GLM-Image for text-to-image and image-to-image generation using vLLM-Omni.

## Model Introduction

GLM-Image is an image generation model that adopts a hybrid autoregressive + diffusion decoder architecture. In general image generation quality, GLM-Image aligns with mainstream latent diffusion approaches, but it shows significant advantages in text-rendering and knowledge-intensive generation scenarios.

### Architecture

- **Autoregressive Generator**: A 9B-parameter model initialized from GLM-4-9B-0414, with an expanded vocabulary to incorporate visual tokens. The model first generates a compact encoding of approximately 256 tokens, then expands to 1K–4K tokens, corresponding to 1K–2K high-resolution image outputs.
- **Diffusion Decoder**: A 7B-parameter decoder based on a single-stream DiT architecture for latent-space image decoding. It is equipped with a Glyph Encoder text module, significantly improving accurate text rendering within images.

### Key Capabilities

- **Text-to-Image**: Generates high-detail images from textual descriptions, with particularly strong performance in information-dense scenarios.
- **Image-to-Image**: Supports a wide range of tasks, including image editing, style transfer, multi-subject consistency, and identity-preserving generation for people and objects.
- **Text Rendering**: Exceptional ability to render accurate text within generated images.
- **Knowledge-Intensive Generation**: Strong performance in tasks requiring precise semantic understanding and complex information expression.

## Installing Dependencies

```bash
# init uv env
uv venv --python 3.12 --seed
source .venv/bin/activate

# install vllm
uv pip install -U vllm --torch-backend auto

# install vllm-omni
uv pip install vllm-omni

# install up-to-date transformers and diffusers
pip install git+https://github.com/huggingface/transformers.git
pip install git+https://github.com/huggingface/diffusers.git
```

## Offline Text-to-Image Inference

Run the text-to-image generation script:

```bash
# Text to Image
cd examples/offline_inference/text_to_image
python3 text_to_image.py --model zai-org/GLM-Image --output t2i_output.png

# Image to Image
cd examples/offline_inference/image_to_image
wget https://vllm-public-assets.s3.us-west-2.amazonaws.com/omni-assets/qwen-bear.png
python3 image_to_image.py --model zai-org/GLM-Image --image qwen-bear.png --output i2i_output.png
```

### Generation Configuration

The default configuration for GLM-Image:

```
============================================================
Generation Configuration:
  Model: zai-org/GLM-Image
  Inference steps: 50
  Cache backend: None (no acceleration)
  Parallel configuration: tensor_parallel_size=1, ulysses_degree=1,
                          ring_degree=1, cfg_parallel_size=1
  Image size: 1024x1024
============================================================
```

### Custom Text-to-Image Example

```python
from vllm_omni import Omni

# Initialize the model
omni = Omni(model="zai-org/GLM-Image")

# Generate image from text prompt
prompt = "a cup of coffee on the table"
outputs = omni.generate(prompt)

# Save the generated image
for output in outputs:
    for req_output in output.request_output:
        if req_output.images:
            req_output.images[0].save("output.png")
```

### Notes

- The target image resolution must be divisible by 32, otherwise it will throw an error.
- The AR model used in GLM-Image is configured with `do_sample=True`, temperature of `0.9`, and top_p of `0.75` by default.
- A higher temperature results in more diverse and rich outputs, but may decrease output stability.
- Please ensure that all text intended to be rendered in the image is enclosed in quotation marks in the model input.
- Model loading takes approximately 33 GB GPU memory and ~10 seconds.

## Online Serving

```bash
vllm serve zai-org/GLM-Image --omni
```

### Client Usage

#### Using cURL

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "A beautiful landscape painting"}
    ]
  }' | jq -r '.choices[0].message.content[0].image_url.url' | cut -d',' -f2- | base64 -d > output.png
```

#### Using OpenAI SDK

```python
import base64
from openai import OpenAI

# Initialize client
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)

# Text-to-Image generation
prompt = "A beautiful landscape painting with mountains and a lake at sunset"

response = client.chat.completions.create(
    model="zai-org/GLM-Image",
    messages=[
        {"role": "user", "content": prompt}
    ]
)

# Extract and save the generated image
image_url = response.choices[0].message.content[0].image_url.url
# The image_url is in format: data:image/png;base64,<base64_data>
image_data = base64.b64decode(image_url.split(",")[1])

with open("output.png", "wb") as f:
    f.write(image_data)

print("Image saved to output.png")
```

#### Text with Specific Rendering

```python
import base64
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)

# Use quotation marks for text that should be rendered in the image
prompt = '''A coffee shop menu board with "Today's Special" written at the top,
featuring "Cappuccino $4.50" and "Latte $5.00" in elegant handwriting'''

response = client.chat.completions.create(
    model="zai-org/GLM-Image",
    messages=[
        {"role": "user", "content": prompt}
    ]
)

image_url = response.choices[0].message.content[0].image_url.url
image_data = base64.b64decode(image_url.split(",")[1])

with open("menu_board.png", "wb") as f:
    f.write(image_data)
```

### Image-to-Image

#### Using cURL

```bash
# Encode local image to base64
IMAGE_BASE64=$(base64 -i input.png)

curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,'"$IMAGE_BASE64"'"}
          },
          {
            "type": "text",
            "text": "Replace the background with a sunset beach scene"
          }
        ]
      }
    ]
  }' | jq -r '.choices[0].message.content[0].image_url.url' | cut -d',' -f2- | base64 -d > output.png
```

#### Using OpenAI SDK

```python
import base64
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)

# Read and encode the input image
with open("input.png", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode("utf-8")

# Image-to-Image generation
response = client.chat.completions.create(
    model="zai-org/GLM-Image",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_base64}"}
                },
                {
                    "type": "text",
                    "text": "Replace the background with a sunset beach scene"
                }
            ]
        }
    ]
)

# Extract and save the generated image
image_url = response.choices[0].message.content[0].image_url.url
image_data = base64.b64decode(image_url.split(",")[1])

with open("output.png", "wb") as f:
    f.write(image_data)

print("Image saved to output.png")
```

## Notes

- **Transformers Version**: This model requires `transformers >= 5.0.0` for optimal compatibility.

## Additional Resources

- [Model Card](https://huggingface.co/zai-org/GLM-Image)
- [Technical Blog](https://z.ai/blog/glm-image)
- [GitHub Repository](https://github.com/zai-org/GLM-Image)
