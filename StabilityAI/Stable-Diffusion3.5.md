# Wan2.2 Usage Guide

This guide provides instructions for running Stable-Diffusion3.5 text-to-image generation models using vLLM-Omni with Cache-DiT acceleration.

## Supported Models

- **stabilityai/stable-diffusion-3.5-large**: 8.1B parameters model
- **stabilityai/stable-diffusion-3.5-large-turbo**: 8.1B parameters model (timestep-distilled enabling few-step inference)
- **stabilityai/stable-diffusion-3.5-medium**: 2.5B parameters model

## Installing vLLM-Omni

```bash
uv venv
source .venv/bin/activate
uv pip install vllm==0.12.0
uv pip install git+https://github.com/vllm-project/vllm-omni.git
```

The CLI examples below are from the vLLM-Omni repo. If you want to run them directly, clone that repo and run the scripts from its `examples/offline_inference` directory.

## Text-to-Image Generation

### Basic Usage

```python
from vllm_omni.entrypoints.omni import Omni

omni = Omni(model="stabilityai/stable-diffusion-3.5-medium")

images = omni.generate(
    prompt="a cat wearing sunglasses, cyberpunk style",
    negative_prompt="blurry, low quality",
    height=1024,
    width=1024,
    num_inference_steps=28,
    guidance_scale=7.5,
    num_outputs_per_prompt=2,
)
```

### CLI Usage

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
  --model /model/stable-diffusion-3.5-medium \
  --prompt "a cat wearing sunglasses, cyberpunk style" \
  --negative_prompt "blurry, low quality" \
  --height 1024 \
  --width 1024 \
  --num_inference_steps 28 \
  --guidance_scale 7.5 \
```


## Cache-DiT Acceleration

vLLM-Omni supports Cache-DiT acceleration for stable-diffusion-3.5 models, which can significantly speed up video generation through caching mechanisms.

### Enabling Cache-DiT

```python
from vllm_omni.entrypoints.omni import Omni

omni = Omni(
    model="stabilityai/stable-diffusion-3.5-medium",
    cache_backend="cache_dit",
)

frames = omni.generate(
    prompt="a cat wearing sunglasses, cyberpunk style",
    height=1024,
    width=1024,
    num_inference_steps=28,
)
```

### Custom Cache-DiT Configuration

For fine-tuned control over the acceleration:

```python
omni = Omni(
    model="stabilityai/stable-diffusion-3.5-medium",
    cache_backend="cache_dit",
    cache_config={
        "Fn_compute_blocks": 8,
        "Bn_compute_blocks": 0,
        "max_warmup_steps": 4,
        "residual_diff_threshold": 0.12,
    },
)
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `height` | 1024 | image height (multiples of 16) |
| `width` | 1024 | image width (multiples of 16) |
| `num_inference_steps` | 28 | Denoising steps |
| `guidance_scale` | 1.0 | Classifier-free guidance scale |


## Additional Resources

- [Cache-DiT Acceleration Guide](https://github.com/vipshop/cache-dit)
- [Stable-Diffusion-3.5-Large-Turbo Model Card](https://huggingface.co/stabilityai/stable-diffusion-3.5-large-turbo)
- [Stable-Diffusion-3.5-Large Model Card](https://huggingface.co/stabilityai/stable-diffusion-3.5-large)
- [Stable-Diffusion-3.5-Medium Model Card](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium)
