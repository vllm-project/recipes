# Wan2.2 Usage Guide

This guide provides instructions for running Wan2.2 video generation models using vLLM-Omni with Cache-DiT acceleration.

## Supported Models

- **Wan-AI/Wan2.2-T2V-A14B-Diffusers**: Text-to-Video (MoE architecture, 14B active parameters)
- **Wan-AI/Wan2.2-I2V-A14B-Diffusers**: Image-to-Video (MoE architecture, 14B active parameters)
- **Wan-AI/Wan2.2-TI2V-5B-Diffusers**: Unified Text-to-Video + Image-to-Video (dense 5B)

## Installing vLLM-Omni

```bash
uv venv
source .venv/bin/activate
uv pip install vllm==0.12.0
uv pip install git+https://github.com/vllm-project/vllm-omni.git
```

The CLI examples below are from the vLLM-Omni repo. If you want to run them directly, clone that repo and run the scripts from its `examples/offline_inference` directory.

## Text-to-Video Generation

### Basic Usage

```python
from vllm_omni.entrypoints.omni import Omni

omni = Omni(model="Wan-AI/Wan2.2-T2V-A14B-Diffusers")

frames = omni.generate(
    "Two anthropomorphic cats in comfy boxing gear fight on a spotlighted stage.",
    height=720,
    width=1280,
    num_frames=81,
    num_inference_steps=40,
    guidance_scale=4.0,
)
```

### CLI Usage

```bash
python examples/offline_inference/text_to_video/text_to_video.py \
  --model Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --prompt "A serene lakeside sunrise with mist over the water." \
  --height 720 \
  --width 1280 \
  --num_frames 81 \
  --num_inference_steps 40 \
  --guidance_scale 4.0 \
  --fps 24 \
  --output t2v_output.mp4
```

## Image-to-Video Generation

### Basic Usage

```python
import PIL.Image
from vllm_omni.entrypoints.omni import Omni

omni = Omni(model="Wan-AI/Wan2.2-I2V-A14B-Diffusers")

image = PIL.Image.open("input.jpg").convert("RGB")
frames = omni.generate(
    "A cat playing with yarn",
    pil_image=image,
    height=480,
    width=832,
    num_frames=81,
    num_inference_steps=50,
    guidance_scale=5.0,
)
```

### CLI Usage

```bash
python examples/offline_inference/image_to_video/image_to_video.py \
  --model Wan-AI/Wan2.2-I2V-A14B-Diffusers \
  --image input.jpg \
  --prompt "A cat playing with yarn" \
  --num_frames 81 \
  --num_inference_steps 50 \
  --guidance_scale 5.0 \
  --fps 16 \
  --output i2v_output.mp4
```

### TI2V CLI Usage

```bash
python examples/offline_inference/image_to_video/image_to_video.py \
  --model Wan-AI/Wan2.2-TI2V-5B-Diffusers \
  --image input.jpg \
  --prompt "A cat playing with yarn" \
  --num_frames 81 \
  --num_inference_steps 50 \
  --guidance_scale 5.0 \
  --fps 16 \
  --output ti2v_output.mp4
```

## Cache-DiT Acceleration

vLLM-Omni supports Cache-DiT acceleration for Wan2.2 models, which can significantly speed up video generation through caching mechanisms.

### Enabling Cache-DiT

```python
from vllm_omni.entrypoints.omni import Omni

omni = Omni(
    model="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    cache_backend="cache_dit",
)

frames = omni.generate(
    "A beautiful sunset over the ocean",
    height=720,
    width=1280,
    num_frames=81,
    num_inference_steps=40,
)
```

### Custom Cache-DiT Configuration

For fine-tuned control over the acceleration:

```python
omni = Omni(
    model="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
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
| `height` | 720 (T2V) / auto (I2V) | Video height (multiples of 16) |
| `width` | 1280 (T2V) / auto (I2V) | Video width (multiples of 16) |
| `num_frames` | 81 | Number of frames to generate |
| `num_inference_steps` | 40-50 | Denoising steps |
| `guidance_scale` | 4.0-5.0 | Classifier-free guidance scale |
| `boundary_ratio` | 0.875 | Boundary split ratio for MoE models |
| `flow_shift` | 5.0 (720p) / 12.0 (480p) | Scheduler flow shift |

## Notes

- The CLI scripts use `diffusers.utils.export_to_video`, so `diffusers` must be installed in the environment where you run them.

## Additional Resources

- [Cache-DiT Acceleration Guide](https://github.com/vipshop/cache-dit)
- [Wan2.2-T2V-A14B-Diffusers Model Card](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers)
- [Wan2.2-I2V-A14B-Diffusers Model Card](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers)
- [Wan2.2-TI2V-5B-Diffusers Model Card](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers)
