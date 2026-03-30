# Qwen-Image Usage Guide

Qwen-Image models include the following models:

| Model | HuggingFace | Description |
|-------|-------------|-------------|
| **Qwen-Image** | [🤗 Qwen/Qwen-Image](https://huggingface.co/Qwen/Qwen-Image) | Text-to-image generation (20B parameters, Aug 2025) |
| **Qwen-Image-2512** | [🤗 Qwen/Qwen-Image-2512](https://huggingface.co/Qwen/Qwen-Image-2512) | Updated T2I with enhanced realism and text rendering (Dec 2025) |
| **Qwen-Image-Edit** | [🤗 Qwen/Qwen-Image-Edit](https://huggingface.co/Qwen/Qwen-Image-Edit) | Single-image editing with semantic and appearance control (Aug 2025) |
| **Qwen-Image-Edit-2509** | [🤗 Qwen/Qwen-Image-Edit-2509](https://huggingface.co/Qwen/Qwen-Image-Edit-2509) | Multi-image editing with improved consistency (Sep 2025) |
| **Qwen-Image-Edit-2511** | [🤗 Qwen/Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) | Further enhanced consistency, built-in LoRA support (Nov 2025) |
| **Qwen-Image-Layered** | [🤗 Qwen/Qwen-Image-Layered](https://huggingface.co/Qwen/Qwen-Image-Layered) | Decomposes an input image into multiple RGBA layers (Dec 2025) |

All models share the same DiT transformer core; hence, the acceleration methods (e.g., cache methods, parallelism methods) are applicable across the entire series.

## Installation

```bash
# Clone and install vllm-omni
git clone https://github.com/vllm-project/vllm-omni.git
cd vllm-omni
uv venv
source .venv/bin/activate
uv pip install -e . vllm==0.18.0
cd vllm-omni
```

## Usage

### Text-to-Image (Qwen-Image, Qwen-Image-2512)

Qwen-Image and Qwen-Image-2512 are text-to-image models. Use the `text_to_image.py` script:

```bash
# Qwen-Image (default)
python3 ./examples/offline_inference/text_to_image/text_to_image.py \
    --model Qwen/Qwen-Image \
    --prompt "a cup of coffee on the table" \
    --output output_qwen_image.png \
    --num-inference-steps 50 \
    --cfg-scale 4.0

# Qwen-Image-2512
python3 ./examples/offline_inference/text_to_image/text_to_image.py \
    --model Qwen/Qwen-Image-2512 \
    --prompt "a cup of coffee on the table" \
    --output output_qwen_image_2512.png \
    --num-inference-steps 50 \
    --cfg-scale 4.0
```

> Notes: 
> 1. vLLM-Omni enables torch.compile by default. Try `--enforce-eager` if you want to disable it.
> 2. vLLM-Omni does not enable CPU offload automatically. If you encounter OOM, please `--enable-cpu-offload` or `--enable-layerwise-offload`.


### Image Editing (Qwen-Image-Edit)

Qwen-Image-Edit is the image editing version of Qwen-Image. It simultaneously feeds the input image into Qwen2.5-VL (for visual semantic control) and the VAE Encoder (for visual appearance control), achieving capabilities in both semantic and appearance editing.

```bash
# Single image input (Qwen-Image-Edit)
python3 ./examples/offline_inference/image_to_image/image_edit.py \
    --model Qwen/Qwen-Image-Edit \
    --image qwen_bear.png \
    --prompt "Let this mascot dance under the moon, surrounded by floating stars and poetic bubbles such as 'Be Kind'" \
    --output output_image_edit.png \
    --num-inference-steps 50 \
    --cfg-scale 4.0
```

For multiple image inputs, use `Qwen/Qwen-Image-Edit-2509` or `Qwen/Qwen-Image-Edit-2511`:

```bash
# Qwen-Image-Edit-2511 example (multiple images)
python3 ./examples/offline_inference/image_to_image/image_edit.py \
    --model Qwen/Qwen-Image-Edit-2511 \
    --image image1.png image2.png \
    --prompt "Add a white art board written with colorful text 'vLLM-Omni' on grassland. Add a paintbrush in the bear's hands. position the bear standing in front of the art board as if painting" \
    --output output_image_edit.png \
    --num-inference-steps 50 \
    --cfg-scale 4.0
```

### Image Layering (Qwen-Image-Layered)

Qwen-Image-Layered decomposes an input image into multiple RGBA layers:

```bash
python3 ./examples/offline_inference/image_to_image/image_edit.py \
    --model Qwen/Qwen-Image-Layered \
    --image input.png \
    --prompt "" \
    --output layered \
    --num-inference-steps 50 \
    --cfg-scale 4.0 \
    --layers 4 \
    --color-format "RGBA"
```

### Key Arguments

| Argument | Description |
|----------|-------------|
| `--model` | Model name or local path. Use `Qwen/Qwen-Image-Edit-2509` or later for multiple image support. |
| `--image` | Path(s) to the source image(s) (PNG/JPG, converted to RGB). Can specify multiple images. |
| `--prompt` / `--negative-prompt` | Text description (string). |
| `--cfg-scale` | True classifier-free guidance scale (default: 4.0). Classifier-free guidance is enabled by setting `cfg_scale > 1` and providing a `negative_prompt`. Higher guidance scale encourages images closely linked to the text prompt, usually at the expense of lower image quality. |
| `--guidance-scale` | Guidance scale for guidance-distilled models (default: 1.0, disabled). Unlike classifier-free guidance (`--cfg-scale`), guidance-distilled models take the guidance scale directly as an input parameter. Enabled when `guidance_scale > 1`. Ignored when not using guidance-distilled models. |
| `--num-inference-steps` | Diffusion sampling steps (more steps = higher quality, slower). |
| `--output` | Path to save the generated PNG. For Qwen-Image-Layered, this is used as the filename prefix. |
| `--vae-use-slicing` | Enable VAE slicing for memory optimization. |
| `--vae-use-tiling` | Enable VAE tiling for memory optimization. |
| `--cfg-parallel-size` | Set to `2` to enable CFG Parallel. |
| `--enable-cpu-offload` | Enable CPU offloading for diffusion models. |
| `--layers` | Number of layers to decompose the input image into (Qwen-Image-Layered only). |
| `--color-format` | Output color channel format (`RGB` or `RGBA`). Qwen-Image-Layered uses `RGBA`. |

---

## Acceleration Methods

### Cache Acceleration

vLLM-Omni supports **cache-dit** and **tea-cache** for Qwen-Image models.

#### Cache-DiT

```bash
# Text-to-Image with Cache-DiT
python3 ./examples/offline_inference/text_to_image/text_to_image.py \
    --model Qwen/Qwen-Image \
    --prompt "a cup of coffee on the table" \
    --cache-backend cache_dit

```

Advanced Cache-DiT options:

```bash
python3 ./examples/offline_inference/image_to_image/image_edit.py \
    --model Qwen/Qwen-Image-Edit \
    --image qwen_bear.png \
    --prompt "Edit description" \
    --cache-backend cache_dit \
    --cache-dit-max-continuous-cached-steps 3 \
    --cache-dit-residual-diff-threshold 0.24 \
    --cache-dit-enable-taylorseer
```

#### TeaCache

```bash
# Text-to-Image with TeaCache
python3 ./examples/offline_inference/text_to_image/text_to_image.py \
    --model Qwen/Qwen-Image \
    --prompt "a cup of coffee on the table" \
    --cache-backend tea_cache

```

---

### Ulysses Sequence Parallelism

Distributes computation across GPUs without quality loss. Recommended for high-resolution images (>1536px) with 2–8 GPUs.

```bash
# Text-to-Image with Ulysses SP
python3 ./examples/offline_inference/text_to_image/text_to_image.py \
    --model Qwen/Qwen-Image \
    --prompt "a cup of coffee on the table" \
    --ulysses-degree 4

# Image Editing with Ulysses SP
python3 ./examples/offline_inference/image_to_image/image_edit.py \
    --model Qwen/Qwen-Image-Edit \
    --image qwen_bear.png \
    --prompt "Add a white art board written with colorful text 'vLLM-Omni' on grassland." \
    --output output_image_edit.png \
    --num-inference-steps 50 \
    --cfg-scale 4.0 \
    --ulysses-degree 4
```

---

### Ring-Attention Sequence Parallelism

Ring-based sequence parallelism, suitable for memory-constrained environments with very long sequences.

```bash
# Text-to-Image with Ring-Attention
python3 ./examples/offline_inference/text_to_image/text_to_image.py \
    --model Qwen/Qwen-Image \
    --prompt "a cup of coffee on the table" \
    --ring-degree 4

# Image Editing with Ring-Attention
python3 ./examples/offline_inference/image_to_image/image_edit.py \
    --model Qwen/Qwen-Image-Edit \
    --image qwen_bear.png \
    --prompt "Edit description" \
    --ring-degree 4
```

---

### CFG Parallelism

Splits classifier-free guidance positive/negative branches across 2 GPUs. Particularly effective for image editing with `cfg-scale > 1`.

```bash
# Image Editing with CFG Parallel (2 GPUs)
python3 ./examples/offline_inference/image_to_image/image_edit.py \
    --model Qwen/Qwen-Image-Edit \
    --image qwen_bear.png \
    --prompt "Edit description" \
    --cfg-parallel-size 2 \
    --num-inference-steps 50 \
    --cfg-scale 4.0
```

---

### Tensor Parallelism

Shards model weights across multiple GPUs. Useful for running the 20B model across 2+ GPUs.

```bash
# Text-to-Image with Tensor Parallelism (2 GPUs)
python3 ./examples/offline_inference/text_to_image/text_to_image.py \
    --model Qwen/Qwen-Image \
    --prompt "a cup of coffee on the table" \
    --tensor-parallel-size 2

# Image Editing with Tensor Parallelism
python3 ./examples/offline_inference/image_to_image/image_edit.py \
    --model Qwen/Qwen-Image-Edit \
    --image qwen_bear.png \
    --prompt "Edit description" \
    --tensor-parallel-size 2
```

---

### CPU Offload (Layerwise)

Offloads DiT layers to CPU memory between forward passes. Enables inference on limited VRAM.

```bash
# Text-to-Image with layerwise CPU offload
python3 ./examples/offline_inference/text_to_image/text_to_image.py \
    --model Qwen/Qwen-Image \
    --prompt "a cup of coffee on the table" \
    --enable-layerwise-offload

# Image Editing with layerwise CPU offload
python3 ./examples/offline_inference/image_to_image/image_edit.py \
    --model Qwen/Qwen-Image-Edit \
    --image qwen_bear.png \
    --prompt "Edit description" \
    --enable-layerwise-offload
```



---

### VAE Patch Parallelism

Distributes VAE decode tiling across GPUs, reducing peak VAE memory usage at high resolutions.

```bash
# Text-to-Image with VAE Patch Parallelism
python3 ./examples/offline_inference/text_to_image/text_to_image.py \
    --model Qwen/Qwen-Image \
    --prompt "a cup of coffee on the table" \
    --height 1536 --width 1536 \
    --ulysses-degree 2
    --vae-patch-parallel-size 2
```

> VAE patch parallelism cannot be used alone. It must be used together with other parallelism methods.

---

### Quantization

Qwen-Image and Qwen-Image-2512 support FP8 and INT8 quantization. Qwen-Image-Edit variants do **not** support quantization.

#### FP8


```bash
# Text-to-Image with FP8 quantization
python3 ./examples/offline_inference/text_to_image/text_to_image.py \
    --model Qwen/Qwen-Image \
    --prompt "a cup of coffee on the table" \
    --quantization fp8

# Skip sensitive layers (recommended for better quality)
python3 ./examples/offline_inference/text_to_image/text_to_image.py \
    --model Qwen/Qwen-Image \
    --prompt "a cup of coffee on the table" \
    --quantization fp8 \
    --ignored-layers "img_mlp"
```

#### INT8


```bash
python3 ./examples/offline_inference/text_to_image/text_to_image.py \
    --model Qwen/Qwen-Image \
    --prompt "a cup of coffee on the table" \
    --quantization int8
```

---

## Feature Support Summary

> For detailed features support for Qwen-Image series models in vLLM-Omni, see the [Feature Support Table](https://github.com/vllm-project/vllm-omni/blob/main/docs/user_guide/diffusion_features.md#supported-models)
> For detailed compatibility between features (e.g., combining Cache + SP + CFG-Parallel), see the [Feature Compatibility Guide](https://github.com/vllm-project/vllm-omni/blob/main/docs/user_guide/feature_compatibility.md).

---

## Combining Acceleration Methods

Multiple acceleration methods can be combined for maximum throughput. For example, Cache + Ulysses SP:

```bash
python3 ./examples/offline_inference/text_to_image/text_to_image.py \
    --model Qwen/Qwen-Image \
    --prompt "a cup of coffee on the table" \
    --cache-backend tea_cache \
    --ulysses-degree 4
```

Cache + CFG-Parallel (image editing):

```bash
python3 ./examples/offline_inference/image_to_image/image_edit.py \
    --model Qwen/Qwen-Image-Edit \
    --image qwen_bear.png \
    --prompt "Edit description" \
    --cache-backend cache_dit \
    --cfg-parallel-size 2 \
    --num-inference-steps 50 \
    --cfg-scale 4.0
```

> **Note:** TeaCache and Cache-DiT cannot be used together. 
