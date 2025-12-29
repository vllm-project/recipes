# Qwen-Image Usage Guide

Qwen-Image models include the following models:

+ Qwen-Image
+ Qwen-Image-Edit
+ Qwen-Image-Layered

These models share the same DiT transformer core; hence, the following acceleration methods (e.g., cache and ulysses parallelism) are applicable across the entire series.

This guide describes how to run Qwen-Image-Edit.

[Qwen-Image-Edit](https://huggingface.co/Qwen/Qwen-Image-Edit) is the image editing version of Qwen-Image. Built upon our 20B Qwen-Image model, Qwen-Image-Edit successfully extends Qwen-Image's unique text rendering capabilities to image editing tasks, enabling precise text editing. Furthermore, Qwen-Image-Edit simultaneously feeds the input image into Qwen2.5-VL (for visual semantic control) and the VAE Encoder (for visual appearance control), achieving capabilities in both semantic and appearance editing.

## Installation

```bash
# Clone and install vllm-omni
git clone https://github.com/vllm-project/vllm-omni.git
cd vllm-omni
uv venv
source .venv/bin/activate
uv pip install -e . vllm==0.12.0
```

## Usage

```bash
cd vllm-omni
python3 ./examples/offline_inference/image_to_image/image_edit.py \
  --image qwen_bear.png \
  --prompt "Let this mascot dance under the moon, surrounded by floating stars and poetic bubbles such as 'Be Kind'" \
  --output output_image_edit.png \
  --num_inference_steps 50 \
  --cfg_scale 4.0 \
```

For multiple image inputs, use `Qwen/Qwen-Image-Edit-2509` or `Qwen/Qwen-Image-Edit-2511`:


```bash
# Qwen-Image-Edit-2511 example
python3 ./examples/offline_inference/image_to_image/image_edit.py \
    --model Qwen/Qwen-Image-Edit-2511 \
    --image qwen_bear.png \
    --prompt "Add a white art board written with colorful text 'vLLM-Omni' on grassland. Add a paintbrush in the bear's hands. position the bear standing in front of the art board as if painting" \
    --output output_image_edit.png \
    --num_inference_steps 50 \
    --cfg_scale 4.0
```

For Qwen-Image-Layered:

```bash
python3 ./examples/offline_inference/image_to_image/image_edit.py \
    --model Qwen/Qwen-Image-Layered \
    --image 1.png \
    --prompt "" \
    --output layered \
    --num_inference_steps 50 \
    --cfg_scale 4.0 \
    --layers 4 \
    --color-format "RGBA"
```

Key arguments:

- `--model`: model name or path. Use `Qwen/Qwen-Image-Edit-2509` or later for multiple image support.
- `--image`: path(s) to the source image(s) (PNG/JPG, converted to RGB). Can specify multiple images.
- `--prompt` / `--negative_prompt`: text description (string).
- `--cfg_scale`: true classifier-free guidance scale (default: 4.0). Classifier-free guidance is enabled by setting cfg_scale > 1 and providing a negative_prompt. Higher guidance scale encourages images closely linked to the text prompt, usually at the expense of lower image quality.
- `--guidance_scale`: guidance scale for guidance-distilled models (default: 1.0, disabled). Unlike classifier-free guidance (--cfg_scale), guidance-distilled models take the guidance scale directly as an input parameter. Enabled when guidance_scale > 1. Ignored when not using guidance-distilled models.
- `--num_inference_steps`: diffusion sampling steps (more steps = higher quality, slower).
- `--output`: path to save the generated PNG.


## Acceleration methods

### Cache-Dit

```bash
python3 ./examples/offline_inference/image_to_image/image_edit.py \
    --model Qwen/Qwen-Image-Edit-2511 \
    --image qwen_bear.png \
    --prompt "Add a white art board written with colorful text 'vLLM-Omni' on grassland. Add a paintbrush in the bear's hands. position the bear standing in front of the art board as if painting" \
    --output output_image_edit.png \
    --num_inference_steps 50 \
    --cfg_scale 4.0 \
    --cache_backend cache_dit
```

### Ulysses Parallelism

```bash
python3 ./examples/offline_inference/image_to_image/image_edit.py \
    --model Qwen/Qwen-Image-Edit-2511 \
    --image qwen_bear.png \
    --prompt "Add a white art board written with colorful text 'vLLM-Omni' on grassland. Add a paintbrush in the bear's hands. position the bear standing in front of the art board as if painting" \
    --output output_image_edit.png \
    --num_inference_steps 50 \
    --cfg_scale 4.0 \
    --ulysses_degree 4
```
