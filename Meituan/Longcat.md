# Longcat Usage Guide

LongCat models include those model below:

+ LongCat-Image-Edit

[LongCat-Image-Edit](https://huggingface.co/meituan-longcat/LongCat-Image-Edit) is the image editing version of Longcat-Image. LongCat-Image-Edit supports bilingual (Chinese-English) editing, achieves state-of-the-art performance among open-source image editing models, delivering leading instruction-following and image quality with superior visual consistency.

This guide describes how to run LongCat-Image-Edit.

## Installation

```bash
# Clone and install vllm-omni
git clone https://github.com/vllm-project/vllm-omni.git
cd vllm-omni
uv venv
source .venv/bin/activate
uv pip install -e . vllm==0.12.0

# Update xformers to the latest version
uv pip install -U xformers --index-url https://download.pytorch.org/whl/cu128

# Update diffusers to the latest version
git clone https://github.com/huggingface/diffusers.git
cd diffusers
uv pip install -e .
```

## Usage

```bash
cd vllm-omni
python3 ./examples/offline_inference/image_to_image/image_edit.py \
    --image qwen_bear.png \
    --prompt "Add a white art board written with colorful text 'vLLM-Omni' on grassland. Add a paintbrush in the bear's hands. position the bear standing in front of the art board as if painting" \
    --output output_image_edit.png \
    --num_inference_steps 50 \
    --guidance_scale 4.5 \
    --seed 42 \
    --model meituan-longcat/LongCat-Image-Edit \
    --cache_backend cache_dit \
    --cache_dit_max_continuous_cached_steps 2
```

