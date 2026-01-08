# NVIDIA Nemotron-Nano-12B-v2-VL User Guide

This guide describes how to run Nemotron-Nano-12B-v2-VL series on the targeted accelerated stack.

## Installing vLLM

* vLLM 0.11.0 does not include Nemotron-Nano-12B-v2-VL, so either [install from source](https://docs.vllm.ai/en/v0.6.0/getting_started/installation.html) or refer to [this](https://hub.docker.com/layers/vllm/vllm-openai/nightly-8bff831f0aa239006f34b721e63e1340e3472067/images/sha256-ef112680ed30e4b9d7bf794dcda4abd829e9405a73e013f9e046658cf22d0577) nightly build
```bash
docker pull vllm/vllm-openai:nightly-8bff831f0aa239006f34b721e63e1340e3472067
```

For DGX Spark, container relase is avaiable 
https://catalog.ngc.nvidia.com/orgs/nvidia/containers/vllm?version=25.12.post1-py3

```bash
docker pull nvcr.io/nvidia/vllm:25.12.post1-py3
```

## Serving Nemotron-Nano-12B-v2-VL
### Server:
The following command will launch an inference server on 1 GPU.

Notes:
* Examples are using [BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16) precision model. We encourage you to try [FP8](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-FP8) and [NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-NVFP4-QAD) as well!
* You can set `--max-model-len <len>` ([doc](https://docs.vllm.ai/en/latest/configuration/engine_args.html#-max-model-len)) to preserve memory. Model is trained on a context length of ~131K, but unless the use-case is long context videos, a smaller context would fit as-well.
* You can set `--allowed-local-media-path <root>` ([doc](https://docs.vllm.ai/en/latest/configuration/engine_args.html#-allowed-local-media-path)) to limit the accessibility of local files.

#### Efficient Video Sampling (EVS)
* You can set `--video-pruning-rate <fraction>` to tweak video compression. Read more about EVS on [arXiv](https://arxiv.org/abs/2510.14624).

```bash
export VLLM_VIDEO_LOADER_BACKEND=opencv
export CHECKPOINT_PATH="nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16"
export CUDA_VISIBLE_DEVICES=0

python3 -m vllm.entrypoints.openai.api_server \
   --model ${CHECKPOINT_PATH} \
   --trust-remote-code \
   --media-io-kwargs '{"video": {"fps": 2, "num_frames": 128} }' \
   --max-model-len 131072 \
   --data-parallel-size 1 \
   --port 5566 \
   --allowed-local-media-path / \
   --video-pruning-rate 0.75 \
   --served-model-name "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16"
```

### Client (bash):
```bash
curl -X 'POST' \
  'http://127.0.0.1:5566/v1/chat/completions' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "model": "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16",
  "messages": [{"role": "user", "content": [{"type": "text", "text": "Describe the video."}, {"type": "video_url", "video_url": {"url": "file:///path/to/video.mp4"}}]}]
}'
```

### Client (Python):
```python
from openai import OpenAI
client = OpenAI(
    base_url="http://localhost:5566/v1",
    api_key="<ignored>",
)

completion = client.chat.completions.create(
    model="nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16",
    messages=[
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "Describe the video."
          },
          {
            "type": "video_url",
            "video_url": {
              "url": "file:///path/to/video.mp4"
            }
          }
        ]
      }
    ],
)

print(completion.choices[0].message.content)

completion = client.chat.completions.create(
    model="nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16",
    messages=[
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "Describe the image."
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "file:///path/to/image.jpg"
            }
          }
        ]
      }
    ],
)

print(completion.choices[0].message.content)
```

### vLLM `LLM` API

Notes:
* Examples are using [BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16) precision model. We encourage you to try [FP8](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-FP8) and [NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-NVFP4-QAD) as well!
* You can set `max_model_len <len>` ([doc](https://docs.vllm.ai/en/latest/configuration/engine_args.html#-max-model-len)) to preserve memory. Model is trained on a context length of ~131K, but unless the use-case is long context videos, a smaller context would fit as-well.
* You can set `allowed_local_media_path <root>` ([doc](https://docs.vllm.ai/en/latest/configuration/engine_args.html#-allowed-local-media-path)) to limit the accessibility of local files.

#### Efficient Video Sampling (EVS)
* You can set `video_pruning_rate <fraction>` to tweak video compression. Read more about EVS on [arXiv](https://arxiv.org/abs/2510.14624).


#### Usage with image path
```python
from vllm import LLM, SamplingParams

model_path = "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16"

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Describe the image.",
            },
            {
                "type": "image_url",
                "image_url": {"url": f"file:///path/to/image.jpg"},
            },
        ],
    },
]

llm = LLM(
    model_path,
    trust_remote_code=True,
    max_model_len=2**17,  # 131,072
    # '/' is too permissive and used for example; use a specific directory instead
    allowed_local_media_path="/",
)

outputs = llm.chat(
    messages, 
    sampling_params=SamplingParams(temperature=0, max_tokens=1024),
    # configure the number of tiles from 1 to (default) 12
    # note: for videos, the number of tiles must be 1
    mm_processor_kwargs=dict(max_num_tiles=12),
)

for o in outputs:
    print(o.outputs[0].text)
```

#### Usage with video path
* See Efficient Video Sampling (EVS): affects videos only, defines how much of the video tokens to prune
```python
import os
os.environ["VLLM_VIDEO_LOADER_BACKEND"] = "opencv"

from vllm import LLM, SamplingParams

model_path = "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16"

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Describe the video.",
            },
            {
                "type": "video_url",
                "video_url": {"url": f"file:///path/to/video.mp4"},
            },
        ],
    },
]

# Efficient Video Sampling (EVS): affects videos only, defines how much of the video tokens to prune
# To turn EVS off, use `video_pruning_rate = 0`
video_pruning_rate = 0.75

llm = LLM(
    model_path,
    trust_remote_code=True,
    video_pruning_rate=video_pruning_rate,
    max_model_len=2**17,  # 131,072
    # '/' is too permissive and used for example; use a specific directory instead
    allowed_local_media_path="/",
    media_io_kwargs=dict(video=dict(fps=2, num_frames=128)),
)

outputs = llm.chat(
    messages, sampling_params=SamplingParams(temperature=0, max_tokens=1024)
)

for o in outputs:
    print(o.outputs[0].text)
```

#### Usage with video tensors and custom sampling
```python
from vllm import LLM, SamplingParams
import decord
import numpy as np
from transformers.video_utils import VideoMetadata
from transformers import AutoTokenizer

def sample_video_frames(video_path_local, fps=0, nframe=0, nframe_max=-1):
    """
    Sample frames from a video and return them as a numpy array along with metadata.

    Args:
        video_path_local: Path to the video file
        fps: Target frames per second for sampling (if > 0, uses fps-based sampling)
        nframe: Number of frames to sample (used if fps <= 0)
        nframe_max: Maximum number of frames to sample

    Returns:
        tuple: (images, metadata)
        - images: A numpy array of the sampled frame images.
        - metadata: VideoMetadata dataclass containing info about the sampled frames:
            - total_num_frames: Number of sampled frames
            - fps: Effective frame rate of the sampled frames
            - duration: Duration covered by the sampled frames (in seconds)
            - video_backend: Backend used for video processing ('opencv_dynamic')
    """

    vid = decord.VideoReader(video_path_local)
    total_frames = len(vid)
    video_fps = vid.get_avg_fps()
    total_duration = total_frames / max(1e-6, video_fps)

    if fps > 0:
        required_frames = int(total_duration * fps)
        desired_frames = max(1, required_frames)
        if nframe_max > 0 and desired_frames > nframe_max:
            desired_frames = nframe_max
        if desired_frames >= total_frames:
            indices = list(range(total_frames))
        elif desired_frames == 1:
            indices = [0]  # Always use first frame for single frame sampling
        else:
            # Generate evenly spaced indices and ensure uniqueness
            raw_indices = np.linspace(0, total_frames - 1, desired_frames)
            indices = list(np.unique(np.round(raw_indices).astype(int)))
    else:
        desired_frames = max(1, int(nframe) if nframe and nframe > 0 else 8)
        if nframe_max > 0 and desired_frames > nframe_max:
            desired_frames = nframe_max
        if desired_frames >= total_frames:
            indices = list(range(total_frames))
        elif desired_frames == 1:
            indices = [0]  # Always use first frame for single frame sampling
        else:
            # Generate evenly spaced indices and ensure uniqueness
            raw_indices = np.linspace(0, total_frames - 1, desired_frames)
            indices = list(np.unique(np.round(raw_indices).astype(int)))

    images = vid.get_batch(indices).asnumpy()

    # Calculate timestamps for each sampled frame
    timestamps = [float(idx) / video_fps for idx in indices]

    # Calculate metadata for the sampled frames
    sampled_num_frames = len(indices)

    # Duration is the time span from first to last frame
    if len(timestamps) > 1:
        sampled_duration = timestamps[-1] - timestamps[0]
        sampled_fps = (
            (sampled_num_frames - 1) / sampled_duration if sampled_duration > 0 else 1.0
        )
    else:
        # Single frame case
        sampled_duration = None
        sampled_fps = None

    metadata = VideoMetadata(
        total_num_frames=sampled_num_frames,
        fps=sampled_fps,
        duration=sampled_duration,
        frames_indices=indices,
        video_backend="opencv_dynamic",
    )

    return images, metadata


def main():
    model_path = "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16"
    video_path = "/path/to/video.mp4"

    examples = {
        "8fps_max128frames": dict(fps=8, nframe_max=128),
        "2fps": dict(fps=2),
        "16frames": dict(nframe=16),
    }

    examples = {
        k: sample_video_frames_to_data_urls(video_path, **kwargs)
        for k, kwargs in examples.items()
    }

    for k, (vid, meta) in examples.items():
        print(f"key={k}, {vid.shape=}, {vid.max().item()=}, {vid.min().item()=}")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Describe the video.",
                },
                # Note: we add a placeholder of type 'video' so that the tokenizer will insert <video> when it is tokenizing the prompt
                {
                    "type": "video",
                    "text": None,
                }
            ],
        },
    ]

    # Efficient Video Sampling (EVS): affects videos only, defines how much of the video tokens to prune
    # To turn EVS off, use `video_pruning_rate = 0`
    video_pruning_rate = 0.75

    llm = LLM(
        model_path,
        trust_remote_code=True,
        video_pruning_rate=video_pruning_rate,
        max_model_len=2**17,  # 131,072
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    print(f"Prompt: {prompt}")

    outputs = llm.generate(
        [
            {
                "prompt": prompt,
                "multi_modal_data": {"video": (vid, metadata)},
            }
            for (vid, metadata) in examples.values()
        ],
        sampling_params=SamplingParams(temperature=0, max_tokens=1024),
    )

    for k, o in zip(examples.keys(), outputs):
        print(k)
        print(o.outputs[0].text)
        print("-" * 10)


if __name__ == "__main__":
    main()
```
