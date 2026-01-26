# Stable Audio Open Usage Guide

This guide provides instructions for running Stable Audio Open text-to-audio generation using vLLM-Omni.

## Supported Models

- **stabilityai/stable-audio-open-1.0**: Text-to-Audio generation (44.1kHz, up to ~47s audio)

## Installing vLLM-Omni

```bash
uv venv
source .venv/bin/activate
uv pip install vllm==0.12.0
uv pip install git+https://github.com/vllm-project/vllm-omni.git
```

For audio file saving, install one of these packages:

```bash
uv pip install soundfile  # Recommended
# or
uv pip install scipy
```

The CLI examples below are from the vLLM-Omni repo. If you want to run them directly, clone that repo and run the scripts from its `examples/offline_inference` directory.

## Text-to-Audio Generation

### Basic Usage

```python
import torch
import soundfile as sf
from vllm_omni.entrypoints.omni import Omni

omni = Omni(model="stabilityai/stable-audio-open-1.0")

generator = torch.Generator(device="cuda").manual_seed(42)

audio = omni.generate(
    "The sound of a dog barking",
    negative_prompt="Low quality.",
    generator=generator,
    guidance_scale=7.0,
    num_inference_steps=100,
    extra={
        "audio_start_in_s": 0.0,
        "audio_end_in_s": 10.0,
    },
)

# Save audio output
audio_data = audio[0].cpu().float().numpy().T  # [samples, channels]
sf.write("output.wav", audio_data, 44100)
```

### CLI Usage

```bash
python examples/offline_inference/text_to_audio/text_to_audio.py \
  --model stabilityai/stable-audio-open-1.0 \
  --prompt "The sound of a dog barking" \
  --audio_length 10.0 \
  --num_inference_steps 100 \
  --guidance_scale 7.0 \
  --output dog_barking.wav
```

### More Examples

```bash
# Generate a piano melody
python examples/offline_inference/text_to_audio/text_to_audio.py \
  --prompt "A piano playing a gentle melody" \
  --audio_length 15.0 \
  --output piano_melody.wav

# Generate ambient sounds with negative prompt
python examples/offline_inference/text_to_audio/text_to_audio.py \
  --prompt "Thunder and rain sounds" \
  --negative_prompt "Low quality, distorted" \
  --audio_length 20.0 \
  --output thunder_rain.wav

# Generate multiple waveforms
python examples/offline_inference/text_to_audio/text_to_audio.py \
  --prompt "A bird singing in the forest" \
  --num_waveforms 3 \
  --output bird_singing.wav
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `audio_start_in_s` | 0.0 | Audio start time in seconds |
| `audio_end_in_s` | 10.0 | Audio end time in seconds |
| `audio_length` | 10.0 | Audio duration (CLI convenience, sets end time) |
| `num_inference_steps` | 100 | Number of denoising steps |
| `guidance_scale` | 7.0 | Classifier-free guidance scale |
| `negative_prompt` | "Low quality." | Text describing unwanted audio characteristics |
| `num_waveforms` | 1 | Number of audio samples to generate per prompt |
| `sample_rate` | 44100 | Output sample rate in Hz |
| `seed` | 42 | Random seed for reproducibility |

## Notes

- **Maximum audio length**: 47 seconds for stable-audio-open-1.0.
- **Output format**: Stereo audio at 44.1kHz sample rate.
- **Inference steps**: Higher `num_inference_steps` produces better quality but takes longer. The diffusers default is 200; vLLM-Omni example uses 100 for faster generation.
- **Negative prompts**: Use to guide the model away from undesirable characteristics (e.g., "Low quality, distorted").
- **Model size**: Approximately 1.2 billion parameters.

## Limitations

- **No realistic vocals**: The model cannot generate realistic singing or speech.
- **English only**: Trained on English descriptions; performance degrades with other languages.
- **Sound effects over music**: Better at generating sound effects than complex music.
- **Prompt engineering**: May require experimentation with prompts for optimal results.

## License

Stable Audio Open is released under the [Stability AI Community License](https://huggingface.co/stabilityai/stable-audio-open-1.0/blob/main/LICENSE). Commercial use requires a separate license from Stability AI.

## Additional Resources

- [vLLM-Omni Text-to-Audio Example](https://github.com/vllm-project/vllm-omni/blob/main/examples/offline_inference/text_to_audio/text_to_audio.py)
- [Stable Audio Open Model Card](https://huggingface.co/stabilityai/stable-audio-open-1.0)
- [Stability AI](https://stability.ai/)
