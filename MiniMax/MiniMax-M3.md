# MiniMax-M3 Series Usage Guide

[MiniMax-M3](https://huggingface.co/MiniMaxAI/MiniMax-M3) is a multimodal (vision-language)
Mixture-of-Experts model from [MiniMax](https://www.minimax.io/) (427B total parameters,
`minimax_m3_vl` architecture, arxiv:2606.13392). It targets agentic coding, tool use, and
long-context reasoning, and adds image/video understanding on top of the M2 series.

This guide covers serving M3 on **AMD Instinct (ROCm)**. It was written and verified on
**8x AMD Instinct MI308X (gfx942 / CDNA3)** with the MXFP8 checkpoint; MI300X / MI325X share the
same gfx942 architecture and should be compatible, but were not separately verified here.

## Supported Models

- [MiniMaxAI/MiniMax-M3](https://huggingface.co/MiniMaxAI/MiniMax-M3) (BF16 / FP16 weights)
- [MiniMaxAI/MiniMax-M3-MXFP8](https://huggingface.co/MiniMaxAI/MiniMax-M3-MXFP8) (MXFP8 checkpoint; the config verified below)

License: `other` (MiniMax custom license); accept it on the model page before downloading.

## System Requirements

- OS: Linux with ROCm
- ROCm driver and `amd-smi` on the host; `/dev/kfd` and `/dev/dri` present
- Docker
- Supported AMD GPUs: gfx942 (MI300X / MI325X / MI308X). Verified here on 8x MI308X.

> **Precision note (gfx942).** gfx942 has no native MXFP8 MoE kernel in vLLM 0.25.0, so the MXFP8
> checkpoint is dequantized to BF16 once at load time and the MoE runs in BF16 (weights stay
> MXFP8-compressed in VRAM; compute is BF16). GEMM uses `EmulationMxfp8LinearKernel`. Treat these as
> BF16-compute numbers, not native-MXFP8.

## Launching M3 with vLLM on AMD GPU (ROCm)

Image used for verification: `vllm/vllm-openai-rocm:v0.25.0`
(`sha256:20f7f877b3641595119b36f9096087325ca063c7ba1972788ab9bbbe73f43964`).

Point the volume mount at your local checkpoint. The `--group-add` below resolves the `/dev/kfd`
group id automatically with `$(stat -c '%g' /dev/kfd)`.

- **TP8 (8x MI308X), verified**

```bash
docker run -d --name mm3-vllm \
  --network host --ipc host --shm-size 32g \
  --device /dev/kfd --device /dev/dri --group-add "$(stat -c '%g' /dev/kfd)" \
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  -v /path/to/MiniMax-M3-MXFP8:/models/MiniMax-M3-MXFP8:ro \
  -e VLLM_USE_V1=1 \
  --entrypoint vllm vllm/vllm-openai-rocm:v0.25.0 serve /models/MiniMax-M3-MXFP8 \
    --served-model-name minimax-m3 \
    --port 8000 --trust-remote-code \
    --tensor-parallel-size 8 \
    --block-size 128 \
    --attention-backend TRITON_ATTN \
    --mm-encoder-attn-backend ROCM_AITER_FA \
    --mm-encoder-tp-mode data \
    --tool-call-parser minimax_m3 --reasoning-parser minimax_m3 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 16384
```

Notes on the flags, which differ from the M2 series on ROCm:

- **`--attention-backend TRITON_ATTN`.** M3's sparse "lightning indexer" attention is served through
  the Triton backend here. The M2 ROCm recipe uses `ROCM_AITER_FA`; AITER flash-attn was not verified
  against M3's indexer in this setup, so Triton is the known-good choice. `--mm-encoder-attn-backend
  ROCM_AITER_FA` is still used for the vision encoder.
- **Pure TP8 works.** Unlike the M2 guide (which notes pure TP8 is unsupported), M3 served cleanly at
  `--tensor-parallel-size 8` in this setup. DP+EP (`--data-parallel-size 8 --enable-expert-parallel`)
  also runs; at matched concurrency, TP8 was the stronger throughput/latency point on this hardware.
- **`--max-model-len` must cover input + output tokens combined**, not just input.

## Performance Metrics

Measured on 8x MI308X (gfx942), MXFP8 checkpoint, BF16 compute, vLLM 0.25.0:

- **Accuracy:** GSM8K (lm-eval `local-completions`, 5-shot, full `main/test` n=1319):
  **0.9136 flexible / 0.9121 strict**. Raw `/v1/completions`, no chat template; a chat/reasoning
  harness scores higher and would be reported separately.
- **Throughput:** ~**739 output tok/s** at 1k-in / 1k-out, concurrency 32 (`vllm bench serve`,
  `--dataset-name random`, prefix caching off). This is a compile-disabled, untuned-MoE baseline
  (see Using Tips), so it is a floor, not a tuned ceiling.

## Using Tips

- **Long context (> ~32k) needs `--skip-mm-profiling`.** M3 is multimodal, and vLLM profiles a
  max-size video item against an encoder budget that scales with `--max-model-len`; without the flag
  this can deadlock near 128k. Raise `--max-model-len` (covering input + output) and add
  `--skip-mm-profiling` for long-context serving.
- **FP8-dynamic checkpoint needs a patched vLLM.** The
  [EmbeddedLLM/MiniMax-M3-FP8-dynamic](https://huggingface.co/EmbeddedLLM/MiniMax-M3-FP8-dynamic)
  (compressed-tensors W8A8) checkpoint runs a native-FP8 path on gfx942, but stock vLLM 0.25.0 serves
  it as garbage: M3 uses `hidden_act="swigluoai"` with non-default `swiglu_alpha` / `swiglu_beta`, and
  the W8A8-FP8 MoE path ignored them. See vLLM
  [PR #46845](https://github.com/vllm-project/vllm/pull/46845) and the checkpoint's HF discussion.
  This is the M3 analog of the M2 series' "corrupted output -> upgrade vLLM" note.
- **Open tuning (not yet A/B-verified on M3).** The M2 ROCm recipe enables `VLLM_ROCM_USE_AITER=1`
  (plus `VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1`) and `--compilation-config
  '{"mode":3,"pass_config":{"fuse_minimax_qk_norm":true}}'`. The baseline above runs with
  torch.compile disabled and AITER MoE off, so those are the obvious next levers to test on M3 for
  higher throughput. The compile path is gated by `VLLM_USE_BREAKABLE_CUDAGRAPH`: vLLM auto-enables it
  (`=1`) here, which is what disables `torch.compile`, so setting `-e VLLM_USE_BREAKABLE_CUDAGRAPH=0`
  opts back into the compiled path. It is left out of the verified command above because M3's
  full-cudagraph capture has aborted at `decode_query_len > 0` in offline profiling; confirm it serves
  before relying on it.
