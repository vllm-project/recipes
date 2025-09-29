# DeepSeek-V3.2 Usage Guide

## Introduction
[DeepSeek-V3.2](https://huggingface.co/deepseek-ai/DeepSeek-V3.2) is a sparse attention model. The main architecture is similar to DeepSeek-V3.1, but with a sparse attention mechanism.

## Installing vLLM

Need to install vLLM after PR #xxxx is merged.

```bash
uv venv
source .venv/bin/activate
uv pip install vllm --extra-index-url https://wheels.vllm.ai/nightly
```

It also requires installing DeepGEMM with both Hopper and Blackwell support, after PR #xxxx is merged.

```bash
wget https://github.com/vllm-project/vllm/raw/main/tools/install_deepgemm.sh
chmod +x install_deepgemm.sh
TORCH_CUDA_ARCH_LIST="9.0a 10.0a" ./install_deepgemm.sh
```

## Launching DeepSeek-V3.2

### Serving on 8xH200 (or H20) GPUs (141GB × 8)

Using the recommended EP/DP mode:

```bash
vllm serve deepseek-ai/DeepSeek-V3.2 -dp 8 --enable-expert-parallel
```

Using tensor parallel:

```bash
vllm serve deepseek-ai/DeepSeek-V3.2 -tp 8
```

### Serving on 8xB200 GPUs

Same as the above.

Only Hopper and Blackwell data center GPUs are supported for now.

## Performance Tips

1. The kernels are mainly optimized for TP=1, so it is recommended to run this model under EP/DP mode, i.e. DP=8, EP=8, TP=1 as shown above. If you hit any errors or hangs, try tensor parallel instead. Simple tensor parallel works and is more robust, but the performance is not optimal.
2. The default config uses a custom `fp8` kvcache. You can also use `bfloat16` kvcache by specifying `kv_cache_dtype=bfloat16`. The default case allows more tokens to be cached in the kvcache, but incurs additional quantization/dequantization overhead. In general, we recommend using `bfloat16` kvcache for short requests, and `fp8` kvcache for long requests.

If you hit some errors like `CUDA error (flashmla-src/csrc/smxx/mla_combine.cu:201): invalid configuration argument`, it might be caused by too large batchsize. Try with `--max-num-seqs 256` or smaller (the default is 1024).

For other usage tips, such as enabling or disabling thinking mode, please refer to the DeepSeek-V3.1 Usage Guide.
