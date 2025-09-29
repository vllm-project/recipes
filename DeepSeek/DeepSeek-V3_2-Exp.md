# DeepSeek-V3.2-Exp Usage Guide

## Introduction
[DeepSeek-V3.2-Exp](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp) is a sparse attention model. The main architecture is similar to DeepSeek-V3.1, but with a sparse attention mechanism.

## Installing vLLM

```bash
pip install https://wheels.vllm.ai/dsv32/vllm-0.10.2rc3.dev369%2Bgaeee9295b-cp38-abi3-linux_x86_64.whl
pip install https://wheels.vllm.ai/dsv32/deep_gemm-2.1.0%2B594953a-cp312-cp312-linux_x86_64.whl
```

## Launching DeepSeek-V3.2-Exp

### Serving on 8xH200 (or H20) GPUs (141GB × 8)

Using the recommended EP/DP mode:

```bash
vllm serve deepseek-ai/DeepSeek-V3.2-Exp -dp 8 --enable-expert-parallel
```

Using tensor parallel:

```bash
vllm serve deepseek-ai/DeepSeek-V3.2-Exp -tp 8
```

### Serving on 8xB200 GPUs

Same as the above.

Only Hopper and Blackwell data center GPUs are supported for now.

## Performance Tips

1. The kernels are mainly optimized for TP=1, so it is recommended to run this model under EP/DP mode, i.e. DP=8, EP=8, TP=1 as shown above. If you hit any errors or hangs, try tensor parallel instead. Simple tensor parallel works and is more robust, but the performance is not optimal.
2. The default config uses a custom `fp8` kvcache. You can also use `bfloat16` kvcache by specifying `kv_cache_dtype=bfloat16`. The default case allows more tokens to be cached in the kvcache, but incurs additional quantization/dequantization overhead. In general, we recommend using `bfloat16` kvcache for short requests, and `fp8` kvcache for long requests.

If you hit some errors like `CUDA error (flashmla-src/csrc/smxx/mla_combine.cu:201): invalid configuration argument`, it might be caused by too large batchsize. Try with `--max-num-seqs 256` or smaller (the default is 1024).

For other usage tips, such as enabling or disabling thinking mode, please refer to the DeepSeek-V3.1 Usage Guide.
