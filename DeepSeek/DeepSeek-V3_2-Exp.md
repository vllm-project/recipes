# DeepSeek-V3.2-Exp Usage Guide

## Introduction
[DeepSeek-V3.2-Exp](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp) is a sparse attention model. The main architecture is similar to DeepSeek-V3.1, but with a sparse attention mechanism.

## Installing vLLM

```bash
uv pip install vllm --extra-index-url https://wheels.vllm.ai/nightly
uv pip install git+https://github.com/deepseek-ai/DeepGEMM.git@v2.1.1.post3 --no-build-isolation # Other versions may also work. We recommend using the latest released version from https://github.com/deepseek-ai/DeepGEMM/releases
```

Note: DeepGEMM is used in two places: MoE and MQA logits computation. It is necessary for MQA logits computation. If you want to disable the MoE part, you can set `VLLM_USE_DEEP_GEMM=0` in the environment variable. Some users reported that the performance is better with `VLLM_USE_DEEP_GEMM=0`, e.g. on H20 GPUs. It might be also beneficial to disable DeepGEMM if you want to skip the long warmup.

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

## Accuracy Benchmarking:

```bash
lm-eval --model local-completions --tasks gsm8k   --model_args model=deepseek-ai/DeepSeek-V3.2-Exp,base_url=http://127.0.0.1:8000/v1/completions,num_concurrent=100,max_retries=3,tokenized_requests=False
```

Results:

```bash
local-completions (model=deepseek-ai/DeepSeek-V3.2-Exp,base_url=http://127.0.0.1:8000/v1/completions,num_concurrent=100,max_retries=3,tokenized_requests=False), gen_kwargs: (None), limit: None, num_fewshot: None, batch_size: 1
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9591|±  |0.0055|
|     |       |strict-match    |     5|exact_match|↑  |0.9591|±  |0.0055|
```

GSM8K score `0.9591` is pretty good!

And then we can use `num_fewshot=20` to increase the context length, testing if the model can handle longer context:

```bash
lm-eval --model local-completions --tasks gsm8k   --model_args model=deepseek-ai/DeepSeek-V3.2-Exp,base_url=http://127.0.0.1:8000/v1/completions,num_concurrent=100,max_retries=3,tokenized_requests=False --num_fewshot 20
```

Results:

```bash
local-completions (model=deepseek-ai/DeepSeek-V3.2-Exp,base_url=http://127.0.0.1:8000/v1/completions,num_concurrent=100,max_retries=3,tokenized_requests=False), gen_kwargs: (None), limit: None, num_fewshot: 20, batch_size: 1
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|    20|exact_match|↑  |0.9538|±  |0.0058|
|     |       |strict-match    |    20|exact_match|↑  |0.9530|±  |0.0058|
```

GSM8K score `0.9538` is also pretty good!

## Performance Tips

1. The kernels are mainly optimized for TP=1, so it is recommended to run this model under EP/DP mode, i.e. DP=8, EP=8, TP=1 as shown above. If you hit any errors or hangs, try tensor parallel instead. Simple tensor parallel works and is more robust, but the performance is not optimal.
2. The default config uses a custom `fp8` kvcache. You can also use `bfloat16` kvcache by specifying `kv_cache_dtype=bfloat16`. The default case allows more tokens to be cached in the kvcache, but incurs additional quantization/dequantization overhead. In general, we recommend using `bfloat16` kvcache for short requests, and `fp8` kvcache for long requests.

If you hit some errors like `CUDA error (flashmla-src/csrc/smxx/mla_combine.cu:201): invalid configuration argument`, it might be caused by too large batchsize. Try with `--max-num-seqs 256` or smaller (the default is 1024).

For other usage tips, such as enabling or disabling thinking mode, please refer to the DeepSeek-V3.1 Usage Guide.

## Additional Resources

- [An end-to-end tutorial (Jupyter Notebook)](https://github.com/vllm-project/recipes/blob/main/DeepSeek/DeepSeek_v3_2_vLLM_getting_started_guide.ipynb)



## AMD GPU Support
Recommended approaches by hardware type are:

MI300X/MI325X/MI355X 

Please follow the steps here to install and run DeepSeek-V3.2-Exp models on AMD MI300X/MI325X/MI355X GPU.

### Step 1: Installing vLLM (AMD ROCm Backend: MI300X, MI325X, MI355X) 
 > Note: The vLLM wheel for ROCm requires Python 3.12, ROCm 7.0, and glibc >= 2.35. If your environment does not meet these requirements, please use the Docker-based setup as described in the [documentation](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/#pre-built-images).  
 ```bash 
 uv venv 
 source .venv/bin/activate 
 uv pip install vllm --extra-index-url https://wheels.vllm.ai/rocm/
 ```


### Step 2: Start the vLLM server

Run the vllm online serving


```bash
SAFETENSORS_FAST_GPU=1 \
VLLM_ROCM_USE_AITER=1 \
vllm serve deepseek-ai/DeepSeek-V3.2-Exp \
  --tensor-parallel-size 8 \
  --trust-remote-code
```


### Step 3: Run Benchmark
Open a new terminal and run the following command to execute the benchmark script.
```bash
vllm bench serve \
  --model deepseek-ai/DeepSeek-V3.2-Exp \
  --dataset-name random \
  --random-input-len 1000 \
  --random-output-len 200 \
  --request-rate 1 \
  --num-prompts 4 \
  --ignore-eos
```

