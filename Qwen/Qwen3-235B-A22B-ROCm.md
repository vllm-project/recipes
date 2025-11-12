# Qwen3-235B-A22B Usage Guide

[Qwen3-235B-A22B](https://huggingface.co/Qwen/Qwen3-235B-A22B) is an advanced large language model created by the Qwen team from Alibaba Cloud. This is a guide on running the model on MI355 GPUs with vLLM.

## Preparing environment
### Launching docker container
First prepare the docker environment following the guide in [ROCm docker setup](https://docs.vllm.ai/en/stable/getting_started/installation/gpu.html#set-up-using-docker).
All the operations in the next will be performed inside the container you launched.

### Installing vLLM and AITER
We suggest to install the latest vLLM and AITER to leverage all the optimizations available on ROCm plarforms.

```bash
# inside the container
pip uninstall -y aiter vllm
git clone https://github.com/ROCm/aiter.git
cd aiter
git submodule sync && git submodule update --init --recursive
python3 setup.py install

cd .. && git clone https://github.com/vllm-project/vllm.git
cd vllm
PYTORCH_ROCM_ARCH="gfx950" python3 setup.py develop
```

## Launching Qwen3-235B-A22B with vLLM
Let's first depoly the model in parallelism of TP8 + EP8.

### Serving on 8xMI355 GPUs

**BF16 Model**

```bash
#!/bin/bash
export SAFETENSORS_FAST_GPU=1
export VLLM_ROCM_USE_AITER=1

vllm serve Qwen/Qwen3-235B-A22B \
    --tensor-parallel-size 8 \
    --max-num-batched-tokens 32768 \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --gpu_memory_utilization 0.9 \
    --enable-expert-parallel \
    --async-scheduling
```

**FP8 Model**

```bash
#!/bin/bash
export SAFETENSORS_FAST_GPU=1
export VLLM_ROCM_USE_AITER=1

vllm serve Qwen/Qwen3-235B-A22B-FP8 \
    --tensor-parallel-size 8 \
    --max-num-batched-tokens 32768 \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --gpu_memory_utilization 0.9 \
    --enable-expert-parallel \
    --async-scheduling
```

## Performance Metrics

### Benchmarking
We used the following script to benchmark the performance:

```bash
vllm bench serve \
    --model Qwen/Qwen3-235B-A22B-FP8 \
    --dataset-name random \
    --random-input-len 1024 \
    --random-output-len 1024 \
    --max-concurrency 128 \
    --num-prompts 256 \
    --percentile-metrics ttft,tpot,itl,e2el \
    --ignore-eos \
    --seed 123
```
### Accuracy test
We verified the lm_eval accuracy with command:
```bash
lm_eval \
--model local-completions \
--tasks gsm8k \
--model_args model=Qwen/Qwen3-235B-A22B-FP8,base_url=http://127.0.0.1:8000/v1/completions \
--batch_size 100 
```

## More to update

### Optimizations on the way
1. https://github.com/vllm-project/vllm/pull/28500 enables **q_norm + k_norm + rope fusion** on ROCm platforms, which was initially implemented for cuda in https://github.com/vllm-project/vllm/pull/27165.
2. https://github.com/vllm-project/vllm/pull/25693 added new fusion passes to enable **rms_norm + fp8_block_quant** and **silu + fp8_block_quant**, which depends on the triton fused kernel in https://github.com/ROCm/aiter/tree/dev/perf_fused_rms_fp8_group_quant. Need to check if this triton kernel merged into AITER main.
3. **Sequence parallel** code ready in https://github.com/ROCm/vllm/pull/790. But seems poor performance due to the pynccl comm op.
4. **All-reduce + rms_norm** fusion WIP in https://github.com/ROCm/vllm/pull/803.
5. FP8 block GEMM is not efficient enough, i.e., up to 1.2p ~ 1.4p flops even after tuning.

### Other parallelism
1. Try other parallel strategies for best performance across different scenarios. 



