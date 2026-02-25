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




