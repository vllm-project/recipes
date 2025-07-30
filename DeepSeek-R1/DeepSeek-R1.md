# DeepSeek-R1 Usage Guide

This guide describes how to run DeepSeek-R1 FP8 and FP4 models.

## Installing vLLM

```bash
uv venv
source .venv/bin/activate
uv pip install -U vllm --torch-backend auto
```

## Running DeepSeek-R1 FP8/FP4 with vLLM

### Set Environment Config

Set server listening port (change if necessary)
```bash
export PORT=8888
```

Choose whether to use 8 or 4 GPUs:

For 8 GPUs:
```bash
export NUM_GPUS=8
export CUDA_GPUS=0,1,2,3,4,5,6,7
```
For 4 GPUs:
```bash
export NUM_GPUS=4
export CUDA_GPUS=0,1,2,3 # Change indices if needed

```
Choose whether to run FP8 or FP4 model:

For FP8 model:
```bash
# This is an FP8 DeepSeek-R1, size is about 700GB
# (will work on 4xB200 GPUs, but KVCache will be stressed)
export MODEL=deepseek-ai/DeepSeek-R1-0528
```
For FP4 model:

```bash
# This is an FP4 DeepSeek-R1, size is about 404GB
# (will work well on 4xB200 GPUs)
export MODEL=nvidia/DeepSeek-R1-FP4
```

Define max model length and max batched tokens (reduce if there is not enough memory):

```bash
# This defines maximum model length: 64K usually works well 
# for most scenarios (can be reduced to preserve memory).
export MAX_MODEL_LEN=65536

# This defines how many tokens we can batch in one scheduler 
# iteration. 32K is a bit large (and good for heavy-prompt workloads), 
# but it can be reduced to 16k and 8k if necessary to reduce memory usage.
export MAX_BATCHED_TOKENS=32768
```

Set GPU utilization to max possible (provides more KVCache, however, reduce to free memory):
```bash
# Maximize GPU memory utilization (standard is 0.9)
export GPU_UTIL=0.95
```

For blackwell, enable NVIDIA's CUTLASS MLA decode optimization:
```bash
export VLLM_ATTENTION_BACKEND=CUTLASS_MLA_VLLM_V1
```

For blackwell, enable NVIDIA's FP8/FP4 MOE optimizations:

For FP8:
```bash
export VLLM_USE_FLASHINFER_MOE_FP8=1
```
For FP4:
```bash
export VLLM_USE_FLASHINFER_MOE_FP4=1
```

If doing a benchmark run, then ensure prefix caching is disabled:

For standard run:
```bash
export NO_PREFIX_CACHE= 
```

For benchmark run:
```bash
export NO_PREFIX_CACHE=--no-enable-prefix-caching
```

### Start Server

```bash
CUDA_VISIBLE_DEVICES=$CUDA_GPUS python vllm/entrypoints/openai/api_server.py \
  --disable-log-requests \
  --trust-remote-code \
  --port $PORT \
  --model $MODEL \
  --tensor-parallel-size $NUM_GPUS \
  --gpu-memory-utilization $GPU_UTIL \
  --max-model-len=$MAX_MODEL_LEN \
  --max-num-batched-tokens=$MAX_BATCHED_TOKENS \
  --enable_expert_parallel \
  $NO_PREFIX_CACHE

```

### Run Benchmark Client
Here we show how to benchmark the server by running `benchmark_serving.py`. 

Server's port from above:
```bash
export PORT=8888
```

Server's model from above:

For FP8:
```bash
export MODEL=deepseek-ai/DeepSeek-R1-0528
```
For FP4:
```bash
export MODEL=nvidia/DeepSeek-R1-FP4
```

Set framework to vllm:
```bash
export FRAMEWORK=vllm
```

Set prompt/decode length. In this example, it is 8k/1k which is prompt-heavy. For decode-heavy use 1k/8k, for balanced use 1k/1k.
```bash
export INPUT_LEN=8000
export OUTPUT_LEN=1000
```

Set request rate to a large number (infinity) to simulate a full batch execution:
```bash
export REQUEST_RATE=10000
```

To test batch size 1, set:
```bash
export NUM_PROMPTS=1
```

As an example, to test batch size 16, set:

```bash
export NUM_PROMPTS=16
```

Usually, it makes sense to test batch sizes: 1,16,32,64,128,256,512

Run benchmark:

```bash
python3 ./benchmarks/benchmark_serving.py \
	--model $MODEL \
	--dataset-name random \
	--random-input-len $INPUT_LEN \
	--random-output-len $OUTPUT_LEN \
	--request-rate $REQUEST_RATE \
	--num-prompts $NUM_PROMPTS \
	--seed $REQUEST_RATE \
	--ignore-eos \
	--metadata "framework=$FRAMEWORK" \
	--port $PORT

```

If successful, you will see the following output that shows TTFT/TPOT. Ensure that `Successful requests` is the same as NUM_PROMPTS env var.

```shell
============ Serving Benchmark Result ============
Successful requests:                     1         
Benchmark duration (s):                  16.39     
Total input tokens:                      7902      
Total generated tokens:                  1000      
Request throughput (req/s):              0.06      
Output token throughput (tok/s):         61.00     
Total Token throughput (tok/s):          543.06    
---------------Time to First Token----------------
Mean TTFT (ms):                          560.00    
Median TTFT (ms):                        560.00    
P99 TTFT (ms):                           560.00    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          15.85     
Median TPOT (ms):                        15.85     
P99 TPOT (ms):                           15.85     
---------------Inter-token Latency----------------
Mean ITL (ms):                           15.85     
Median ITL (ms):                         15.85     
P99 ITL (ms):                            16.15     
==================================================
```

