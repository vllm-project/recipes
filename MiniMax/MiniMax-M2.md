# MiniMax-M2 Usage Guide

[MiniMax-M2](https://huggingface.co/MiniMaxAI/MiniMax-M2) is an advanced large language model created by [MiniMax](https://www.minimax.io/). It offers the following highlights:

* Superior Intelligence – Ranks #1 among open-source models globally across mathematics, science, coding, and tool use.
* Advanced Coding – Excels at multi-file edits, coding-run-fix loops, and test-validated repairs. Strong performance on SWE-Bench and Terminal-Bench tasks.
* Agent Performance – Plans and executes complex toolchains across shell, browser, and code environments. Maintains traceable evidence and recovers gracefully from errors.
* Efficient Design – 10B activated parameters (230B total) enables lower latency, cost, and higher throughput for interactive and batched workloads.

## Installing vLLM

```bash
uv venv
source .venv/bin/activate
uv pip install 'triton-kernels @ git+https://github.com/triton-lang/triton.git@v3.5.0#subdirectory=python/triton_kernels' \
   vllm --extra-index-url https://wheels.vllm.ai/nightly --prerelease=allow
```

## Launching MiniMax-M2 with vLLM

You can use 4x H200/H20 or 4x A100/A800 GPUs to launch this model.

run tensor-parallel like this:

```bash
vllm serve MiniMaxAI/MiniMax-M2 \
  --tensor-parallel-size 4 \
  --tool-call-parser minimax_m2 \
  --reasoning-parser minimax_m2_append_think  \
  --enable-auto-tool-choice
```

Note that TP8 is not supported. To run the model with >4 GPUs, please use DP+EP:

```bash
vllm serve MiniMaxAI/MiniMax-M2 \
  --data-parallel-size 8 \
  --enable-expert-parallel \
  --tool-call-parser minimax_m2 \
  --reasoning-parser minimax_m2_append_think  \
  --enable-auto-tool-choice
```

If you encounter `torch.AcceleratorError: CUDA error: an illegal memory access was encountered`, you can add `--compilation-config "{\"cudagraph_mode\": \"PIECEWISE\"}"` to the startup parameters to resolve this issue. 

```bash
vllm serve MiniMaxAI/MiniMax-M2 \
  --tensor-parallel-size 4 \
  --tool-call-parser minimax_m2 \
  --reasoning-parser minimax_m2_append_think  \
  --enable-auto-tool-choice \
  --compilation-config "{\"cudagraph_mode\": \"PIECEWISE\"}"
```

To run the model in responsesAPI that natively supports thinking, run it with the minimax_m2 reasoning parser:
```bash
vllm serve MiniMaxAI/MiniMax-M2 \
  --tensor-parallel-size 4 \
  --tool-call-parser minimax_m2 \
  --reasoning-parser minimax_m2  \
  --enable-auto-tool-choice
```

## Performance Metrics


### Benchmarking

We use the following script to demonstrate how to benchmark MiniMaxAI/MiniMax-M2`.

```bash
vllm bench serve \
  --backend vllm \
  --model MiniMaxAI/MiniMax-M2 \
  --endpoint /v1/completions \
  --dataset-name random \
  --random-input 2048 \
  --random-output 1024 \
  --max-concurrency 10 \
  --num-prompt 100 
```


If successful, you should see output similar to the following (TP 4 on NVIDIA_H20-3e *4) :

```
============ Serving Benchmark Result ============
Successful requests:                     100       
Failed requests:                         0         
Maximum request concurrency:             10        
Benchmark duration (s):                  851.51    
Total input tokens:                      204800    
Total generated tokens:                  98734     
Request throughput (req/s):              0.12      
Output token throughput (tok/s):         115.95    
Peak output token throughput (tok/s):    130.00    
Peak concurrent requests:                20.00     
Total Token throughput (tok/s):          356.46    
---------------Time to First Token----------------
Mean TTFT (ms):                          520.98    
Median TTFT (ms):                        523.86    
P99 TTFT (ms):                           1086.48   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          82.82     
Median TPOT (ms):                        82.90     
P99 TPOT (ms):                           84.28     
---------------Inter-token Latency----------------
Mean ITL (ms):                           82.78     
Median ITL (ms):                         82.18     
P99 ITL (ms):                            83.48 
```

## Using Tips

### DeepGEMM Usage

vLLM has DeepGEMM enabled by default, follow the [setup instructions](https://github.com/vllm-project/vllm/blob/v0.11.0/benchmarks/kernels/deepgemm/README.md#setup) to install it. However, on H20 GPUs, we've found that disabling DeepGEMM yields better performance for this model. To disable `DeepGEMM`, set the environment variable `VLLM_USE_DEEP_GEMM=0`.
