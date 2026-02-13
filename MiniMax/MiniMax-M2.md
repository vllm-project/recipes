# MiniMax-M2.1/M2 Usage Guide

[MiniMax-M2.5](https://huggingface.co/MiniMaxAI/MiniMax-M2.5), [MiniMax-M2.1](https://huggingface.co/MiniMaxAI/MiniMax-M2.1) and [MiniMax-M2](https://huggingface.co/MiniMaxAI/MiniMax-M2) are advanced large language models created by [MiniMax](https://www.minimax.io/). They offer the following highlights:

* Superior Intelligence – Ranks #1 among open-source models globally across mathematics, science, coding, and tool use.
* Advanced Coding – Excels at multi-file edits, coding-run-fix loops, and test-validated repairs. Strong performance on SWE-Bench and Terminal-Bench tasks.
* Agent Performance – Plans and executes complex toolchains across shell, browser, and code environments. Maintains traceable evidence and recovers gracefully from errors.
* Efficient Design – 10B activated parameters (230B total) enables lower latency, cost, and higher throughput for interactive and batched workloads.

## Supported Models

This guide applies to the following models. You only need to update the model name during deployment. The following examples use **MiniMax-M2.5**:

- [MiniMaxAI/MiniMax-M2.5](https://huggingface.co/MiniMaxAI/MiniMax-M2.5)
- [MiniMaxAI/MiniMax-M2.1](https://huggingface.co/MiniMaxAI/MiniMax-M2.1)
- [MiniMaxAI/MiniMax-M2](https://huggingface.co/MiniMaxAI/MiniMax-M2)

## Usage

## System Requirements

- OS: Linux

- Python: 3.10 - 3.13

- GPU:

  - compute capability 7.0 or higher

  - Memory requirements: 220 GB for weights, 240 GB per 1M context tokens

The following are recommended configurations; actual requirements should be adjusted based on your use case:

- **96G x4** GPU: Supports a total KV Cache capacity of 400K tokens.

- **144G x8** GPU: Supports a total KV Cache capacity of up to 3M tokens.

> **Note**: The values above represent the total aggregate hardware KV Cache capacity. The maximum context length per individual sequence remains **196K** tokens.


### Using Docker

```bash
docker run --gpus all \
  -p 8000:8000 \
  --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:nightly MiniMaxAI/MiniMax-M2.5 \
      --tensor-parallel-size 4 \
      --tool-call-parser minimax_m2 \
      --reasoning-parser minimax_m2_append_think \
      --enable-auto-tool-choice \
      --trust-remote-code
```

### Installing vLLM from source

#### Install nightly version

- If you encounter corrupted output when using vLLM to serve these models, you can upgrade to the nightly version (ensure it is a version after commit [cf3eacfe58fa9e745c2854782ada884a9f992cf7](https://github.com/vllm-project/vllm/commit/cf3eacfe58fa9e745c2854782ada884a9f992cf7))

```bash
uv venv
source .venv/bin/activate
uv pip install -U vllm --extra-index-url https://wheels.vllm.ai/nightly
```

#### Install verified version

- We also verified the accuracy of `MiniMax-M2.5` in commit [dea63512bb9bdf7521d591546c52138d9d79e8ce](https://github.com/vllm-project/vllm/commit/dea63512bb9bdf7521d591546c52138d9d79e8ce)
  on AIME25, GPQA-D, and AA-LCR. The accuracy aligns with the official accuracy. You can use the following method to install this version.

```bash
uv venv
source .venv/bin/activate
export VLLM_COMMIT=dea63512bb9bdf7521d591546c52138d9d79e8ce # use full commit hash from the main branch
uv pip install vllm \
    --torch-backend=auto \
    --extra-index-url https://wheels.vllm.ai/${VLLM_COMMIT} # add variant subdirectory here if needed
```



## Launching  M2.5/M2.1/M2 with vLLM

You can use 4x H200/H20 or 4x A100/A800 GPUs to launch this model.

run tensor-parallel like this:

```bash
vllm serve MiniMaxAI/MiniMax-M2.5 \
  --tensor-parallel-size 4 \
  --tool-call-parser minimax_m2 \
  --reasoning-parser minimax_m2_append_think  \
  --enable-auto-tool-choice
```

Note that pure TP8 is not supported. To run the model with >4 GPUs, please use DP+EP or TP+EP:

- DP8+EP
```bash
vllm serve MiniMaxAI/MiniMax-M2.5 \
  --data-parallel-size 8 \
  --enable-expert-parallel \
  --tool-call-parser minimax_m2 \
  --reasoning-parser minimax_m2_append_think  \
  --enable-auto-tool-choice
```

- TP8+EP

```bash
vllm serve MiniMaxAI/MiniMax-M2.5 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --tool-call-parser minimax_m2 \
  --reasoning-parser minimax_m2_append_think  \
  --enable-auto-tool-choice
```



If you encounter `torch.AcceleratorError: CUDA error: an illegal memory access was encountered`, you can add `--compilation-config "{\"cudagraph_mode\": \"PIECEWISE\"}"` to the startup parameters to resolve this issue. 

```bash
vllm serve MiniMaxAI/MiniMax-M2.5 \
  --tensor-parallel-size 4 \
  --tool-call-parser minimax_m2 \
  --reasoning-parser minimax_m2_append_think  \
  --enable-auto-tool-choice \
  --compilation-config "{\"cudagraph_mode\": \"PIECEWISE\"}"
```

To run the model in responsesAPI that natively supports thinking, run it with the minimax_m2 reasoning parser:
```bash
vllm serve MiniMaxAI/MiniMax-M2.5 \
  --tensor-parallel-size 4 \
  --tool-call-parser minimax_m2 \
  --reasoning-parser minimax_m2 \
  --enable-auto-tool-choice
```

## Performance Metrics


### Benchmarking

We use the following script to demonstrate how to benchmark MiniMax-M2 models.

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

vLLM has DeepGEMM enabled by default, you can install DeepGEMM using [install_deepgemm.sh](https://github.com/vllm-project/vllm/blob/main/tools/install_deepgemm.sh).

### GB200 Usage

- On B200 GPUs, you may encounter the following error when serving this model:

  <details>
  <summary>FlashInfer FP8 MoE Error</summary>

    ```bash

    (Worker_TP2 pid=479523) ERROR 02-13 00:28:06 [multiproc_executor.py:863]   File "/mnt/data/vllm/vllm/model_executor/layers/fused_moe/flashinfer_trtllm_moe.py", line 222, in flashinfer_fused_moe_blockscale_fp8
    (Worker_TP2 pid=479523) ERROR 02-13 00:28:06 [multiproc_executor.py:863]     return flashinfer_trtllm_fp8_block_scale_moe(
    (Worker_TP2 pid=479523) ERROR 02-13 00:28:06 [multiproc_executor.py:863]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    (Worker_TP2 pid=479523) ERROR 02-13 00:28:06 [multiproc_executor.py:863]   File "/mnt/data/vllm/vllm/utils/flashinfer.py", line 102, in wrapper
    (Worker_TP2 pid=479523) ERROR 02-13 00:28:06 [multiproc_executor.py:863]     return impl(*args, **kwargs)
    (Worker_TP2 pid=479523) ERROR 02-13 00:28:06 [multiproc_executor.py:863]            ^^^^^^^^^^^^^^^^^^^^^
    (Worker_TP2 pid=479523) ERROR 02-13 00:28:06 [multiproc_executor.py:863]   File "/mnt/data/vllm/.venv/lib/python3.12/site-packages/flashinfer/fused_moe/core.py", line 2335, in trtllm_fp8_block_scale_moe
    (Worker_TP2 pid=479523) ERROR 02-13 00:28:06 [multiproc_executor.py:863]     return get_trtllm_moe_sm100_module().trtllm_fp8_block_scale_moe(
    (Worker_TP2 pid=479523) ERROR 02-13 00:28:06 [multiproc_executor.py:863]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    (Worker_TP2 pid=479523) ERROR 02-13 00:28:06 [multiproc_executor.py:863]   File "/mnt/data/vllm/.venv/lib/python3.12/site-packages/flashinfer/fused_moe/core.py", line 1660, in trtllm_fp8_block_scale_moe_op
    (Worker_TP2 pid=479523) ERROR 02-13 00:28:06 [multiproc_executor.py:863]     result = moe_op.trtllm_fp8_block_scale_moe(
    (Worker_TP2 pid=479523) ERROR 02-13 00:28:06 [multiproc_executor.py:863]              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    (Worker_TP2 pid=479523) ERROR 02-13 00:28:06 [multiproc_executor.py:863]   File "python/tvm_ffi/cython/function.pxi", line 923, in tvm_ffi.core.Function.__call__
    (Worker_TP2 pid=479523) ERROR 02-13 00:28:06 [multiproc_executor.py:863]   File "<unknown>", line 0, in __tvm_ffi_trtllm_fp8_block_scale_moe
    (Worker_TP2 pid=479523) ERROR 02-13 00:28:06 [multiproc_executor.py:863]   File "<unknown>", line 0, in flashinfer::trtllm_fp8_block_scale_moe(tvm::ffi::Optional<tvm::ffi::TensorView, void>, tvm::ffi::TensorView, tvm::ffi::TensorView, tvm::ffi::Optional<tvm::ffi::TensorView, void>, tvm::ffi::TensorView, tvm::ffi::TensorView, tvm::ffi::TensorView, tvm::ffi::TensorView, tvm::ffi::TensorView, tvm::ffi::TensorView, tvm::ffi::TensorView, long, long, tvm::ffi::Optional<long, void>, tvm::ffi::Optional<long, void>, long, long, long, tvm::ffi::Optional<double, void>, long, bool, long, bool, tvm::ffi::Array<long, void>)
    (EngineCore_DP0 pid=479301) ERROR 02-13 00:28:06 [core.py:1006] EngineCore failed to start.
    (Worker_TP2 pid=479523) ERROR 02-13 00:28:06 [multiproc_executor.py:863]   File "<unknown>", line 0, in flashinfer::Fp8BlockScaleLauncher::run(long, bool, bool, bool)
    (Worker_TP2 pid=479523) ERROR 02-13 00:28:06 [multiproc_executor.py:863]   File "/mnt/data/vllm/.venv/lib/python3.12/site-packages/flashinfer/data/csrc/trtllm_fused_moe_kernel_launcher.cu", line 776, in virtual void flashinfer::Fp8BlockScaleLauncher::check_routing() const
    (EngineCore_DP0 pid=479301) ERROR 02-13 00:28:06 [core.py:1006] Traceback (most recent call last):
    (Worker_TP2 pid=479523) ERROR 02-13 00:28:06 [multiproc_executor.py:863]     TVM_FFI_ICHECK(args->n_group != 0) << "n_group should not be zero for DeepSeekV3 routing";
    (EngineCore_DP0 pid=479301) ERROR 02-13 00:28:06 [core.py:1006]   File "/mnt/data/vllm/vllm/v1/engine/core.py", line 996, in run_engine_core
    (Worker_TP2 pid=479523) ERROR 02-13 00:28:06 [multiproc_executor.py:863]

    ```

  </details>


  As a workaround, you can set `export VLLM_USE_FLASHINFER_MOE_FP8=0`.
