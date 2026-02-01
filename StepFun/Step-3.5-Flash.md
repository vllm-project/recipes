# Step-3.5-Flash Guide

[Step-3.5-Flash](https://huggingface.co/stepfun-ai/Step-3.5-Flash) is an advanced large language model developed by [StepFun](https://www.stepfun.com/company). It is a production-grade reasoning engine built to decouple elite intelligence from heavy compute, and cuts attention cost for low-latency, cost-effective long-context inference—purpose-built for autonomous agents in real-world workflows. It offers the following highlights:

* Hybrid Attention Schedules and Compensation for SWA
* Sparse Mixture-of-Experts (MoE) structure (only 11B active parameters out of 196B parameters)
* Multi-token prediction mechanism for faster inference


## Installing vLLM

Install vLLM nightly wheel until next vllm version is released
```bash
uv venv
source .venv/bin/activate
uv pip install -U vllm --pre \
    --extra-index-url https://wheels.vllm.ai/nightly/cu129 \
    --extra-index-url https://download.pytorch.org/whl/cu129 \
    --index-strategy unsafe-best-match
```

## Serving with vLLM


### Official Provided Formats

`Step-3.5-Flash` provides three precision options, You can choose the appropriate model based on your needs.
- [stepfun-ai/Step-3.5-Flash](https://huggingface.co/stepfun-ai/Step-3.5-Flash)
- [stepfun-ai/Step-3.5-Flash-FP8](https://huggingface.co/stepfun-ai/Step-3.5-Flash-FP8)
- [stepfun-ai/Step-3.5-Flash-Int4](https://huggingface.co/stepfun-ai/Step-3.5-Flash-Int4)

**NOTE**: Currently vLLM doesn't support the Int4 format of this model.


### Running Step-3.5-Flash on 4xH200/H20/B200


There are two ways to parallelize the model over multiple GPUs: (1) Tensor-parallel or (2) Data-parallel. Each one has its own advantages, where tensor-parallel is usually more beneficial for low-latency / low-load scenarios and data-parallel works better for cases where there is a lot of data with heavy-loads.


<details>
<summary>Data Parallel Script</summary>

```bash
vllm serve stepfun-ai/Step-3.5-Flash \
    --data-parallel-size 4 \
    --enable-expert-parallel \
    --reasoning-parser step3p5 \
    --tool-call-parser step3p5 \
    --enable-auto-tool-choice \
    --trust-remote-code
```

</details>


<details>
<summary>Tensor Parallel Script</summary>

``` bash
vllm serve stepfun-ai/Step-3.5-Flash \
    --tensor-parallel-size 4 \
    --reasoning-parser step3p5 \
    --tool-call-parser step3p5 \
    --enable-auto-tool-choice \
    --trust-remote-code
```

**Note:** The FP8 version of `Step-3.5-Flash` cannot use TP4. You can try DP4 instead.


</details>


<details>
<summary>Enable MTP Script</summary>


- To enhance MTP usage, append the following speculative configuration:

```bash
--hf-overrides '{"num_nextn_predict_layers": 1}' \
--speculative-config '{"method": "step3p5_mtp", "num_speculative_tokens": 1}' 
```

The server startup script is shown below:

``` bash
vllm serve stepfun-ai/Step-3.5-Flash \
    --tensor-parallel-size 4 \
    --reasoning-parser step3p5 \
    --tool-call-parser step3p5 \
    --enable-auto-tool-choice  \
    --trust-remote-code \
    --hf-overrides '{"num_nextn_predict_layers": 1}' \
    --speculative-config '{"method": "step3p5_mtp", "num_speculative_tokens": 1}' 
```
</details>


## Benchmark 

We use the following script to demonstrate how to benchmark stepfun-ai/Step-3.5-Flash`.

```bash
vllm bench serve \
  --backend vllm \
  --model stepfun-ai/Step-3.5-Flash \
  --endpoint /v1/completions \
  --dataset-name random \
  --random-input 2048 \
  --random-output 1024 \
  --max-concurrency 10 \
  --num-prompt 100 
```

If successful, you should see output similar to the following (TP4+FP16 on 4*H200)

```bash
tip: install termplotlib and gnuplot to plot the metrics
============ Serving Benchmark Result ============
Successful requests:                     100       
Failed requests:                         0         
Maximum request concurrency:             10        
Benchmark duration (s):                  126.12    
Total input tokens:                      204700    
Total generated tokens:                  102400    
Request throughput (req/s):              0.79      
Output token throughput (tok/s):         811.94    
Peak output token throughput (tok/s):    940.00    
Peak concurrent requests:                20.00     
Total token throughput (tok/s):          2435.02   
---------------Time to First Token----------------
Mean TTFT (ms):                          422.62    
Median TTFT (ms):                        482.04    
P99 TTFT (ms):                           1092.98   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          11.91     
Median TPOT (ms):                        11.87     
P99 TPOT (ms):                           12.61     
---------------Inter-token Latency----------------
Mean ITL (ms):                           11.91     
Median ITL (ms):                         11.78     
P99 ITL (ms):                            13.45     
==================================================
```


## Usage Tips

- See [tune-moe-kernel](https://github.com/vllm-project/recipes/blob/main/Qwen/Qwen3-Next.md#tune-moe-kernel) to perform MoE Triton kernel tuning for your hardware.

- For FP8 model, you can install DeepGEMM using [install_deepgemm.sh](https://github.com/vllm-project/vllm/blob/v0.16.0rc0/tools/install_deepgemm.sh).

- On B200 GPUs, you may encounter the following error when you serving the FP8 model:

    <details>
    <summary>FlashInfer FP8 MoE Error</summary>

    ```bash
        fer/fused_moe/core.py", line 1635, in trtllm_fp8_block_scale_moe_op
        (EngineCore_DP0 pid=1026745) (Worker_TP0 pid=1026751) ERROR 02-01 09:32:13 [multiproc_executor.py:852]     result = moe_op.trtllm_fp8_block_scale_moe(
        (EngineCore_DP0 pid=1026745) (Worker_TP0 pid=1026751) ERROR 02-01 09:32:13 [multiproc_executor.py:852]              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        (EngineCore_DP0 pid=1026745) (Worker_TP0 pid=1026751) ERROR 02-01 09:32:13 [multiproc_executor.py:852]   File "python/tvm_ffi/cython/function.pxi", line 923, in tvm_ffi.core.Function.__call__
        (EngineCore_DP0 pid=1026745) (Worker_TP0 pid=1026751) ERROR 02-01 09:32:13 [multiproc_executor.py:852]   File "<unknown>", line 0, in __tvm_ffi_trtllm_fp8_block_scale_moe
        (EngineCore_DP0 pid=1026745) (Worker_TP0 pid=1026751) ERROR 02-01 09:32:13 [multiproc_executor.py:852]   File "/vllm/.venv/lib/python3.12/site-packages/flashinfer/data/csrc/trtllm_fused_moe_kernel_launcher.cu", line 1580, in tvm::ffi::Tensor flashinfer::trtllm_fp8_block_scale_moe(tvm::ffi::TensorView, tvm::ffi::Optional<tvm::ffi::TensorView>, tvm::ffi::TensorView, tvm::ffi::TensorView, tvm::ffi::TensorView, tvm::ffi::TensorView, tvm::ffi::TensorView, tvm::ffi::TensorView, tvm::ffi::TensorView, int64_t, int64_t, tvm::ffi::Optional<long int>, tvm::ffi::Optional<long int>, int64_t, int64_t, int64_t, tvm::ffi::Optional<double>, int64_t, bool, int64_t, bool, tvm::ffi::Array<long int>)
        (EngineCore_DP0 pid=1026745) (Worker_TP0 pid=1026751) ERROR 02-01 09:32:13 [multiproc_executor.py:852]     TVM_FFI_ICHECK_EQ(routing_logits.dtype(), dl_bfloat16) << "routing_logits must be bfloat16.";
        (EngineCore_DP0 pid=1026745) (Worker_TP0 pid=1026751) ERROR 02-01 09:32:13 [multiproc_executor.py:852]     
        (EngineCore_DP0 pid=1026745) (Worker_TP0 pid=1026751) ERROR 02-01 09:32:13 [multiproc_executor.py:852] RuntimeError: Check failed: routing_logits.dtype() == dl_bfloat16 (float32 vs. bfloat16) : routing_logits must be bfloat16.
    ```

    </details>

    As a workaround, you can set `export VLLM_USE_FLASHINFER_MOE_FP8=0`.


