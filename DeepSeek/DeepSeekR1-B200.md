# DSR1 Status with vLLM: Aggregated Serving on B200

**Overall Health**: Most paths work. DP Attention is failing in combination with Flashinfer MOE Kernels

---

## FP8 DeepSeek Models

### FP8 Blockscale Gemm Blocksize 128

**Available Backends:**
- FlashInfer (CUTLASS, TRTLLM-Gen)
- vLLM CUTLASS BlockScale (SM90/SM100)
- DeepGemm

**Most Performant:** DeepGemm (High Throughput)

**How to Invoke:**

FlashInfer:
- Automatic on SM100 (requires flashinfer installed)
- Selection: `vllm/model_executor/layers/quantization/utils/w8a8_utils.py:407-408`
- Implementation: `vllm/utils/flashinfer.py:405-432` (calls `bmm_fp8`)

DeepGemm:
- `VLLM_USE_DEEP_GEMM=1` (default)
- `VLLM_USE_DEEP_GEMM_E8M0=1` (default)

CUTLASS BlockScale:
- Automatic fallback (requires CUDA 12.8+ for SM100)
- SM100: `csrc/quantization/cutlass_w8a8/scaled_mm_c3x_sm100.cu:19`
- Kernel: `csrc/quantization/cutlass_w8a8/c3x/scaled_mm_blockwise_sm100_fp8.cu`

**Notes:**
- DeepGemm requires shapes divisible by 128x128
- DeepGemm uses E8M0 scale on B200, requires requantization
- CUTLASS BlockScale used when DeepGemm requirements not met

**Related PRs:**
- [#25696](https://github.com/vllm-project/vllm/pull/25696) Fix w8a8 block fp8 linear
- [#25871](https://github.com/vllm-project/vllm/pull/25871) Update deepgemm for dsv3.2
- [#25517](https://github.com/vllm-project/vllm/pull/25517) DeepGEMM Col Major TMA

---

### FP8 Blockscale MOE with Group Size 128

**Available Backends:**

TP/EP:
- Flashinfer TRTLLM-Gen
- DeepGemm

TP-only:
- CUTLASS BlockScale MOE

**Most Performant:** Flashinfer TRTLLM-Gen or DeepGemm (Needs to be Evaluated)

**How to Invoke:**

DeepGemm:
- `VLLM_USE_DEEP_GEMM=1`

FlashInfer TRTLLM:
- `VLLM_USE_FLASHINFER_MOE_FP8=1 VLLM_FLASHINFER_MOE_BACKEND=latency`

FlashInfer CUTLASS:
- `VLLM_USE_FLASHINFER_MOE_FP8=1 VLLM_FLASHINFER_MOE_BACKEND=throughput`

CUTLASS BlockScale:
- Default on SM100 (auto-selected with block quant)

**Notes:**
- FlashInfer CUTLASS: Lacks block quant support for DeepSeek - fails with `assert self.block_quant is None`
- DeepGemm uses E8M0 scale on B200

**Related PRs:**
- [#26044](https://github.com/vllm-project/vllm/pull/26044) Optimize FP8 MOE Backend
- [#25968](https://github.com/vllm-project/vllm/pull/25968) TRTLLM NVFP4 MOE
- [#25895](https://github.com/vllm-project/vllm/pull/25895) Fix TRTLLM FP8 MOE accuracy
- [#25871](https://github.com/vllm-project/vllm/pull/25871) Update to latest deepgemm

---

### FMHA Prefill

**Available Backends:**
- FlashInfer CUTLASS
- TRT-LLM Ragged Attention (WIP)
- CUDNN
- FlashAttention-2

**Most Performant:** FlashInfer CUTLASS (default on SM100)

**How to Invoke:**

FlashInfer:
- Default on SM100 (automatic)
- Selection: `vllm/v1/attention/backends/mla/common.py:425-433`
- Disable: `VLLM_DISABLE_FLASHINFER_PREFILL=1`

TRT-LLM Ragged (WIP):
- Enable with `VLLM_USE_TRTLLM_RAGGED_DEEPSEEK_PREFILL=1`
- Uses `flashinfer.prefill.trtllm_ragged_attention_deepseek`
- Implementation: `vllm/v1/attention/backends/mla/common.py:_run_prefill_context_chunk_trtllm_ragged`
- For context chunks (non-causal)

CUDNN:
- Enable with `VLLM_USE_CUDNN_PREFILL=1`
- Requires SM100 + nvidia artifactory

FlashAttention-2:
- Automatic fallback on SM100

**Notes:**
- FlashInfer backend uses CUTLASS on SM100

**Related PRs:**
- [#26063](https://github.com/vllm-project/vllm/pull/26063) Fix FI accuracy issue for MLA prefill
- [#26397](https://github.com/vllm-project/vllm/pull/26397) WIP: TRT-LLM Ragged Attention

---

### MLA Decode

**Available Backends:**

DP/TP Attention:
- FlashInfer MLA (TRTLLM)
- CUTLASS MLA (default)

**Most Performant:** FLASHINFER MLA

**How to Invoke:**

CUTLASS MLA:
- Default on SM100 with block_size=128
- Selection: `vllm/platforms/cuda.py:269-291`
- Implementation: `vllm/v1/attention/backends/mla/cutlass_mla.py`
- Force: `VLLM_ATTENTION_BACKEND=CUTLASS_MLA`
- Debug hangs: `FORCE_NUM_KV_SPLITS=1`

FlashInfer MLA:
- Default on SM100 with block_size=32 or 64
- Force: `VLLM_ATTENTION_BACKEND=FLASHINFER_MLA`
- Implementation: `vllm/v1/attention/backends/mla/flashinfer_mla.py`

**Notes:**
- CUTLASS MLA: num_kv_splits limited to avoid hangs
- FlashInfer MLA: LSE not yet returned (pending flashinfer#1566)
- Both support FP8 compute with dynamic scales

**Related PRs:**
- [#26026](https://github.com/vllm-project/vllm/pull/26026) Fix CUTLASS MLA hang under load
- [#26132](https://github.com/vllm-project/vllm/pull/26132) Improve DS MLA cache kernel
- [#24705](https://github.com/vllm-project/vllm/pull/24705) Enable FP8 FlashInfer MLA decode
- [#24385](https://github.com/vllm-project/vllm/pull/24385) Decode context parallelism with CUTLASS MLA

---

### MTP Speculative Decode

**Available Backends:**
- FlashInfer MLA (TRTLLM)

**Most Performant:** FlashInfer MLA (TRTLLM)

**How to Invoke:**

FlashInfer MLA:
- Uses `trtllm_batch_decode_with_kv_cache_mla`
- Implementation: `vllm/v1/attention/backends/mla/flashinfer_mla.py`
- Enables `supports_uniform_spec_as_decode=True`
- Supports multi-token prediction via q_len_per_request dimension

Example:
```bash
VLLM_ATTENTION_BACKEND=FLASHINFER_MLA VLLM_USE_FLASHINFER_MOE_FP4=1 vllm serve nvidia/DeepSeek-R1-FP4 -tp 4 --max-model-len 8192 --no-enable-prefix-caching --port 8049 --speculative-config '{"method": "mtp", "num_speculative_tokens": 3}'
```

**Notes:**
- Auto-reshapes query for spec decode
- Handles uniform batches efficiently
- Fallback to prefill for non-uniform requests

**Related PRs:**
- [#25984](https://github.com/vllm-project/vllm/pull/25984) Enable efficient spec decode with FlashInfer-MLA

---

### Communications

**Available Backends:**

TP:
- AllReduce

EP:
- DeepEP High-Throughput
- DeepEP Low-Latency
- PPLX
- Naive All2All

**Most Performant:** DeepEP HT (needs evaluation)

**How to Invoke:**

TP/EP All-Reduce:
- After MOE combine when TP>1 or EP>1
- Called via `maybe_all_reduce_tensor_model_parallel` (`vllm/model_executor/layers/fused_moe/layer.py:2284`)
- Skipped if EP combine kernel already reduced
- Tested with FP8 DeepSeek: `tests/quantization/test_blackwell_moe.py`

EP Dispatch/Combine:
- `VLLM_ALL2ALL_BACKEND` env var

DeepEP HT:
- `deepep_high_throughput`
- Buffer: `VLLM_DEEPEP_BUFFER_SIZE_MB=1024`
- SMs: `VLLM_DBO_COMM_SMS=20`

DeepEP LL:
- `deepep_low_latency`
- RDMA, 0 SMs

Others:
- `pplx`
- `naive`

Implementation: `vllm/distributed/device_communicators/all2all.py`

**Notes:**
- AG-RS backend is FP4-only (see FP4 section)

**Related PRs:**
- [#21837](https://github.com/vllm-project/vllm/pull/21837) DeepEPHT Quant before Dispatch
- [#24845](https://github.com/vllm-project/vllm/pull/24845) DBO DeepEP HT

---

## FP4 Deepseek Models

**Available Models:**
- `nvidia/DeepSeek-R1-FP4`
- `nvidia/DeepSeek-R1-FP4-v2`
- `nvidia/DeepSeek-R1-0528-FP4`
- `nvidia/DeepSeek-R1-0528-FP4-v2`

---

### FP4 Gemms with Group Size 16

**Available Backends:**
- FlashInfer CUDNN (in review)
- FlashInfer TRTLLM
- FlashInfer CUTLASS
- vLLM CUTLASS (scaled_mm_fp4)
- Marlin GEMM (fallback)

**Most Performant:** FlashInfer CUDNN (~1% improvement)

**How to Invoke:**

Backend Selection:
- `VLLM_NVFP4_GEMM_BACKEND` env var (introduced in [#26107](https://github.com/vllm-project/vllm/pull/26107))
- Options: `flashinfer-cudnn`, `flashinfer-trtllm`, `flashinfer-cutlass`, `cutlass`
- Default: Uses first available (FlashInfer CUTLASS → FlashInfer TRTLLM → vLLM CUTLASS → Marlin)

FlashInfer CUDNN:
- `VLLM_NVFP4_GEMM_BACKEND=flashinfer-cudnn`
- Requires: `pip install nvidia-cudnn-cu12 nvidia-cudnn-frontend`
- Used in shared expert and output projection layers
- Implementation: `vllm/model_executor/layers/quantization/modelopt.py`

FlashInfer TRTLLM:
- `VLLM_NVFP4_GEMM_BACKEND=flashinfer-trtllm`

FlashInfer CUTLASS:
- `VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass`

vLLM CUTLASS:
- `VLLM_NVFP4_GEMM_BACKEND=cutlass`
- Uses `cutlass_scaled_fp4_mm`
- Fallback when FlashInfer not available

Marlin:
- Automatic fallback for non-SM100 GPUs

**Related PRs:**
- [#26107](https://github.com/vllm-project/vllm/pull/26107) In Review: Add CUDNN FP4 GEMM

---

### FP4 Blockscale MOE with Group Size 16

**Available Backends:**

TP/EP:
- FlashInfer CUTLASS
- FlashInfer TRTLLM

TP-only:
- vLLM CUTLASS FP4

Non-SM100:
- Marlin FP4

**Most Performant:** FlashInfer TRTLLM (latency) or CUTLASS (throughput)

**How to Invoke:**

Backend Selection:
- Two env vars control FlashInfer:
  - `VLLM_USE_FLASHINFER_MOE_FP4=1` to enable FlashInfer
  - `VLLM_FLASHINFER_MOE_BACKEND` to select variant

FlashInfer CUTLASS:
- `VLLM_USE_FLASHINFER_MOE_FP4=1 VLLM_FLASHINFER_MOE_BACKEND=throughput` (default)
- High-throughput batch inference
- Supports TP/EP
- Uses `grouped_gemm_nt_masked` from flashinfer

FlashInfer TRTLLM:
- `VLLM_USE_FLASHINFER_MOE_FP4=1 VLLM_FLASHINFER_MOE_BACKEND=latency`
- Low-latency inference
- Supports TP/EP

vLLM CUTLASS FP4:
- Automatic fallback when FlashInfer unavailable
- TP-only (no EP support)

Marlin FP4:
- Automatic fallback for non-SM100 GPUs

**Notes:**
- FlashInfer requires SM100 (Blackwell)
- vLLM CUTLASS FP4 doesn't support Expert Parallelism

**Related PRs:**
- [#25968](https://github.com/vllm-project/vllm/pull/25968) TRTLLM NVFP4 MOE
- [#25990](https://github.com/vllm-project/vllm/pull/25990) WIP: Masked Gemm for NVFP4

---

### FP4 Communications

**Available Backends:**

TP:
- AllReduce

EP (FP4-only):
- AllGather-ReduceScatter (AG-RS)
- FlashInfer All2Allv (MNNVL)
- Naive All2All

**Most Performant:** AllGather-ReduceScatter (default for EP)

**How to Invoke:**

TP/EP All-Reduce:
- After MOE combine when TP>1 or EP>1

EP Dispatch/Combine:
- `VLLM_ALL2ALL_BACKEND` env var

AG-RS (default):
- `VLLM_ALL2ALL_BACKEND=allgather_reducescatter` ([#23964](https://github.com/vllm-project/vllm/pull/23964))
- Dispatch: `all_gatherv`, Combine: `reduce_scatterv`

FlashInfer All2Allv:
- `VLLM_ALL2ALL_BACKEND=flashinfer_all2allv` (requires MNNVL hardware)

Naive:
- `VLLM_ALL2ALL_BACKEND=naive`

**Notes:**

Backend Compatibility:
- With FlashInfer CUTLASS MOE: Can use All2Allv or AG-RS
- With FlashInfer TRTLLM MOE: Can only use AG-RS or Naive

Recommendations:
- All2Allv is for multinode setups
- AG-RS is generally better when Data Parallelism is enabled

AG-RS only works with FP4 quantization (not compatible with FP8)

**Related PRs:**
- [#23964](https://github.com/vllm-project/vllm/pull/23964) Enable AG/RS backend
- [#21003](https://github.com/vllm-project/vllm/pull/21003) MNNVL all2allv

---

### FP4 Attention (FMHA Prefill, MLA Decode, MTP Speculative Decode)

**Note:** FP4 models use the same attention and MTP implementations as FP8 models. Only Gemms and MOE components are FP4-quantized.

For details, see:
- **FMHA Prefill:** [FP8 FMHA Prefill](#fmha-prefill)
- **MLA Decode:** [FP8 MLA Decode](#mla-decode)
- **MTP Speculative Decode:** [FP8 MTP Speculative Decode](#mtp-speculative-decode)

---

## Example Usage: Expert Parallel Communication Backends (FP4)

This example demonstrates benchmarking different EP communication backends with FP4 DeepSeek-R1 on 4 GPUs using Data Parallelism (DP=4) and Expert Parallelism.

**Note**: AllGather-ReduceScatter (AG-RS) is FP4-specific and not compatible with FP8 quantization yet.

### Serve with AllGather-ReduceScatter (default, recommended):

```bash
VLLM_ALL2ALL_BACKEND="allgather_reducescatter" \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
VLLM_USE_STANDALONE_COMPILE=0 \
VLLM_USE_FLASHINFER_MOE_FP4=1 \
VLLM_FLASHINFER_MOE_BACKEND="latency" \
  vllm serve nvidia/DeepSeek-R1-FP4 
    --quantization="modelopt_fp4" \
    --trust-remote-code \
    --max-model-len=2048 \
    --block-size=128 \
    --max-num-seqs=256 \
    --enable-expert-parallel \
    --gpu-memory-utilization=0.8 \
    --tensor-parallel-size=1 \
    --data-parallel-size=4
```

### Run benchmark:

```bash
python benchmarks/benchmark_serving.py \
  --model nvidia/DeepSeek-R1-FP4 \
  --dataset-name random \
  --ignore-eos \
  --num-prompts 256 \
  --max-concurrency 256 \
  --random-input-len 128 \
  --random-output-len 1024
```

### Compare with Naive backend:

Change the backend to naive (less efficient, for comparison):
```bash
VLLM_ALL2ALL_BACKEND="naive" \
# ... use same serve command and benchmark above
```

**Other backend options:**
- `deepep_high_throughput`: High-throughput with NVLink buffers ([#24845](https://github.com/vllm-project/vllm/pull/24845), [#21837](https://github.com/vllm-project/vllm/pull/21837))
  - Adjust: `VLLM_DEEPEP_BUFFER_SIZE_MB=1024`, `VLLM_DBO_COMM_SMS=20`
- `deepep_low_latency`: Low-latency with RDMA ([#25904](https://github.com/vllm-project/vllm/pull/25904))
- `pplx`: PPLX kernels (intranode P2P, internode NVSHMEM) ([#20825](https://github.com/vllm-project/vllm/pull/20825))
- `flashinfer_all2allv`: FlashInfer MNNVL (requires MNNVL hardware) ([#21003](https://github.com/vllm-project/vllm/pull/21003))

