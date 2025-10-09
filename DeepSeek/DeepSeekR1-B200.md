DSR1 Status with vLLM: Aggregated Serving on B200

**Overall Health**: Most paths work. DP Attention is failing in combination with Flashinfer MOE Kernels

---

## FP8 - Compatible

| Component | Available Backends | Most Performant at Kernel Level | How to Invoke | Notes | Related Recent PRs |
|-----------|-------------------|----------------------------------|---------------|-------|--------------------|
| **Gemms** | - Flashinfer (CUTLASS, TRTLLM-Gen)<br>- vLLM CUTLASS BlockScale (SM90/SM100)<br>- DeepGemm | DeepGemm (High Throughput) | **FlashInfer**: Automatic on SM100 (requires flashinfer installed)<br>Selection: `vllm/model_executor/layers/quantization/utils/w8a8_utils.py:407-408`<br>Impl: `vllm/utils/flashinfer.py:405-432` (calls `bmm_fp8`)<br><br>**DeepGemm**: `VLLM_USE_DEEP_GEMM=1` (default), `VLLM_USE_DEEP_GEMM_E8M0=1` (default)<br><br>**CUTLASS BlockScale**: Automatic fallback (requires CUDA 12.8+ for SM100)<br>SM100: `csrc/quantization/cutlass_w8a8/scaled_mm_c3x_sm100.cu:19`<br>Kernel: `csrc/quantization/cutlass_w8a8/c3x/scaled_mm_blockwise_sm100_fp8.cu` | DeepGemm requires shapes divisible by 128x128<br>DeepGemm uses E8M0 scale on B200, requires requantization<br>CUTLASS BlockScale used when DeepGemm requirements not met | [#25696](https://github.com/vllm-project/vllm/pull/25696) Fix w8a8 block fp8 linear<br>[#25871](https://github.com/vllm-project/vllm/pull/25871) Update deepgemm for dsv3.2<br>[#25517](https://github.com/vllm-project/vllm/pull/25517) DeepGEMM Col Major TMA |
| **MOE** | **TP/EP:**<br>  - Flashinfer TRTLLM-Gen<br>  - DeepGemm<br><br>**TP-only:**<br>  - CUTLASS BlockScale MOE | Flashinfer TRTLLM-Gen or DeepGemm (Needs to be Evaluated) | DeepGemm: `VLLM_USE_DEEP_GEMM=1`<br>FlashInfer TRTLLM: `VLLM_USE_FLASHINFER_MOE_FP8=1 VLLM_FLASHINFER_MOE_BACKEND=latency`<br>FlashInfer CUTLASS: `VLLM_USE_FLASHINFER_MOE_FP8=1 VLLM_FLASHINFER_MOE_BACKEND=throughput`<br>CUTLASS BlockScale: Default on SM100 (auto-selected with block quant) | FlashInfer CUTLASS: Lacks block quant support for DeepSeek - fails with `assert self.block_quant is None`<br>DeepGemm uses E8M0 scale on B200 | [#26044](https://github.com/vllm-project/vllm/pull/26044) Optimize FP8 MOE Backend<br>[#25968](https://github.com/vllm-project/vllm/pull/25968) TRTLLM NVFP4 MOE<br>[#25895](https://github.com/vllm-project/vllm/pull/25895) Fix TRTLLM FP8 MOE accuracy<br>[#25871](https://github.com/vllm-project/vllm/pull/25871) Update to latest deepgemm |
| **FMHA (Prefill)** | - FlashInfer CUTLASS<br>- TRT-LLM Ragged Attention (WIP)<br>- CUDNN<br>- FlashAttention-2 | FlashInfer CUTLASS (default on SM100) | **FlashInfer**: Default on SM100 (automatic)<br>Selection: `vllm/v1/attention/backends/mla/common.py:425-433`<br>Disable: `VLLM_DISABLE_FLASHINFER_PREFILL=1`<br><br>**TRT-LLM Ragged (WIP)**: Enable with `VLLM_USE_TRTLLM_RAGGED_DEEPSEEK_PREFILL=1`<br>Uses `flashinfer.prefill.trtllm_ragged_attention_deepseek`<br>Impl: `vllm/v1/attention/backends/mla/common.py:_run_prefill_context_chunk_trtllm_ragged`<br>For context chunks (non-causal)<br><br>**CUDNN**: Enable with `VLLM_USE_CUDNN_PREFILL=1`<br>Requires SM100 + nvidia artifactory<br><br>**FlashAttention-2**: Automatic fallback on SM100 | FlashInfer backend uses CUTLASS on SM100<br> | [#26063](https://github.com/vllm-project/vllm/pull/26063) Fix FI accuracy issue for MLA prefill<br>[#26397](https://github.com/vllm-project/vllm/pull/26397) WIP: TRT-LLM Ragged Attention |
| **MLA (Decode)** | **DP/TP Attention:**<br>  - FlashInfer MLA (TRTLLM)<br>  - CUTLASS MLA(default) | FLASHINFER MLA | **CUTLASS MLA**: Default on SM100 with block_size=128<br>Selection: `vllm/platforms/cuda.py:269-291`<br>Impl: `vllm/v1/attention/backends/mla/cutlass_mla.py`<br>Force: `VLLM_ATTENTION_BACKEND=CUTLASS_MLA`<br>Debug hangs: `FORCE_NUM_KV_SPLITS=1`<br><br>**FlashInfer MLA**: Default on SM100 with block_size=32 or 64<br>Force: `VLLM_ATTENTION_BACKEND=FLASHINFER_MLA`<br>Impl: `vllm/v1/attention/backends/mla/flashinfer_mla.py` | CUTLASS MLA: num_kv_splits limited to avoid hangs<br>FlashInfer MLA: LSE not yet returned (pending flashinfer#1566)<br>Both support FP8 compute with dynamic scales | [#26026](https://github.com/vllm-project/vllm/pull/26026) Fix CUTLASS MLA hang under load<br>[#26132](https://github.com/vllm-project/vllm/pull/26132) Improve DS MLA cache kernel<br>[#24705](https://github.com/vllm-project/vllm/pull/24705) Enable FP8 FlashInfer MLA decode<br>[#24385](https://github.com/vllm-project/vllm/pull/24385) Decode context parallelism with CUTLASS MLA |
| **MTP (Speculative Decode)** | - FlashInfer MLA (TRTLLM) | FlashInfer MLA (TRTLLM) | **FlashInfer MLA**: Uses `trtllm_batch_decode_with_kv_cache_mla`<br>Impl: `vllm/v1/attention/backends/mla/flashinfer_mla.py`<br>Enables `supports_uniform_spec_as_decode=True`<br>Supports multi-token prediction via q_len_per_request dimension<br><br>**Example**:<br>`VLLM_ATTENTION_BACKEND=FLASHINFER_MLA VLLM_USE_FLASHINFER_MOE_FP4=1 vllm serve nvidia/DeepSeek-R1-FP4 -tp 4 --max-model-len 8192 --no-enable-prefix-caching --port 8049 --speculative-config '{"method": "mtp", "num_speculative_tokens": 3}'` | Auto-reshapes query for spec decode<br>Handles uniform batches efficiently<br>Fallback to prefill for non-uniform requests | [#25984](https://github.com/vllm-project/vllm/pull/25984) Enable efficient spec decode with FlashInfer-MLA |
| **Communications** | **TP:**<br>  - AllReduce<br><br>**EP:**<br>  - DeepEP High-Throughput<br>  - DeepEP Low-Latency<br>  - PPLX<br>  - Naive All2All | DeepEP HT (needs evaluation) | **TP/EP All-Reduce**: After MOE combine when TP>1 or EP>1<br>Called via `maybe_all_reduce_tensor_model_parallel` (`vllm/model_executor/layers/fused_moe/layer.py:2284`)<br>Skipped if EP combine kernel already reduced<br>Tested with FP8 DeepSeek: `tests/quantization/test_blackwell_moe.py`<br><br>**EP Dispatch/Combine**: `VLLM_ALL2ALL_BACKEND` env var<br>**DeepEP HT**: `deepep_high_throughput` (Buffer: `VLLM_DEEPEP_BUFFER_SIZE_MB=1024`, SMs: `VLLM_DBO_COMM_SMS=20`)<br>**DeepEP LL**: `deepep_low_latency` (RDMA, 0 SMs)<br>**Others**: `pplx`, `naive`<br>Impl: `vllm/distributed/device_communicators/all2all.py`<br><br>**Note**: AG-RS backend is FP4-only (see FP4 section) | | [#21837](https://github.com/vllm-project/vllm/pull/21837) DeepEPHT Quant before Dispatch<br>[#24845](https://github.com/vllm-project/vllm/pull/24845) DBO DeepEP HT |

---

## FP4

| Component | Available Backends | Most Performant at Kernel Level | How to Invoke | Notes | Related PRs |
|-----------|-------------------|----------------------------------|---------------|-------|-------------|
| **Gemms** | - FlashInfer CUDNN (in review)<br>- FlashInfer TRTLLM<br>- FlashInfer CUTLASS<br>- vLLM CUTLASS (scaled_mm_fp4)<br>- Marlin GEMM (fallback) | FlashInfer CUDNN (~1% improvement) | **Backend Selection**: `VLLM_NVFP4_GEMM_BACKEND` env var (introduced in [#26107](https://github.com/vllm-project/vllm/pull/26107))<br>Options: `flashinfer-cudnn`, `flashinfer-trtllm`, `flashinfer-cutlass`, `cutlass`<br>Default: Uses first available (FlashInfer CUTLASS → FlashInfer TRTLLM → vLLM CUTLASS → Marlin)<br><br>**FlashInfer CUDNN**: `VLLM_NVFP4_GEMM_BACKEND=flashinfer-cudnn`<br>Requires: `pip install nvidia-cudnn-cu12 nvidia-cudnn-frontend`<br>Used in shared expert and output projection layers<br>Impl: `vllm/model_executor/layers/quantization/modelopt.py`<br><br>**FlashInfer TRTLLM**: `VLLM_NVFP4_GEMM_BACKEND=flashinfer-trtllm`<br><br>**FlashInfer CUTLASS**: `VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass`<br><br>**vLLM CUTLASS**: `VLLM_NVFP4_GEMM_BACKEND=cutlass`<br>Uses `cutlass_scaled_fp4_mm`<br>Fallback when FlashInfer not available<br><br>**Marlin**: Automatic fallback for non-SM100 GPUs | | [#26107](https://github.com/vllm-project/vllm/pull/26107) In Review: Add CUDNN FP4 GEMM |
| **MOE** | **TP/EP:**<br>  - FlashInfer CUTLASS<br>  - FlashInfer TRTLLM<br><br>**TP-only:**<br>  - vLLM CUTLASS FP4<br><br>**Non-SM100:**<br>  - Marlin FP4 | FlashInfer TRTLLM (latency) or CUTLASS (throughput) | **Backend Selection**: Controlled by two env vars<br>`VLLM_USE_FLASHINFER_MOE_FP4=1` to enable FlashInfer<br>`VLLM_FLASHINFER_MOE_BACKEND` to select variant<br><br>**FlashInfer CUTLASS**: `VLLM_USE_FLASHINFER_MOE_FP4=1 VLLM_FLASHINFER_MOE_BACKEND=throughput` (default)<br>High-throughput batch inference<br>Supports TP/EP<br>Uses `grouped_gemm_nt_masked` from flashinfer<br><br>**FlashInfer TRTLLM**: `VLLM_USE_FLASHINFER_MOE_FP4=1 VLLM_FLASHINFER_MOE_BACKEND=latency`<br>Low-latency inference<br>Supports TP/EP<br><br>**vLLM CUTLASS FP4**: Automatic fallback when FlashInfer unavailable<br>TP-only (no EP support)<br><br>**Marlin FP4**: Automatic fallback for non-SM100 GPUs | FlashInfer requires SM100 (Blackwell)<br>vLLM CUTLASS FP4 doesn't support Expert Parallelism | [#25968](https://github.com/vllm-project/vllm/pull/25968) TRTLLM NVFP4 MOE<br>[#25990](https://github.com/vllm-project/vllm/pull/25990) WIP: Masked Gemm for NVFP4 |
| **FMHA (Prefill)** | - FlashInfer CUTLASS<br>- TRT-LLM Ragged Attention (WIP)<br>- CUDNN<br>- FlashAttention-2 | FlashInfer CUTLASS (default on SM100) | **FlashInfer**: Default on SM100 (automatic)<br>Selection: `vllm/v1/attention/backends/mla/common.py:425-433`<br>Disable: `VLLM_DISABLE_FLASHINFER_PREFILL=1`<br><br>**TRT-LLM Ragged (WIP)**: Enable with `VLLM_USE_TRTLLM_RAGGED_DEEPSEEK_PREFILL=1`<br>Uses `flashinfer.prefill.trtllm_ragged_attention_deepseek`<br>Impl: `vllm/v1/attention/backends/mla/common.py:_run_prefill_context_chunk_trtllm_ragged`<br>For context chunks (non-causal)<br><br>**CUDNN**: Enable with `VLLM_USE_CUDNN_PREFILL=1`<br>Requires SM100 + nvidia artifactory<br><br>**FlashAttention-2**: Automatic fallback on SM100 | FlashInfer backend uses CUTLASS on SM100<br> | [#26063](https://github.com/vllm-project/vllm/pull/26063) Fix FI accuracy issue for MLA prefill<br>[#26397](https://github.com/vllm-project/vllm/pull/26397) WIP: TRT-LLM Ragged Attention |
| **MLA (Decode)** | **DP/TP Attention:**<br>  - FlashInfer MLA (TRTLLM)<br>  - CUTLASS MLA(default) | FLASHINFER MLA | **CUTLASS MLA**: Default on SM100 with block_size=128<br>Selection: `vllm/platforms/cuda.py:269-291`<br>Impl: `vllm/v1/attention/backends/mla/cutlass_mla.py`<br>Force: `VLLM_ATTENTION_BACKEND=CUTLASS_MLA`<br>Debug hangs: `FORCE_NUM_KV_SPLITS=1`<br><br>**FlashInfer MLA**: Default on SM100 with block_size=32 or 64<br>Force: `VLLM_ATTENTION_BACKEND=FLASHINFER_MLA`<br>Impl: `vllm/v1/attention/backends/mla/flashinfer_mla.py` | CUTLASS MLA: num_kv_splits limited to avoid hangs<br>FlashInfer MLA: LSE not yet returned (pending flashinfer#1566)<br>Both support FP8 compute with dynamic scales | [#26026](https://github.com/vllm-project/vllm/pull/26026) Fix CUTLASS MLA hang under load<br>[#26132](https://github.com/vllm-project/vllm/pull/26132) Improve DS MLA cache kernel<br>[#24705](https://github.com/vllm-project/vllm/pull/24705) Enable FP8 FlashInfer MLA decode<br>[#24385](https://github.com/vllm-project/vllm/pull/24385) Decode context parallelism with CUTLASS MLA |
| **MTP (Speculative Decode)** | - FlashInfer MLA (TRTLLM) | FlashInfer MLA (TRTLLM) | **FlashInfer MLA**: Uses `trtllm_batch_decode_with_kv_cache_mla`<br>Impl: `vllm/v1/attention/backends/mla/flashinfer_mla.py`<br>Enables `supports_uniform_spec_as_decode=True`<br>Supports multi-token prediction via q_len_per_request dimension<br><br>**Example**:<br>`VLLM_ATTENTION_BACKEND=FLASHINFER_MLA VLLM_USE_FLASHINFER_MOE_FP4=1 vllm serve nvidia/DeepSeek-R1-FP4 -tp 4 --max-model-len 8192 --no-enable-prefix-caching --port 8049 --speculative-config '{"method": "mtp", "num_speculative_tokens": 3}'` | Auto-reshapes query for spec decode<br>Handles uniform batches efficiently<br>Fallback to prefill for non-uniform requests | [#25984](https://github.com/vllm-project/vllm/pull/25984) Enable efficient spec decode with FlashInfer-MLA |
| **Communications** | **TP:**<br>  - AllReduce<br><br>**EP (FP4-only):**<br>  - AllGather-ReduceScatter (AG-RS)<br>  - FlashInfer All2Allv (MNNVL)<br>  - Naive All2All | AllGather-ReduceScatter (default for EP) | **TP/EP All-Reduce**: After MOE combine when TP>1 or EP>1<br><br>**EP Dispatch/Combine**: `VLLM_ALL2ALL_BACKEND` env var<br>**AG-RS** (default): `allgather_reducescatter` ([#23964](https://github.com/vllm-project/vllm/pull/23964))<br>Dispatch: `all_gatherv`, Combine: `reduce_scatterv`<br><br>**FlashInfer All2Allv**: `flashinfer_all2allv` (requires MNNVL hardware)<br><br>**Naive**: `naive`<br><br>See example usage section below | **Backend Compatibility:**<br>- With FlashInfer CUTLASS MOE: Can use All2Allv or AG-RS<br>- With FlashInfer TRTLLM MOE: Can only use AG-RS or Naive<br><br>**Recommendations:**<br>- All2Allv is for multinode setups<br>- AG-RS is generally better when Data Parallelism is enabled<br><br>AG-RS only works with FP4 quantization (not compatible with FP8) | [#23964](https://github.com/vllm-project/vllm/pull/23964) Enable AG/RS backend<br>[#21003](https://github.com/vllm-project/vllm/pull/21003) MNNVL all2allv |

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