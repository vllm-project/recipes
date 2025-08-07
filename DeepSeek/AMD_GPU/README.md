## AMD GPU Installation and Benchmarking Guide
#### Support Matrix 

##### GPU TYPE       
MI300X
##### DATA TYPE
FP8

#### Step by Step Guide
Please follow the steps here to install and run DeepSeek-R1 models on AMD MI300X GPU.
The model requires 8 * MI300X GPU.

#### Step 1
Verify the GPU environment: 
```shell
================================================== ROCm System Management Interface ==================================================
============================================================ Concise Info ============================================================
Device  Node  IDs              Temp        Power     Partitions          SCLK    MCLK    Fan  Perf              PwrCap  VRAM%  GPU%
              (DID,     GUID)  (Junction)  (Socket)  (Mem, Compute, ID)
======================================================================================================================================
0       9     0x74b5,   21947  51.0°C      163.0W    NPS1, SPX, 0        144Mhz  900Mhz  0%   perf_determinism  750.0W  0%     0%
1       8     0x74b5,   37820  45.0°C      154.0W    NPS1, SPX, 0        141Mhz  900Mhz  0%   perf_determinism  750.0W  0%     0%
2       7     0x74b5,   39350  46.0°C      163.0W    NPS1, SPX, 0        142Mhz  900Mhz  0%   perf_determinism  750.0W  0%     0%
3       6     0x74b5,   24497  53.0°C      172.0W    NPS1, SPX, 0        142Mhz  900Mhz  0%   perf_determinism  750.0W  0%     0%
4       5     0x74b5,   36258  51.0°C      169.0W    NPS1, SPX, 0        145Mhz  900Mhz  0%   perf_determinism  750.0W  0%     0%
5       4     0x74b5,   19365  44.0°C      158.0W    NPS1, SPX, 0        148Mhz  900Mhz  0%   perf_determinism  750.0W  0%     0%
6       3     0x74b5,   16815  53.0°C      167.0W    NPS1, SPX, 0        141Mhz  900Mhz  0%   perf_determinism  750.0W  0%     0%
7       2     0x74b5,   34728  46.0°C      165.0W    NPS1, SPX, 0        141Mhz  900Mhz  0%   perf_determinism  750.0W  0%     0%
======================================================================================================================================
```
Lock the GPU frequency
```shell
rocm-smi --setperfdeterminism 1900
```

### Step 2
Launch the Rocm-vllm docker: 
```shell
docker run -it --rm \
  --cap-add=SYS_PTRACE \
  -e SHELL=/bin/bash \
  --network=host \
  --security-opt seccomp=unconfined \
  --device=/dev/kfd \
  --device=/dev/dri \
  -v /:/workspace \
  --group-add video \
  --ipc=host \
  --name vllm_DS \
rocm/vllm:latest
```
Huggingface login
```shell
   pip install -U "huggingface_hub[cli]"
   huggingface-cli login 
```   
#### Step 3
##### FP8

Run the vllm online serving
Sample Command
```shell
NCCL_MIN_NCHANNELS=112 SAFETENSORS_FAST_GPU=1 VLLM_USE_V1=1 VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_RMSNORM=0 VLLM_ROCM_USE_AITER_MHA=0 VLLM_V1_USE_PREFILL_DECODE_ATTENTION=1 \
vllm serve deepseek-ai/DeepSeek-R1 \
--tensor-parallel-size 8 \
--max-model-len 65536 \
--max-num-seqs 1024 \
--max-num-batched-tokens 32768 \
--disable-log-requests \
--block-size 1 \
--compilation-config '{"full_cuda_graph":false}' \
--trust-remote-code
```

##### Tips: Users may modify the following parameters as needed.
--max-model-len=65536: A good sweet spot in most cases; preserves memory while still allowing long context.

--max-num-batched-tokens=32768: Balances throughput with manageable memory/latency.

If OOM errors or sluggish performance occur → decrease max-model-len (e.g., 32k or 8k) or reduce max-num-batched-tokens (e.g., 16k or 8k).For low latency needs, consider reducing max-num-batched-tokens.To maximize throughput and you have available VRAM, keep it high—but stay aware of latency trade-offs.

--max-num-seqs=1024: It affects throughput vs latency trade-offs:Higher values yield better throughput (more parallel requests) but may raise memory pressure and latency.Lower values reduce GPU memory footprint and latency, at the cost of throughput.


#### Step 4 
Open a new terminal, access the running Docker container, and execute the online serving benchmark script as follows:

```shell
docker exec -it vllm_DS /bin/bash 
python3 /app/vllm/benchmarks/benchmark_serving.py --model deepseek-ai/DeepSeek-R1 --dataset-name random --ignore-eos --num-prompts 500  --max-concurrency 256 --random-input-len 3200 --random-output-len 800  --percentile-metrics ttft,tpot,itl,e2el
```
```shell
Maximum request concurrency: 256
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [03:54<00:00,  2.14it/s]
============ Serving Benchmark Result ============
Successful requests:                     500
Benchmark duration (s):                  234.00
Total input tokens:                      1597574
Total generated tokens:                  400000
Request throughput (req/s):              2.14
Output token throughput (tok/s):         1709.39
Total Token throughput (tok/s):          8536.59
---------------Time to First Token----------------
Mean TTFT (ms):                          18547.34
Median TTFT (ms):                        5711.21
P99 TTFT (ms):                           59776.29
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          124.24
Median TPOT (ms):                        140.70
P99 TPOT (ms):                           144.12
---------------Inter-token Latency----------------
Mean ITL (ms):                           124.24
Median ITL (ms):                         71.91
P99 ITL (ms):                            2290.11
----------------End-to-end Latency----------------
Mean E2EL (ms):                          117819.02
Median E2EL (ms):                        118451.88
P99 E2EL (ms):                           174508.24
==================================================
```
