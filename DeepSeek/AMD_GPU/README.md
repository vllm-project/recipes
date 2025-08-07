## AMD GPU Installation and Benchmarking Guide
#### Support Matrix 

##### GPU TYPE       
MI300X
##### DATA TYPE
FP8

#### Step by Step Guide
Please follow the steps here to install and run DeepSeek-R1 models on AMD MI300X GPU.
#### Step 1
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
#### Step 2
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
SAFETENSORS_FAST_GPU=1 VLLM_USE_V1=1 VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_RMSNORM=0 VLLM_ROCM_USE_AITER_MHA=0 VLLM_V1_USE_PREFILL_DECODE_ATTENTION=1 \
vllm serve deepseek-ai/DeepSeek-R1 \
--tensor-parallel-size 8 \
--max-model-len 32768 \
--max-num-seqs 1024 \
--max-num-batched-tokens 32768 \
--disable-log-requests \
--block-size 1 \
--compilation-config '{"full_cuda_graph":false}' \
--trust-remote-code
```
#### Step 4 
Open a new terminal, enter into the running docker and run the following benchmark script.
```shell
docker exec -it vllm_DS /bin/bash 
python3 /app/vllm/benchmarks/benchmark_serving.py --model deepseek-ai/DeepSeek-R1 --dataset-name random --ignore-eos --num-prompts 500  --max-concurrency 128 --random-input-len 3200 --random-output-len 800  --percentile-metrics ttft,tpot,itl,e2el
```
```shell
Maximum request concurrency: 128
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [04:43<00:00,  1.76it/s]
============ Serving Benchmark Result ============
Successful requests:                     500
Benchmark duration (s):                  283.98
Total input tokens:                      1597574
Total generated tokens:                  400000
Request throughput (req/s):              1.76
Output token throughput (tok/s):         1408.53
Total Token throughput (tok/s):          7034.09
---------------Time to First Token----------------
Mean TTFT (ms):                          7585.82
Median TTFT (ms):                        4689.25
P99 TTFT (ms):                           30544.70
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          80.02
Median TPOT (ms):                        83.26
P99 TPOT (ms):                           88.89
---------------Inter-token Latency----------------
Mean ITL (ms):                           80.02
Median ITL (ms):                         50.92
P99 ITL (ms):                            2263.85
----------------End-to-end Latency----------------
Mean E2EL (ms):                          71521.56
Median E2EL (ms):                        71237.75
P99 E2EL (ms):                           97463.28
==================================================
```
