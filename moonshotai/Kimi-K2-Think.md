# Kimi-K2-Thinking Usage Guide



[Kimi K2 Thinking](https://huggingface.co/moonshotai/Kimi-K2-Thinking/tree/main) is an advanced large language model created by [moonshotai](https://www.moonshot.ai/). It offers the following highlights:


- **Deep Thinking & Tool Orchestration**: End-to-end trained to interleave chain-of-thought reasoning with function calls, enabling autonomous research, coding, and writing workflows that last hundreds of steps without drift.
- **Native INT4 Quantization**: Quantization-Aware Training (QAT) is employed in post-training stage to achieve lossless 2x speed-up in low-latency mode.
- **Stable Long-Horizon Agency**: Maintains coherent goal-directed behavior across up to 200–300 consecutive tool invocations, surpassing prior models that degrade after 30–50 steps.


## Installing vLLM

```bash
uv venv
source .venv/bin/activate
uv pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly --extra-index-url https://download.pytorch.org/whl/cu129 --index-strategy unsafe-best-match # for xformers
```



## Launching Kimi-K2-Thinking with vLLM

You can use 8x H200/H20 to launch this model. See sections below for detailed launch arguments for low latency and high throughput scenarios


<details>
<summary>Low Latency Scenarios</summary>

run tensor-parallel like this:

```bash
vllm serve moonshotai/Kimi-K2-Thinking \
  --tensor-parallel-size 8 \
  --enable-auto-tool-choice \
  --tool-call-parser kimi_k2 \
  --reasoning-parser kimi_k2  \
  --trust-remote-code

```
The `--reasoning-parser` flag specifies the reasoning parser to use for extracting reasoning content from the model output.

</details>


<details>
<summary>High Throughput Scenarios</summary>

vLLM supports [Decode Context Parallel](https://docs.vllm.ai/en/latest/serving/context_parallel_deployment.html#decode-context-parallel), significant benefits in high throughput scenarios. You can enable DCP by adding `--decode-context-parallel-size number`, like:

```bash
vllm serve moonshotai/Kimi-K2-Thinking \
  --tensor-parallel-size 8 \
  --decode-context-parallel-size 8 \
  --enable-auto-tool-choice \
  --tool-call-parser kimi_k2 \
  --reasoning-parser kimi_k2  \
  --trust-remote-code

```
The `--reasoning-parser` flag specifies the reasoning parser to use for extracting reasoning content from the model output.

</details>




## Metrics

We tested the `GSM8K` accuracy of 2 types of launch scripts (TP8 vs TP8+DCP8)

- TP8

```bash
local-completions (model=moonshotai/Kimi-K2-Thinking,base_url=http://0.0.0.0:8000/v1/completions,tokenized_requests=False,tokenizer_backend=None,num_concurrent=32), gen_kwargs: (None), limit: None, num_fewshot: 5, batch_size: 1
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9416|±  |0.0065|
|     |       |strict-match    |     5|exact_match|↑  |0.9409|±  |0.0065|

```


- TP8+DCP8

```bash
local-completions (model=moonshotai/Kimi-K2-Thinking,temperature=0.0,base_url=http://0.0.0.0:8000/v1/completions,tokenized_requests=False,tokenizer_backend=None,num_concurrent=32,timeout=3000), gen_kwargs: (None), limit: None, num_fewshot: 5, batch_size: 1
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9386|±  |0.0066|
|     |       |strict-match    |     5|exact_match|↑  |0.9371|±  |0.0067|

```


## Benchmarking

We used the following script to benchmark `moonshotai/Kimi-K2-Thinking` on 8*H200.

```bash
vllm bench serve \
  --model moonshotai/Kimi-K2-Thinking \
  --dataset-name random \
  --random-input 8000 \
  --random-output 4000 \
  --request-rate 100 \
  --num-prompt 1000  \
  --trust-remote-code
```


We separately benchmarked the performance of TP8 and TP8+DCP8.


### TP8 Benchmark Output

```shell
============ Serving Benchmark Result ============
Successful requests:                     998       
Failed requests:                         2         
Request rate configured (RPS):           100.00    
Benchmark duration (s):                  800.26    
Total input tokens:                      7984000   
Total generated tokens:                  388750    
Request throughput (req/s):              1.25      
Output token throughput (tok/s):         485.78    
Peak output token throughput (tok/s):    2100.00   
Peak concurrent requests:                988.00    
Total Token throughput (tok/s):          10462.57  
---------------Time to First Token----------------
Mean TTFT (ms):                          271196.67 
Median TTFT (ms):                        227389.87 
P99 TTFT (ms):                           686294.46 
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          381.29    
Median TPOT (ms):                        473.04    
P99 TPOT (ms):                           578.64    
---------------Inter-token Latency----------------
Mean ITL (ms):                           111.22    
Median ITL (ms):                         38.04     
P99 ITL (ms):                            490.93    
==================================================

```
### TP8+DCP8 Benchmark Output

```shell
============ Serving Benchmark Result ============
Successful requests:                     994       
Failed requests:                         6         
Request rate configured (RPS):           100.00    
Benchmark duration (s):                  631.35    
Total input tokens:                      7952000   
Total generated tokens:                  438872    
Request throughput (req/s):              1.57      
Output token throughput (tok/s):         695.13    
Peak output token throughput (tok/s):    2618.00   
Peak concurrent requests:                988.00    
Total Token throughput (tok/s):          13290.35  
---------------Time to First Token----------------
Mean TTFT (ms):                          227780.13 
Median TTFT (ms):                        227055.20 
P99 TTFT (ms):                           451255.55 
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          398.60    
Median TPOT (ms):                        472.81    
P99 TPOT (ms):                           569.91    
---------------Inter-token Latency----------------
Mean ITL (ms):                           104.73    
Median ITL (ms):                         43.97     
P99 ITL (ms):                            483.37    
==================================================

```


## DCP Gain Analysis

Furthermore we analyzed the gain achieved by DCP.


| Metric | TP8 | TP8+DCP8 | Change | Improvement(%) |
|--------|-------------|-------------|--------|---------------|
| Request Throughput(req/s) | 1.25 | 1.57 | +0.32 | +25.6% |
| Output Token Throughput(tok/s) | 485.78 | 695.13 | +209.35 | +43.1% |
| Mean TTFT (sec) | 271.2 | 227.8 | -43.4 | +16.0% |
| Median TTFT (sec) | 227.4 | 227.1 | -0.3 | +0.1% |

You can observe from the service startup logs that the kv cache token number has increased by 8 times.

- TP8
```shell
(Worker_TP0 pid=591236) INFO 11-06 12:08:54 [gpu_worker.py:349] Available KV cache memory: 46.80 GiB
(EngineCore_DP0 pid=591074) INFO 11-06 12:08:55 [kv_cache_utils.py:1229] GPU KV cache size: 715,072 tokens
```
- TP8+DCP8
```shell
(Worker_TP0 pid=666845) INFO 11-06 15:34:58 [gpu_worker.py:349] Available KV cache memory: 46.80 GiB
(EngineCore_DP0 pid=666657) INFO 11-06 15:34:59 [kv_cache_utils.py:1224] Multiplying the GPU KV cache size by the dcp_world_size 8.
(EngineCore_DP0 pid=666657) INFO 11-06 15:34:59 [kv_cache_utils.py:1229] GPU KV cache size: 5,721,088 tokens

```

Enabling DCP delivers strong advantages (43% faster token generation, 26% higher throughput) with minimal drawbacks (marginal median latency improvement). We recommend reading our [DCP DOC](https://docs.vllm.ai/en/latest/serving/context_parallel_deployment.html#decode-context-parallel) and trying out DCP in your LLM workloads.


## AMD GPU Support 

Please follow the steps here to install and run Kimi-K2-Thinking models on AMD MI300X GPU.
### Step 1: Prepare Docker Environment
Pull the latest vllm docker:
```shell
docker pull rocm/vllm-dev:nightly
```
Launch the ROCm vLLM docker: 
```shell
docker run -it \
  --ipc=host \
  --network=host \
  --privileged \
  --cap-add=CAP_SYS_ADMIN \
  --device=/dev/kfd \
  --device=/dev/dri \
  --device=/dev/mem \
  --group-add video \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -v $(pwd):/work \
  -e SHELL=/bin/bash \
  --name Kimi-K2-Thinking \
  rocm/vllm-dev:nightly
```
### Step 2: Log in to Hugging Face
Huggingface login
```shell
huggingface-cli login
```

### Step 3: Start the vLLM server

Run the vllm online serving
Sample Command
```shell
SAFETENSORS_FAST_GPU=1 \
VLLM_USE_V1=1 \
VLLM_USE_TRITON_FLASH_ATTN=0 vllm serve moonshotai/Kimi-K2-Thinking \
  --tensor-parallel-size 8 \
  --no-enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser kimi_k2 \
  --reasoning-parser kimi_k2 \
  --trust-remote-code

```


### Step 4: Run Benchmark
Open a new terminal and run the following command to execute the benchmark script inside the container.
```shell
docker exec -it Kimi-K2-Thinking vllm bench serve \
  --model "moonshotai/Kimi-K2-Thinking" \
  --dataset-name random \
  --random-input-len 8192 \
  --random-output-len 1024 \
  --request-rate 10000 \
  --num-prompts 16 \
  --ignore-eos \
  --trust-remote-code 
```



