## `gpt-oss` vLLM Usage Guide

`gpt-oss-20b` and `gpt-oss-120b` are powerful reasoning models open-sourced by OpenAI. 
In vLLM, you can run it on NVIDIA H100, H200, B200 as well as MI300x, MI325x, MI355x and Radeon AI PRO R9700. 
We are actively working on ensuring this model can work on Ampere, Ada Lovelace, and RTX 5090. 
Specifically, vLLM optimizes for `gpt-oss` family of models with

* **Flexible parallelism options**: the model can be sharded across 2, 4, 8 GPUs, scaling throughput.
* **High performance attention and MoE kernels**: attention kernel is specifically optimized for the attention sinks mechanism and sliding window shapes.   
* **Asynchronous scheduling**: optimizing for maximum utilization and high throughput by overlapping CPU operations with GPU operations. 

This is a living document and we welcome contributions, corrections, and creation of new recipes! 

## Quickstart

### Installation from pre-built wheels

We recommend using the official [vLLM 0.10.1 release](https://github.com/vllm-project/vllm/releases/tag/v0.10.1) as your starting point. Create a new virtual environment and install the official release:

```
uv venv
source .venv/bin/activate
uv pip install vllm==0.10.1 --torch-backend=auto
```

We also provide a docker container with all the dependencies built in

```
docker run --gpus all \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:v0.10.1 \
    --model openai/gpt-oss-20b
```

### A100

GPT-OSS works on Ampere devices by default, using the `TRITON_ATTN` attention backend and Marlin MXFP4 MoE:

* `--async-scheduling` can be enabled for higher performance. Currently it is not compatible with structured output.

```
# openai/gpt-oss-20b should run on a single A100
vllm serve openai/gpt-oss-20b --async-scheduling 

# gpt-oss-120b will fit on a single A100 (80GB), but scaling it to higher TP sizes can help with throughput
vllm serve openai/gpt-oss-120b --async-scheduling
vllm serve openai/gpt-oss-120b --tensor-parallel-size 2 --async-scheduling
vllm serve openai/gpt-oss-120b --tensor-parallel-size 4 --async-scheduling
```

### H100 & H200

GPT-OSS works on Hopper devices by default, using the FlashAttention3 backend and Marlin MXFP4 MoE:

* `--async-scheduling` can be enabled for higher performance. Currently it is not compatible with structured output.
* We recommend TP=2 for H100 and H200 as the best performance tradeoff point. 

```
# openai/gpt-oss-20b should run in single GPU
vllm serve openai/gpt-oss-20b --async-scheduling 

# gpt-oss-120b will fit in a single H100/H200, but scaling it to higher TP sizes can help with throughput
vllm serve openai/gpt-oss-120b --async-scheduling
vllm serve openai/gpt-oss-120b --tensor-parallel-size 2 --async-scheduling
vllm serve openai/gpt-oss-120b --tensor-parallel-size 4 --async-scheduling
```

### B200

NVIDIA Blackwell requires installation of [FlashInfer library](https://github.com/flashinfer-ai/flashinfer), so please install the extra `vllm[flashinfer]`.

```
uv pip install vllm[flashinfer]==0.10.1 --torch-backend=auto
```

We recommend TP=1 as a starting point for a performant option. We are actively working on the performance of vLLM on Blackwell. 

```
# Pick only one out of the two for MoE implementation
# bf16 activation for MoE. matching reference precision (default).
export VLLM_USE_FLASHINFER_MXFP4_BF16_MOE=1 
# mxfp8 activation for MoE. faster, but higher risk for accuracy.
export VLLM_USE_FLASHINFER_MXFP4_MOE=1 

# openai/gpt-oss-20b
vllm serve openai/gpt-oss-20b --async-scheduling 

# gpt-oss-120b 
vllm serve openai/gpt-oss-120b --async-scheduling
vllm serve openai/gpt-oss-120b --tensor-parallel-size 2 --async-scheduling
vllm serve openai/gpt-oss-120b --tensor-parallel-size 4 --async-scheduling
```

### AMD

ROCm supports OpenAI gpt-oss-120b or gpt-oss-20b models on these 3 different GPUs on day one, along with the pre-built docker containers:

* gfx950: MI350x series, `rocm/vllm-dev:open-mi355-08052025`  
* gfx942: MI300x/MI325 series, `rocm/vllm-dev:open-mi300-08052025`  
* gfx1201: Radeon AI PRO R9700, `rocm/vllm-dev:open-r9700-08052025`

To run the container:

```
alias drun='sudo docker run -it --network=host --device=/dev/kfd --device=/dev/dri --group-add=video --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --shm-size 32G -v /data:/data -v $HOME:/myhome -w /myhome'

drun rocm/vllm-dev:open-mi300-08052025
```

For MI300x and R9700:

```
export VLLM_ROCM_USE_AITER=1
export VLLM_USE_AITER_UNIFIED_ATTENTION=1
export VLLM_ROCM_USE_AITER_MHA=0

vllm serve openai/gpt-oss-120b --compilation-config '{"full_cuda_graph": true}' 
```

For MI355x:

```
# MoE preshuffle, fusion and Triton GEMM flags
export VLLM_USE_AITER_TRITON_FUSED_SPLIT_QKV_ROPE=1
export VLLM_USE_AITER_TRITON_FUSED_ADD_RMSNORM_PAD=1
export VLLM_USE_AITER_TRITON_GEMM=1
export VLLM_ROCM_USE_AITER=1
export VLLM_USE_AITER_UNIFIED_ATTENTION=1
export VLLM_ROCM_USE_AITER_MHA=0
export TRITON_HIP_PRESHUFFLE_SCALES=1

vllm serve openai/gpt-oss-120b --compilation-config '{"compile_sizes": [1, 2, 4, 8, 16, 24, 32, 64, 128, 256, 4096, 8192], "full_cuda_graph": true}' --block-size 64 
```

#### Known Issues
- When you encounter this error `The link interface of target "torch::nvtoolsext" contains: CUDA::nvToolsExt but the target was not found.` Please double check your pytorch version has suffix `+cu128`.
- If the output you see is garbage, that might be because you haven't properly set `CUDA_HOME`. The CUDA version needs to be greater than or equal to 12.8 and must be the same for installation and serving. 

## Usage

Once the `vllm serve` runs and `INFO: Application startup complete` has been displayed, you can send requests using HTTP request or OpenAI SDK to the following endpoints:

* `/v1/responses` endpoint can perform tool use (browsing, python, mcp) in between chain-of-thought and deliver a final response. This endpoint leverages the `openai-harmony` library for input rendering and output parsing. Stateful operation and full streaming API are work in progress. Responses API is recommended by OpenAI as the way to interact with this model.
* `/v1/chat/completions` endpoint offers a familiar interface to this model. No tool will be invoked but reasoning and final text output will be returned structurally. You can also set the parameter `include_reasoning: false` in request parameter to skip CoT being part of the output.
* `/v1/completions` endpoint is the endpoint for a simple input output interface without any sorts of template rendering. 

All endpoints accept `stream: true` as part of the operations to enable incremental token streaming. Please note that vLLM currently does not cover the full scope of responses API, for more detail, please see Limitation section below. 

### Tool Use

One premier feature of gpt-oss is the ability to call tools directly, called "built-in tools". In vLLM, we offer several options:

* By default, we integrate with the reference library's browser (with `ExaBackend`) and demo Python interpreter via docker container. In order to use the search backend, you need to get access to [exa.ai](http://exa.ai) and put `EXA_API_KEY=` as an environment variable. For Python, either have docker available, or set `PYTHON_EXECUTION_BACKEND=dangerously_use_uv` to dangerously allow execution of model generated code snippets to be executed on the same machine. Please note that `PYTHON_EXECUTION_BACKEND=dangerously_use_uv` needs `gpt-oss>=0.0.5`.

```bash
uv pip install gpt-oss

vllm serve ... --tool-server demo
```

* Please note that the default options are simply for demo purposes. For production usage, vLLM itself can act as MCP client to multiple services. 
Here is an [example tool server](https://github.com/openai/gpt-oss/tree/main/gpt-oss-mcp-server) that vLLM can work with, they wrap the demo tools: 

```bash
mcp run -t sse browser_server.py:mcp
mcp run -t sse python_server.py:mcp

vllm serve ... --tool-server ip-1:port-1,ip-2:port-2
```

The URLs are expected to be MCP SSE servers that implement `instructions` in server info and well documented tools. The tools will be injected into the system prompt for the model to enable them.

### Function calling

vLLM also supports calling user-defined functions. Make sure to run your gpt-oss models with the following arguments.

```bash
vllm serve ... --tool-call-parser openai --enable-auto-tool-choice
```

## Accuracy Evaluation Panels

OpenAI recommends using the gpt-oss reference library to perform evaluation.

First, deploy the model with vLLM:

```bash
# Example deployment on 8xH100
vllm serve openai/gpt-oss-120b \
  --tensor_parallel_size 8 \
  --max-model-len 131072 \
  --max-num-batched-tokens 10240 \
  --max-num-seqs 128 \
  --gpu-memory-utilization 0.85 \
  --no-enable-prefix-caching
```

Then, run the evaluation with gpt-oss. The following command will run all the 3 reasoning effort levels.

```bash
mkdir -p /tmp/gpqa_openai
OPENAI_API_KEY=empty python -m gpt_oss.evals --model openai/gpt-oss-120b --eval gpqa --n-threads 128
```

To eval on AIME2025, change `gpqa` to `aime25`.

Here is the score we were able to reproduce without tool use, and we encourage you to try reproducing it as well!
We’ve observed that the numbers may vary slightly across runs, so feel free to run the evaluation multiple times to get a sense of the variance.
For a quick correctness check, we recommend starting with the low reasoning effort setting (`--reasoning-effort low`), which should complete within minutes.

Model: 120B

| Reasoning Effort | GPQA | AIME25 |
| :---- | :---- | :---- |
| Low  | 65.3 | 51.2 |
| Mid  | 72.4 | 79.6 |
| High  | 79.4 | 93.0 |

Model: 20B

| Reasoning Effort | GPQA | AIME25 |
| :---- | :---- | :---- |
| Low  | 56.8 | 38.8 |
| Mid  | 67.5 | 75.0 |
| High  | 70.9 | 85.8  |

## Detailed Recipe for NVIDIA Blackwell & Hopper Hardware

This chapter includes more instructions about running gpt-oss-120b on NVIDIA Blackwell & Hopper hardware to get the additional performance optimizations compared to the Quickstart chapter above.

### Pull Docker Image

Pull the vLLM v0.10.2 release docker image.

`pull_image.sh`
```
# On x86_64 systems:
docker pull --platform linux/amd64 vllm/vllm-openai:v0.10.2
# On aarch64 systems:
# docker pull --platform linux/aarch64 vllm/vllm-openai:v0.10.2

docker tag vllm/vllm-openai:v0.10.2 vllm/vllm-openai:deploy
```

### Run Docker Container

Run the docker container using the docker image `vllm/vllm-openai:deploy`.

`run_container.sh`
```
docker run -e HF_TOKEN="$HF_TOKEN" -e HF_HOME="$HF_HOME" --ipc=host --gpus all --entrypoint "/bin/bash" --rm -it vllm/vllm-openai:deploy
```

Note: You can mount additional directories and paths using the `-v <local_path>:<path>` flag if needed, such as mounting the downloaded weight paths.

The `-e HF_TOKEN="$HF_TOKEN" -e HF_HOME="$HF_HOME"` flags are added so that the models are downloaded using your HuggingFace token and the downloaded models can be cached in $HF_HOME. Refer to [HuggingFace documentation](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables#hfhome) for more information about these environment variables and refer to [HuggingFace Quickstart guide](https://huggingface.co/docs/huggingface_hub/en/quick-start#authentication) about steps to generate your HuggingFace access token.

### Prepare the Config File

Prepare the config YAML file to configure vLLM. Below shows the recommended config files for Blackwell and Hopper architectures, respectively. These config files have also been uploaded to the [vLLM recipe repository](https://github.com/vllm-project/recipes/tree/main/OpenAI). The explanation of each config is shown in the "Configs and Parameters" section.

`GPT-OSS_Blackwell.yaml`
```
compilation-config: '{"pass_config":{"enable_fi_allreduce_fusion":true,"enable_noop":true},"custom_ops":["+rms_norm"],"cudagraph_mode":"FULL_AND_PIECEWISE"}'
async-scheduling: true
no-enable-prefix-caching: true
cuda-graph-sizes: 2048
max-num-batched-tokens: 8192
max-model-len: 10240
```

`GPT-OSS_Hopper.yaml`
```
async-scheduling: true
no-enable-prefix-caching: true
cuda-graph-sizes: 2048
max-num-batched-tokens: 8192
max-model-len: 10240
```

### Launch the vLLM Server

Below is an example command to launch the vLLM server with openai/gpt-oss-120b model.

`launch_server.sh`
```
# Set up a few environment variables for better performance for Blackwell architecture.
# They will be removed when the performance optimizations have been verified and enabled by default.
COMPUTE_CAPABILITY=$(nvidia-smi -i 0 --query-gpu=compute_cap --format=csv,noheader)
if [ "$COMPUTE_CAPABILITY" = "10.0" ]; then
    # Set AR+Norm fusion thresholds
    export VLLM_FLASHINFER_ALLREDUCE_FUSION_THRESHOLDS_MB='{"2":32,"4":32,"8":8}'
    # Use FlashInfer MXFP4+MXFP8 MoE
    export VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1
    # Select the config file for Blackwell architecture.
    YAML_CONFIG="GPT-OSS_Blackwell.yaml"
else
    # Select the config file for Hopper architecture.
    YAML_CONFIG="GPT-OSS_Hopper.yaml"
fi

# Launch the vLLM server
vllm serve openai/gpt-oss-120b \
  --config ${YAML_CONFIG} \
  --tensor-parallel-size 1 \
  --max-num-seqs 512 &
```

After the server is set up, the client can now send prompt requests to the server and receive results.

### Configs and Parameters

You can specify the IP address and the port that you would like to run the server with using these flags/configs:

- `host`: IP address of the server. By default, it uses 127.0.0.1.
- `port`: The port to listen to by the server. By default, it uses port 8000.

Below are the config flags that we do not recommend changing or tuning with:

- `compilation-config`: Configuration for vLLM compilation stage. We recommend setting it to `'{"pass_config":{"enable_fi_allreduce_fusion":true,"enable_noop":true},"custom_ops":["+rms_norm"],"cudagraph_mode":"FULL_AND_PIECEWISE"}'` to enable all the necessary fusions for the best performance on Blackwell architecture. However, this feature is not supported on Hopper architecture yet.
- `async-scheduling`: Enable asynchronous scheduling to reduce the host overheads between decoding steps. We recommend always adding this flag for best performance.
- `no-enable-prefix-caching`: Disable prefix caching. We recommend always adding this flag if running with synthetic dataset for consistent performance measurement.
- `cuda-graph-sizes`: Specify the max size for cuda graphs. We recommend setting this to 2048 to leverage the benefit of cuda graphs while not using too much GPU memory.

Below are a few tunable parameters you can modify based on your serving requirements:

- `tensor-parallel-size`: Tensor parallelism size. Increasing this will increase the number of GPUs that are used for inference.
  - Set this to `1` to achieve the best throughput per GPU, and set this to `2`, `4`, or `8` to achieve better per-user latencies.
- `max-num-seqs`: Maximum number of sequences per batch.
  - Set this to a large number like `512` to achieve the best throughput, and set this to a small number like `16` to achieve better per-user latencies.
- `max-num-batched-tokens`: Maximum number of tokens per batch.
  - We recommend setting this to `8192`. Increasing this value may have slight performance improvements if the sequences have long input sequence lengths.
- `max-model-len`: Maximum number of total tokens, including the input tokens and output tokens, for each request.
  - This must be set to a larger number if the expected input/output sequence lengths are large.
  - For example, if the maximum input sequence length is 1024 tokens and maximum output sequence length is 1024, then this must be set to at least 2048.

Refer to the "Balancing between Throughput and Latencies" about how to adjust these tunable parameters to meet your deployment requirements.

### Benchmarking Performance

To benchmark the performance, you can use the `vllm bench serve` command.

`run_performance.sh`
```
vllm bench serve \
  --host 0.0.0.0 \
  --port 8000 \
  --model openai/gpt-oss-120b \
  --trust-remote-code \
  --dataset-name random \
  --random-input-len 1024 \
  --random-output-len 1024 \
  --ignore-eos \
  --max-concurrency 512 \
  --num-prompts 2560 \
  --save-result --result-filename vllm_benchmark_serving_results.json
```

Explanations for the flags:

- `--dataset-name`: Which dataset to use for benchmarking. We use a `random` dataset here.
- `--random-input-len`: Specifies the average input sequence length.
- `--random-output-len`: Specifies the average output sequence length.
- `--ignore-eos`: Disables early returning when eos (end-of-sentence) token is generated. This ensures that the output sequence lengths match our expected range.
- `--max-concurrency`: Maximum number of in-flight requests. We recommend matching this with the `--max-num-seqs` flag used to launch the server.
- `--num-prompts`: Total number of prompts used for performance benchmarking. We recommend setting it to at least five times of the `--max-concurrency` to measure the steady state performance.
- `--save-result --result-filename`: Output location for the performance benchmarking result.

### Interpreting Performance Benchmarking Output

Sample output by the `vllm bench serve` command:

```
============ Serving Benchmark Result ============
Successful requests:                     xxxxxx
Benchmark duration (s):                  xxx.xx
Total input tokens:                      xxxxxx
Total generated tokens:                  xxxxxx
Request throughput (req/s):              xxx.xx
Output token throughput (tok/s):         xxx.xx
Total Token throughput (tok/s):          xxx.xx
---------------Time to First Token----------------
Mean TTFT (ms):                          xxx.xx
Median TTFT (ms):                        xxx.xx
P99 TTFT (ms):                           xxx.xx
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          xxx.xx
Median TPOT (ms):                        xxx.xx
P99 TPOT (ms):                           xxx.xx
---------------Inter-token Latency----------------
Mean ITL (ms):                           xxx.xx
Median ITL (ms):                         xxx.xx
P99 ITL (ms):                            xxx.xx
----------------End-to-end Latency----------------
Mean E2EL (ms):                          xxx.xx
Median E2EL (ms):                        xxx.xx
P99 E2EL (ms):                           xxx.xx
==================================================
```

Explanations for key metrics:

- `Median Time to First Token (TTFT)`: The typical time elapsed from when a request is sent until the first output token is generated.
- `Median Time Per Output Token (TPOT)`: The typical time required to generate each token after the first one.
- `Median Inter-Token Latency (ITL)`: The typical time delay between the completion of one token and the completion of the next.
- `Median End-to-End Latency (E2EL)`: The typical total time from when a request is submitted until the final token of the response is received.
- `Output token throughput`: The rate at which the system generates the output (generated) tokens.
- `Total Token Throughput`: The combined rate at which the system processes both input (prompt) tokens and output (generated) tokens.

### Balancing between Throughput and Latencies

In LLM inference, the "throughput" can be defined as the number of generated tokens per second (the `Output token throughput` metric above) or the number of processed tokens per second (the `Total Token Throughput` metric above). These two throughput metrics are highly correlated. We usually divide the throughput by the number of GPUs used to get the "per-GPU throughput" when comparing across different parallelism configurations. The higher per-GPU throughput is, the fewer GPUs are needed to serve the same amount of the incoming requests.

On the other hand, the “latency” can be defined as the latency from when a request is sent until the first output token is generated (the `TTFT` metric), the latency between two generated tokens after the first one has been generated (the `TPOT` metric), or the end-to-end latency from when a request is sent to when the final token is generated (the `E2EL` metric). The TTFT affects the E2EL more when the input (prompt) sequence lengths are much longer than the output (generated) sequence lengths, while the TPOT affects the E2EL more in the opposite cases.

To achieve higher throughput, tokens from multiple requests must be batched and processed together, but that increases the latencies. Therefore, a balance must be made between throughput and latencies depending on the deployment requirements.

The two main tunable configs for GPT-OSS are the `--tensor-parallel-size` (TP) and `--max-num-seqs` (BS). How they affect the throughput and latencies can be summarized as the following:

- At the same BS, higher TP typically results in lower latencies but also lower throughput.
- At the same TP size, higher BS typically results in higher throughput but worse latencies, but the maximum BS is limited by the amount of available GPU memory for the kv-cache after the weights are loaded.
- Therefore, increasing TP (which would lower the throughput at the same BS) may allow higher BS to run (which would increase the throughput), and the net throughput gain/loss depends on models and configurations.

Note that the statements above assume that the concurrency setting on the client side, like the `--max-concurrency` flag in the performance benchmarking command, matches the `--max-num-seqs` (BS) setting on the server side.

Below are the recommended configs for different throughput-latency scenarios on B200 GPUs:

- **Max Throughput**: Set TP to 1, and increase BS to the maximum possible value without exceeding KV cache capacity.
- **Min Latency**: Set TP to 4 or 8, and set BS to a small value (like `8`) that meets the latency requirements.
- **Balanced**: Set TP to 2 and set BS to 128.

Finally, another minor tunable config is the `--max-num-batched-tokens` flag which controls how many tokens can be batched together within a forward iteration. We recommend setting this to `8192` which works well for most cases. Increasing it to `16384` may result in slightly higher throughput and lower TTFT latencies, with a more uneven distribution of the TPOT latencies since some output tokens may be generated with more prefill-stage tokens in the same batches.

## Known Limitations

* On H100 using tensor parallel size 1, default gpu memory utilization, and batched token will cause CUDA Out-of-memory. When running tp1, please increase your gpu memory utilization or lower batched token

```
vllm serve openai/gpt-oss-120b --gpu-memory-utilization 0.95 --max-num-batched-tokens 1024
```

* When running TP2 on H100, set your gpu memory utilization below 0.95 as that will also cause OOM
* Responses API has several limitations at the current moment; we strongly welcome contribution and maintenance of this service in vLLM
* Usage accounting is currently broken and only returns all zeros.
* Annotations (citing URLs from search results) are not supported.
* Truncation by `max_tokens` might not be able to preserve partial chunks.
* Streaming is fairly barebone at the moment, for example:
  * Item id and indexing needs more work
  * Tool invocation and output are not properly streamed, rather batched.
  * Proper error handling is missing. 

## Troubleshooting

- Attention sink dtype error on Blackwell:

```
  ERROR 08-05 07:31:10 [multiproc_executor.py:559]     assert sinks.dtype == torch.float32, "Sinks must be of type float32"  
  **(VllmWorker TP0 pid=174579)** ERROR 08-05 07:31:10 [multiproc_executor.py:559]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^  
  **(VllmWorker TP0 pid=174579)** ERROR 08-05 07:31:10 [multiproc_executor.py:559] AssertionError: Sinks must be of type float32
```

**Solution: Please refer to Blackwell section to check if related environment variables are added.**

- Triton issue related to `tl.language` not defined:

**Solution: Make sure there's no other triton installed in your environment (pytorch-triton, etc).**

- Run into `openai_harmony.HarmonyError: error downloading or loading vocab file: failed to download or load vocab` error

**Solution: This is caused by a bug in openai_harmony code. This can be worked around by downloading the tiktoken encoding files in advance and setting the TIKTOKEN_ENCODINGS_BASE environment variable. See [this GitHub issue](https://github.com/openai/harmony/pull/41) for more information.**

```
mkdir -p tiktoken_encodings
wget -O tiktoken_encodings/o200k_base.tiktoken "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"
wget -O tiktoken_encodings/cl100k_base.tiktoken "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
export TIKTOKEN_ENCODINGS_BASE=${PWD}/tiktoken_encodings
```

## Harmony Format Support

Below is the support matrix for Harmony format.

Meaning:

* ✅ = Full compatibility
* ❌ = No compatibility

| API Type| Basic Text Generation | Structured Output | Builtin Tools with demo Tool Server | Builtin Tools with MCP | Function Calling |
| :----: | :----: | :----: | :----: | :----: | :----: |
| Response API | ✅ | ✅ | ✅ | ✅ | ✅ |
| Response API with Background Mode | ✅ | ✅ | ✅ | ✅ | ✅ |
| Response API with Streaming | ✅ | ✅ | ✅ | ✅ | ❌ |
| Chat Completion API | ✅ | ✅ | ❌ | ❌ | ✅ |
| Chat Completion API with Streaming | ✅ | ✅ | ❌ | ❌ | ✅ |


If you want to use offline inference, you can treat vLLM as a token-in-token-out service and pass in tokens that are already formatted with Harmony.

For function calling, only tool_choice="auto" is supported.
