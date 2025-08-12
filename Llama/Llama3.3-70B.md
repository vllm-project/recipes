# Quick Start Recipe for Llama 3.3 70B on vLLM - NVIDIA Blackwell & Hopper Hardware

## Introduction

This quick start recipe provides step-by-step instructions for running the Llama 3.3-70B Instruct model using vLLM with FP8 and NVFP4 quantization, optimized for NVIDIA GPUs, including Blackwell and Hopper architectures. It covers the complete setup required; from accessing model weights and preparing the software environment to configuring vLLM parameters, launching the server, and validating inference output.

The recipe is intended for developers and practitioners seeking high-throughput or low-latency inference using NVIDIA’s accelerated stack—building a docker image with vLLM for model serving, FlashInfer for optimized CUDA kernels, and ModelOpt to enable FP8 and NVFP4 quantized execution.


## Access & Licensing

### License

To use Llama 3.3-70B, you must first agree to Meta’s Llama 3 Community License (https://ai.meta.com/resources/models-and-libraries/llama-downloads/). NVIDIA’s quantized versions (FP8 and FP4) are built on top of the base model and are available for research and commercial use under the same license.

### Weights

You only need to download one version of the model weights, depending on the precision in use:

- FP8 model for Blackwell/Hopper: [nvidia/Llama-3.3-70B-Instruct-FP8](https://huggingface.co/nvidia/Llama-3.3-70B-Instruct-FP8)
- FP4 model for Blackwell: [nvidia/Llama-3.3-70B-Instruct-FP4](https://huggingface.co/nvidia/Llama-3.3-70B-Instruct-FP4)

No Hugging Face authentication token is required to download these weights.

Note on Quantization Choice:
For Hopper, FP8 offers the best performance for most workloads. For Blackwell, NVFP4 provides additional memory savings and throughput gains, but may require tuning to maintain accuracy on certain tasks.

## Prerequisites

- OS: Linux
- Drivers: CUDA Driver 575 or above
- GPU: Blackwell architecture or Hopper Architecture
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html)

## Deployment Steps

### Build Docker Image

Build a docker image with vLLM using the official vLLM Dockerfile at a specific commit (`dc5e4a653c859573dfcca99f1b0141c2db9f94cc`) on the main branch. This commit contains more performance optimizations compared to the latest official vLLM docker image (`vllm/vllm-openai:latest`).

`build_image.sh`
```
# Clone the vLLM GitHub repo and checkout the spcific commit.
git clone -b main --single-branch https://github.com/vllm-project/vllm.git
cd vllm
git checkout dc5e4a653c859573dfcca99f1b0141c2db9f94cc

# Build the docker image using official vLLM Dockerfile.
DOCKER_BUILDKIT=1 docker build . \
        --file docker/Dockerfile \
        --target vllm-openai \
        --build-arg CUDA_VERSION=12.8.1 \
        --build-arg max_jobs=32 \
        --build-arg nvcc_threads=2 \
        --build-arg RUN_WHEEL_CHECK=false \
        --build-arg torch_cuda_arch_list="9.0+PTX 10.0+PTX" \
        --build-arg vllm_fa_cmake_gpu_arches="90-real;100-real" \
        -t vllm/vllm-openai:deploy
```

Note: building the docker image may use lots of CPU threads and CPU memory. If you build the docker image on machines with fewer CPU cores or less CPU memory, please reduce the value of `max_jobs`.

### Run Docker Container

Run the docker container using the docker image `vllm/vllm-openai:deploy`.

`run_container.sh`
```
docker run -e HF_TOKEN="$HF_TOKEN" -e HF_HOME="$HF_HOME" --ipc=host --gpus all --entrypoint "/bin/bash" --rm -it vllm/vllm-openai:deploy
```

Note: You can mount additional directories and paths using the `-v <local_path>:<path>` flag if needed, such as mounting the downloaded weight paths.

The `-e HF_TOKEN="$HF_TOKEN" -e HF_HOME="$HF_HOME"` flags are added so that the models are downloaded using your HuggingFace token and the downloaded models can be cached in $HF_HOME. Refer to [HuggingFace documentation](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables#hfhome) for more information about these environment variables and refer to [HuggingFace Quickstart guide](https://huggingface.co/docs/huggingface_hub/en/quick-start#authentication) about steps to generate your HuggingFace access token.

### Launch the vLLM Server

Below is an example command to launch the vLLM server with Llama-3.3-70B-Instruct-FP4/FP8 model. The explanation of each flag is shown in the "Configs and Parameters" section.

`launch_server.sh`
```
# Set up a few environment variables for better performance for Blackwell architecture.
# They will be removed when the performance optimizations have been verified and enabled by default.
COMPUTE_CAPABILITY=$(nvidia-smi -i 0 --query-gpu=compute_cap --format=csv,noheader)
if [ "$COMPUTE_CAPABILITY" = "10.0" ]; then
    # Use FlashInfer backend for attentions
    export VLLM_ATTENTION_BACKEND=FLASHINFER
    # Use FlashInfer trtllm-gen attention kernels
    export VLLM_USE_TRTLLM_ATTENTION=1
    # Enable async scheduling
    ASYNC_SCHEDULING_FLAG="--async-scheduling"
    # Enable FlashInfer fusions
    FUSION_FLAG='{"pass_config":{"enable_fi_allreduce_fusion":true,"enable_noop":true},"custom_ops":["+quant_fp8","+rms_norm"],"full_cuda_graph":true}'
    # Use FP4 for Blackwell architecture
    DTYPE="FP4"
else
    # Disable async scheduling on Hopper architecture due to vLLM limitations
    ASYNC_SCHEDULING_FLAG=""
    # Disable FlashInfer fusions since they are not supported on Hopper architecture
    FUSION_FLAG="{}"
    # Use FP8 for Hopper architecture
    DTYPE="FP8"
fi

# Launch the vLLM server
vllm serve nvidia/Llama-3.3-70B-Instruct-$DTYPE \
  --host 0.0.0.0 \
  --port 8080 \
  --tokenizer nvidia/Llama-3.3-70B-Instruct-$DTYPE \
  --kv-cache-dtype fp8 \
  --trust-remote-code \
  --gpu-memory-utilization 0.9 \
  --compilation-config $FUSION_FLAG \
  $ASYNC_SCHEDULING_FLAG \
  --enable-chunked-prefill \
  --no-enable-prefix-caching \
  --pipeline-parallel-size 1 \
  --tensor-parallel-size 1 \
  --max-num-seqs 512 \
  --max-num-batched-tokens 8192 \
  --max-model-len 9216 &

```

After the server is set up, the client can now send prompt requests to the server and receive results.

### Configs and Parameters

You can specify the IP address and the port that you would like to run the server with using these flags:

- `--host`: IP address of the server. 
- `--port`: The port to listen to by the server.

Below are the config flags that we do not recommend changing or tuning with:

- `--tokenizer`: Specify the path to the model file.
- `--quantization`: Must be `modelopt` for FP8 model and `modelopt_fp4` for FP4 model.
- `--kv-cache-dtype`: Kv-cache data type. We recommend setting it to `fp8` for best performance.
- `--trust-remote-code`: Trust the model code.
- `--gpu-memory-utilization`: The fraction of GPU memory to be used for the model executor. We recommend setting it to `0.9` to use up to 90% of the GPU memory.
- `--compilation-config`: Configuration for vLLM compilation stage. We recommend setting it to `'{"pass_config":{"enable_fi_allreduce_fusion":true,"enable_noop":true},"custom_ops":["+quant_fp8","+rms_norm"],"full_cuda_graph":true}'` to enable all the necessary fusions for the best performance on Blackwell architecture. However, this feature is not supported on Hopper architecture yet.
  - We are trying to enable these fusions by default so that this flag is no longer needed in the future.
- `--enable-chunked-prefill`: Enable chunked prefill stage. We recommend always adding this flag for best performance.
- `--async-scheduling`: Enable asynchronous scheduling to reduce the host overheads between decoding steps. We recommend always adding this flag for best performance on Blackwell architecture. However, this feature is not supported on Hopper architecture yet.
- `--no-enable-prefix-caching` Disable prefix caching. We recommend always adding this flag if running with synthetic dataset for consistent performance measurement.
- `--pipeline-parallel-size`: Pipeline parallelism size. We recommend setting it to `1` for best performance.

Below are a few tunable parameters you can modify based on your serving requirements:

- `--tensor-parallel-size`: Tensor parallelism size. Increasing this will increase the number of GPUs that are used for inference.
  - Set this to `1` to achieve the best throughput, and set this to `2`, `4`, or `8` to achieve better per-user latencies.
- `--max-num-seqs`: Maximum number of sequences per batch.
  - Set this to a large number like `512` to achieve the best throughput, and set this to a small number like `16` to achieve better per-user latencies.
- `--max-num-batched-tokens`: Maximum number of tokens per batch.
  - We recommend setting this to `8192`. Increasing this value may have slight performance improvements if the sequences have long input sequence lengths.
- `--max-model-len`: Maximum number of total tokens, including the input tokens and output tokens, for each request.
  - This must be set to a larger number if the expected input/output sequence lengths are large.
  - For example, if the maximum input sequence length is 1024 tokens and maximum output sequence length is 1024, then this must be set to at least 2048.

Refer to the "Balancing between Throughput and Latencies" about how to adjust these tunable parameters to meet your deployment requirements.

## Validation & Expected Behavior

### Basic Test

After the vLLM server is set up and shows `Application startup complete`, you can send requests to the server 

`run_basic_test.sh`
```
curl http://0.0.0.0:8080/v1/completions -H "Content-Type: application/json" -d '{ "model": "nvidia/Llama-3.3-70B-Instruct-FP4", "prompt": "San Francisco is a", "max_tokens": 20, "temperature": 0 }'
```

Here is an example response, showing that the vLLM server returns "*city that is known for its vibrant culture, stunning architecture, and breathtaking natural beauty. From the iconic...*", completing the input sequence with up to 20 tokens.

```
{"id":"cmpl-4493992056a146f0a65363462eee6218","object":"text_completion","created":1754989721,"model":"nvidia/Llama-3.3-70B-Instruct-FP4","choices":[{"index":0,"text":" city that is known for its vibrant culture, stunning architecture, and breathtaking natural beauty. From the iconic","logprobs":null,"finish_reason":"length","stop_reason":null,"prompt_logprobs":null}],"service_tier":null,"system_fingerprint":null,"usage":{"prompt_tokens":5,"total_tokens":25,"completion_tokens":20,"prompt_tokens_details":null},"kv_transfer_params":null}
```

### Verify Accuracy

When the server is still running, we can run accuracy tests using lm_eval tool.

`run_accuracy.sh`
```
# Install lm_eval that is compatible with the latest vLLM
pip3 install lm-eval[api]==0.4.9.1

# Run lm_eval
lm_eval \
  --model local-completions \
  --tasks gsm8k \
  --model_args \
base_url=http://0.0.0.0:8080/v1/completions,\
model=nvidia/Llama-3.3-70B-Instruct-FP4,\
tokenized_requests=False,tokenizer_backend=None,\
num_concurrent=128,timeout=120,max_retries=5
```

Here is an example accuracy result with the nvidia/Llama-3.3-70B-Instruct-FP4 model on one B200 GPU:

```
local-completions (base_url=http://0.0.0.0:8080/v1/completions,model=nvidia/Llama-3.3-70B-Instruct-FP4,tokenized_requests=False,tokenizer_backend=None,num_concurrent=128,timeout=120,max_retries=5), gen_kwa
rgs: (None), limit: None, num_fewshot: None, batch_size: 1
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9272|±  |0.0072|
|     |       |strict-match    |     5|exact_match|↑  |0.6293|±  |0.0133|
```

### Benchmarking Performance

To benchmark the performance, you can use the `vllm bench serve` command.

`run_performance.sh`
```
vllm bench serve \
  --host 0.0.0.0 \
  --port 8080 \
  --model nvidia/Llama-3.3-70B-Instruct-FP4 \
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

### Interpreting `benchmark_serving.py` Output 

Sample output by the `benchmark_serving.py` script:

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

The two main tunable configs for Llama 3.3 70B are the `--tensor-parallel-size` (TP) and `--max-num-seqs` (BS). How they affect the throughput and latencies can be summarized as the following:

- At the same BS, higher TP typically results in lower latencies but also lower throughput.
- At the same TP size, higher BS typically results in higher throughput but worse latencies, but the maximum BS is limited by the amount of available GPU memory for the kv-cache after the weights are loaded.
- Therefore, increasing TP (which would lower the throughput at the same BS) may allow higher BS to run (which would increase the throughput), and the net throughput gain/loss depends on models and configurations.

Note that the statements above assume that the concurrency setting on the client side, like the `--max-concurrency` flag in the performance benchmarking command, matches the `--max-num-seqs` (BS) setting on the server side.

Below are the recommended configs for different throughput-latency scenarios on B200 GPUs:

- **Max Throughput**: Set TP to 1, and increase BS to the maximum possible value without triggering out-of-memory errors.
- **Min Latency**: Set TP to 4 or 8, and set BS to a small value (like `8`) that meets the latency requirements.
- **Balanced**: Set TP to 2 and set BS to 128.

Finally, another minor tunable config is the `--max-num-batched-tokens` flag which controls how many tokens can be batched together within a forward iteration. We recommend setting this to `8192` which works well for most cases. Increasing it to `16384` may result in slightly higher throughput and lower TTFT latencies, with a more uneven distribution of the TPOT latencies since some output tokens may be generated with more prefill-stage tokens in the same batches.
