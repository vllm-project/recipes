# NVIDIA Nemotron-3-Nano-30B-A3B User Guide

This guide describes how to run Nemotron-3-Nano-30B-A3B using vLLM. There are FP8 and BF16 versions.

## Deployment Steps

We recommend using vLLM 0.12.0 release for full support. However, vLLM 0.11.2 also supports the model.

### Pull Docker Image

Pull the vLLM v0.12.0 release docker image.

`pull_image.sh`
```bash
# On x86_64 systems:
docker pull --platform linux/amd64 vllm/vllm-openai:v0.12.0
# On aarch64 systems:
# docker pull --platform linux/aarch64 vllm/vllm-openai:v0.12.0

docker tag vllm/vllm-openai:v0.12.0 vllm/vllm-openai:deploy
```

### DGX Spark Docker Image Build

Build container from source based on 0.12.0 release
https://github.com/vllm-project/vllm/blob/v0.12.0/docker/Dockerfile
```bash
DOCKER_BUILDKIT=1 docker build \
   --build-arg max_jobs=12 \
   --build-arg RUN_WHEEL_CHECK=false \
   --build-arg CUDA_VERSION=13.0.1 \
   --build-arg BUILD_BASE_IMAGE=nvidia/cuda:13.0.1-devel-ubuntu22.04 \
    --build-arg torch_cuda_arch_list='12.1' \
   --platform "linux/arm64" \
   --tag <tag name> \
   --target vllm-openai \
   --progress plain \
   -f docker/Dockerfile \
.
```

### Run Docker Container

Run the docker container using the docker image `vllm/vllm-openai:deploy`.

`run_container.sh`
```
docker run -e HF_TOKEN="$HF_TOKEN" -e HF_HOME="$HF_HOME" --ipc=host --gpus all --entrypoint "/bin/bash" --rm -it vllm/vllm-openai:deploy
```

Note: You can mount additional directories and paths using the `-v <local_path>:<path>` flag if needed, such as mounting the downloaded weight paths.

The `-e HF_TOKEN="$HF_TOKEN" -e HF_HOME="$HF_HOME"` flags are added so that the models are downloaded using your HuggingFace token and the downloaded models can be cached in $HF_HOME. Refer to [HuggingFace documentation](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables#hfhome) for more information about these environment variables and refer to [HuggingFace Quickstart guide](https://huggingface.co/docs/huggingface_hub/en/quick-start#authentication) about steps to generate your HuggingFace access token.



### Launch the vLLM Server

Below is an example command to launch the vLLM server with Nemotron-3-Nano-30B-A3B-BF16/FP8 model.

`launch_server.sh`
```
# Set up a few environment variables for better performance for Blackwell architecture.
# They will be removed when the performance optimizations have been verified and enabled by default.

# Supported dtypes for this model are: FP8, BF16
DTYPE="FP8"

if [ "$DTYPE" = "FP8" ]; then
    # On FP8 only - set KV cache dtype to FP8
    KV_CACHE_DTYPE="fp8"

    # Enable use of FlashInfer FP8 MoE
    export VLLM_USE_FLASHINFER_MOE_FP8=1
    export VLLM_FLASHINFER_MOE_BACKEND=throughput
else
    KV_CACHE_DTYPE="auto"
fi

# Launch the vLLM server
vllm serve nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-$DTYPE \
  --trust-remote-code \
  --async-scheduling \
  --kv-cache-dtype $KV_CACHE_DTYPE \
  --tensor-parallel-size 1 &
```

After the server is set up, the client can now send prompt requests to the server and receive results.


### DGX Spark vLLM Server Launch

Downloading the custom parser

```bash
wget https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/resolve/main/nano_v3_reasoning_parser.py
```

BF16 model variant

```python
vllm serve nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
 --max-num-seqs 8 \
  --tensor-parallel-size 1 \
  --max-model-len 262144 \
  --port 8000 \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser-plugin nano_v3_reasoning_parser.py \
  --reasoning-parser nano_v3
```

### Configs and Parameters

You can specify the IP address and the port that you would like to run the server with using these flags:

- `host`: IP address of the server. By default, it uses 127.0.0.1.
- `port`: The port to listen to by the server. By default, it uses port 8000.

Below are the config flags that we do not recommend changing or tuning with:

- `kv-cache-dtype`: KV cache data type. We recommend setting it to "fp8" when using the FP8 model, otherwise set to "auto".
- `async-scheduling`: Enable asynchronous scheduling to reduce the host overheads between decoding steps. We recommend always adding this flag for best performance.

Below are a few tunable parameters you can modify based on your serving requirements:

- `mamba-ssm-cache-dtype`: Mamba SSM cache data type. For best model accuracy set to `float32`. When using vLLM from `main` branch or any release newer than 0.12.0, setting to `float16` improves performance while degrading accuracy only slightly when compared to `float32`. The default value with this model until (and including) vLLM release 0.12.0 is `bfloat16`, in newer release or on `main` branch of vLLM, the default value would be either what's specified in the `mamba_ssm_cache_dtype` field in the model's HF `config.json`, or if it's not found there then `float16` would be used.
- `tensor-parallel-size`: Tensor parallelism size. Increasing this will increase the number of GPUs that are used for inference.
- `max-num-seqs`: Maximum number of sequences per batch.
  - By default, this is set to a large number like `1024` on GPUs with large memory sizes.
  - If the actual concurrency is smaller, setting this to a smaller number matching the max concurrency may improve the performance and improve the per-user latencies.
- `max-model-len`: Maximum number of total tokens, including the input tokens and output tokens, for each request.
  - By default, this is set to the maximum sequence length supported by the model.
  - If the actual input+output sequence length is shorter than the default, setting this to a smaller number may improve the performance.
  - For example, if the maximum input sequence length is 1024 tokens and maximum output sequence length is 1024, then this can be set to 2048 for better performance.

Refer to the "Balancing between Throughput and Latencies" about how to adjust these tunable parameters to meet your deployment requirements.
 
 ### Benchmarking Performance

To benchmark the performance, you can use the `vllm bench serve` command.

`run_performance.sh`
```
# Set DTYPE env var to match the benchmarked checkpoint (FP8 or BF16)
vllm bench serve \
  --host 0.0.0.0 \
  --port 8000 \
  --model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-$DTYPE \
  --trust-remote-code \
  --dataset-name random \
  --random-input-len 1024 \
  --random-output-len 1024 \
  --num-warmups 20 \
  --ignore-eos \
  --max-concurrency 1024 \
  --num-prompts 2048 \
  --save-result --result-filename vllm_benchmark_serving_results.json
```

Explanations for the flags:

- `--dataset-name`: Which dataset to use for benchmarking. We use a `random` dataset here.
- `--random-input-len`: Specifies the average input sequence length.
- `--random-output-len`: Specifies the average output sequence length.
- `--num-warmups`: Specifies the number of warmup requests. It helps to ensure the benchmark reflects the actual steady-state performance, ignoring the initial overheads.
- `--ignore-eos`: Disables early returning when eos (end-of-sentence) token is generated. 
- `--max-concurrency`: Maximum number of in-flight requests. We recommend matching this with the `--max-num-seqs` flag used to launch the server.
- `--num-prompts`: Total number of prompts used for performance benchmarking. We recommend setting it to at least five times of the `--max-concurrency` to measure the steady state performance.
- `--save-result --result-filename`: Output location for the performance benchmarking result.

### Interpreting Performance Benchmarking Output

Sample output by the `vllm bench serve` command, with the FP8 model on H200:

```
============ Serving Benchmark Result ============
Successful requests:                     2048
Failed requests:                         0
Maximum request concurrency:             1024
Benchmark duration (s):                  132.49
Total input tokens:                      2097155
Total generated tokens:                  2097152
Request throughput (req/s):              15.46
Output token throughput (tok/s):         15828.30
Peak output token throughput (tok/s):    21157.00
Peak concurrent requests:                1088.00
Total Token throughput (tok/s):          31656.63
---------------Time to First Token----------------
Mean TTFT (ms):                          4490.58
Median TTFT (ms):                        1534.84
P99 TTFT (ms):                           15465.31
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          59.45
Median TPOT (ms):                        61.04
P99 TPOT (ms):                           63.01
---------------Inter-token Latency----------------
Mean ITL (ms):                           59.45
Median ITL (ms):                         52.75
P99 ITL (ms):                            131.46
==================================================
```

Explanations for key metrics:

- `Median Time to First Token (TTFT)`: The typical time elapsed from when a request is sent until the first output token is generated.
- `Median Time Per Output Token (TPOT)`: The typical time required to generate each token after the first one.
- `Median Inter-Token Latency (ITL)`: The typical time delay between a response for the completion of one output token (or output tokens) and the next response for the completion of token(s).
- `Output token throughput`: The rate at which the system generates the output (generated) tokens.
- `Total Token Throughput`: The combined rate at which the system processes both input (prompt) tokens and output (generated) tokens.

### Balancing between Throughput and Latencies

In LLM inference, the "throughput" can be defined as the number of generated tokens per second (the `Output token throughput` metric above) or the number of processed tokens per second (the `Total Token Throughput` metric above). These two throughput metrics are highly correlated. We usually divide the throughput by the number of GPUs used to get the "per-GPU throughput" when comparing across different parallelism configurations. The higher per-GPU throughput is, the fewer GPUs are needed to serve the same amount of the incoming requests.

On the other hand, the “latency” can be defined as the latency from when a request is sent until the first output token is generated (the `TTFT` metric), the latency between two generated tokens after the first one has been generated (the `TPOT` metric), or the end-to-end latency from when a request is sent to when the final token is generated (the `E2EL` metric). The TTFT affects the E2EL more when the input (prompt) sequence lengths are much longer than the output (generated) sequence lengths, while the TPOT affects the E2EL more in the opposite cases.

To achieve higher throughput, tokens from multiple requests must be batched and processed together, but that increases the latencies. Therefore, a balance must be made between throughput and latencies depending on the deployment requirements.

The two main tunable configs for Nemotron Nano 3 are the `--tensor-parallel-size` (TP) and `--max-num-seqs` (BS). How they affect the throughput and latencies can be summarized as the following:

- At the same BS, higher TP typically results in lower latencies but also lower throughput.
- At the same TP size, higher BS typically results in higher throughput but worse latencies, but the maximum BS is limited by the amount of available GPU memory for the kv-cache after the weights are loaded.
- Therefore, increasing TP (which would lower the throughput at the same BS) may allow higher BS to run (which would increase the throughput), and the net throughput gain/loss depends on models and configurations.

Note that the statements above assume that the concurrency setting on the client side, like the `--max-concurrency` flag in the performance benchmarking command, matches the `--max-num-seqs` (BS) setting on the server side.
