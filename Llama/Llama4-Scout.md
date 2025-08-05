# Quick Start Recipe for Llama 4 Scout on vLLM - NVIDIA Blackwell Hardware

## Introduction

This quick start recipe provides step-by-step instructions for running the Llama 4 Scout Instruct model using vLLM with FP8 and NVFP4 quantization, optimized for NVIDIA Blackwell architecture GPUs. It covers the complete setup required; from accessing model weights and preparing the software environment to configuring vLLM parameters, launching the server, and validating inference output.

The recipe is intended for developers and practitioners seeking high-throughput or low-latency inference using NVIDIA’s accelerated stack — building a docker image with vLLM for model serving, FlashInfer for optimized CUDA kernels, and ModelOpt to enable FP8 and NVFP4 quantized execution.

## Access & Licensing

### License

To use Llama 4 Scout, you must first agree to Meta’s Llama 4 Scout Community License (https://ai.meta.com/resources/models-and-libraries/llama-downloads/). NVIDIA’s quantized versions (FP8 and FP4) are built on top of the base model and are available for research and commercial use under the same license.

### Weights

You only need to download one version of the model weights, depending on the precision in use:

- FP8 model for Blackwell: [nvidia/Llama-4-Scout-17B-16E-Instruct-FP8](https://huggingface.co/nvidia/Llama-4-Scout-17B-16E-Instruct-FP8)
- FP4 model for Blackwell: [nvidia/Llama-4-Scout-17B-16E-Instruct-FP4](https://huggingface.co/nvidia/Llama-4-Scout-17B-16E-Instruct-FP4)

No Hugging Face authentication token is required to download these weights.

Note on Quantization Choice:
For Blackwell, NVFP4 provides additional memory savings and throughput gains, but may require tuning to maintain accuracy on certain tasks.

## Prerequisites

- OS: Linux
- Drivers: CUDA Driver 575 or above
- GPU: Blackwell architecture
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html)

## Deployment Steps

### Build Docker Image

Build a docker image with vLLM and other dependencies installed. We will use both the [official vLLM Dockerfile](https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile) and additional `Dockerfile.nvidia` containing the necessary packages and environment settings for NVIDIA GPUs.

First, create `Dockerfile.nvidia` file as follows:

`Dockerfile.nvidia`
```
ARG base_image
FROM ${base_image}

WORKDIR /workspace

# Required environment variables for optimal performance on NVIDIA GPUs
# This will be removed when we enable these optimizations by default in VLLM.
# Use V1 engine
ENV VLLM_USE_V1=1
# Use FlashInfer backend for attentions
ENV VLLM_ATTENTION_BACKEND=FLASHINFER
# Use FlashInfer trtllm-gen attention kernels
ENV VLLM_USE_TRTLLM_DECODE_ATTENTION=1

# Install lm_eval that is compatible with the latest vLLM
RUN pip3 install --no-build-isolation "lm-eval[api] @ git+https://github.com/EleutherAI/lm-evaluation-harness@4f8195f"

ENTRYPOINT ["/bin/bash"]
```

Build the docker image named `vllm-llama-deploy` using the official vLLM Dockerfile and the Dockerfile.nvidia we just created.

`build_image.sh`
```
# Clone the vLLM GitHub repo and checkout the spcific commit.
git clone -b main --single-branch https://github.com/vllm-project/vllm.git
cd vllm
git checkout 055bd3978ededea015fb8f0cb6aa3cc48d84cde8

# Copy your Dockerfile.nvidia to docker/ directory
cp ../Dockerfile.nvidia docker/Dockerfile.nvidia

# Build the docker image using official vLLM Dockerfile and Dockerfile.nvidia.
DOCKER_BUILDKIT=1 docker build . \
        --file docker/Dockerfile \
        --target vllm-openai \
        --build-arg CUDA_VERSION=12.8.1 \
        --build-arg max_jobs=32 \
        --build-arg nvcc_threads=2 \
        --build-arg USE_SCCACHE=1 \
        --build-arg SCCACHE_S3_NO_CREDENTIALS=1 \
        --build-arg RUN_WHEEL_CHECK=false \
        --build-arg torch_cuda_arch_list="9.0+PTX 10.0+PTX" \
        --build-arg vllm_fa_cmake_gpu_arches="90-real;100-real" \
        -t vllm-official

DOCKER_BUILDKIT=1 docker build . \
        --file docker/Dockerfile.nvidia \
        --build-arg base_image=vllm-official \
        -t vllm-llama-deploy
```

### Run Docker Container

Run the docker container using the docker image `vllm-llama-deploy`.

`run_container.sh`
```
docker run -e HF_TOKEN="$HF_TOKEN" -e HF_HOME="$HF_HOME" --ipc=host --gpus all --rm -it vllm-llama-deploy
```

Note: You can mount additional directories and paths using the `-v <local_path>:<path>` flag if needed, such as mounting the downloaded weight paths.

The `-e HF_TOKEN="$HF_TOKEN" -e HF_HOME="$HF_HOME"` flags are added so that the models are downloaded using your HuggingFace token and the downloaded models can be cached in $HF_HOME. Refer to HuggingFace documentation for more information.

### Launch the vLLM Server

Below is an example command to launch the vLLM server with Llama-4-Scout-17B-16E-Instruct-FP8 model. The explanation of each flag is shown in the `Configs and Parameters` section.

`launch_server.sh`
```
vllm serve nvidia/Llama-4-Scout-17B-16E-Instruct-FP8 \
  --host 0.0.0.0 \
  --port 8080 \
  --tokenizer nvidia/Llama-4-Scout-17B-16E-Instruct-FP8 \
  --quantization modelopt \
  --kv-cache-dtype fp8 \
  --trust-remote-code \
  --gpu-memory-utilization 0.9 \
  --compilation-config '{"pass_config": {"enable_fi_allreduce_fusion": true}, "custom_ops": ["+rms_norm"], "level": 3}' \
  --enable-chunked-prefill \
  --async-scheduling \
  --no-enable-prefix-caching \
  --disable-log-requests \
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
- `--compilation-config`: Configuration for vLLM compilation stage. We recommend setting it to `'{"pass_config": {"enable_fi_allreduce_fusion": true}, "custom_ops": ["+rms_norm"], "level": 3}'` to enable all the necessary fusions for the best performance.
  - We are trying to enable these fusions by default so that this flag is no longer needed in the future.
- `--enable-chunked-prefill`: Enable chunked prefill stage. We recommend always adding this flag for best performance.
- `--async-scheduling`: Enable asynchronous scheduling to reduce the host overheads between decoding steps. We recommend always adding this flag for best performance on Blackwell architecture.
- `--no-enable-prefix-caching` Disable prefix caching. We recommend always adding this flag if running with synthetic dataset for consistent performance measurement.
- `--disable-log-requests`: Disable verbose logging from server.
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

## Validation & Expected Behavior

### Basic Test

After the vLLM server is set up and shows `Application startup complete`, you can send requests to the server 

`run_basic_test.sh`
```
curl http://0.0.0.0:8080/v1/completions -H "Content-Type: application/json" -d '{ "model": "nvidia/Llama-4-Scout-17B-16E-Instruct-FP8", "prompt": "San Francisco is a", "max_tokens": 20, "temperature": 0 }'
```

Here is an example response, showing that the vLLM server returns "*city known for its vibrant culture, stunning architecture, and iconic landmarks. One of the most recognizable symbols...*", completing the input sequence with up to 20 tokens.

```
{"id":"cmpl-70f1b1b3b84b4228900e66b6f6aac634","object":"text_completion","created":1753947009,"model":"nvidia/Llama-4-Scout-17B-16E-Instruct-FP8","choices":[{"index":0,"text":" city known for its vibrant culture, stunning architecture, and iconic landmarks. One of the most recognizable symbols","logprobs":null,"finish_reason":"length","stop_reason":null,"prompt_logprobs":null}],"service_tier":null,"system_fingerprint":null,"usage":{"prompt_tokens":5,"total_tokens":25,"completion_tokens":20,"prompt_tokens_details":null},"kv_transfer_params":null}
```

### Verify Accuracy

When the server is still running, we can run accuracy tests using lm_eval tool.

`run_accuracy.sh`
```
lm_eval \
  --model local-completions \
  --tasks gsm8k \
  --model_args \
base_url=http://0.0.0.0:8080/v1/completions,\
model=nvidia/Llama-4-Scout-17B-16E-Instruct-FP8,\
tokenized_requests=False,tokenizer_backend=None,\
num_concurrent=128,timeout=120,max_retries=5
```

Here is an example accuracy result with the nvidia/Llama-4-Scout-17B-16E-Instruct-FP8 model on one B200 GPU:

```
local-completions (base_url=http://0.0.0.0:8080/v1/completions,model=nvidia/Llama-4-Scout-17B-16E-Instruct-FP8,tokenized_requests=False,tokenizer_backend=None,num_concurrent=128,timeout=120,max_retries=5), gen_kwargs: (None), limit: None, num_fewshot: None, batch_size: 1
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9242|±  |0.0073|
|     |       |strict-match    |     5|exact_match|↑  |0.9075|±  |0.0080|
```

### Benchmarking Performance

To benchmark the performance, you can use the `benchmark_serving.py` script in the vLLM repository.

`run_performance.sh`
```
python3 /vllm-workspace/benchmarks/benchmark_serving.py \
  --host 0.0.0.0 \
  --port 8080 \
  --model nvidia/Llama-4-Scout-17B-16E-Instruct-FP8 \
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
- `Total Token Throughput`: The combined rate at which the system processes both input (prompt) tokens and output (generated) tokens. 

