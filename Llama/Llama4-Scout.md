# Quick Start Recipe for Llama 4 Scout on vLLM - NVIDIA Blackwell & Hopper Hardware

## Introduction

This quick start recipe provides step-by-step instructions for running the Llama 4 Scout Instruct model using vLLM with FP8 and NVFP4 quantization, optimized for NVIDIA GPUs, including Blackwell and Hopper architectures. It covers the complete setup required; from accessing model weights and preparing the software environment to configuring vLLM parameters, launching the server, and validating inference output.

The recipe is intended for developers and practitioners seeking high-throughput or low-latency inference using NVIDIA’s accelerated stack—building a docker image with vLLM for model serving, FlashInfer for optimized CUDA kernels, and ModelOpt to enable FP8 and NVFP4 quantized execution.

## Access & Licensing

### License

To use Llama 4 Scout, you must first agree to Meta’s Llama 4 Scout Community License (https://ai.meta.com/resources/models-and-libraries/llama-downloads/). NVIDIA’s quantized versions (FP8 and FP4) are built on top of the base model and are available for research and commercial use under the same license.

### Weights

You only need to download one version of the model weights, depending on the precision in use:

- FP8 model for Blackwell/Hopper: [nvidia/Llama-4-Scout-17B-16E-Instruct-FP8](https://huggingface.co/nvidia/Llama-4-Scout-17B-16E-Instruct-FP8)
- FP4 model for Blackwell: [nvidia/Llama-4-Scout-17B-16E-Instruct-FP4](https://huggingface.co/nvidia/Llama-4-Scout-17B-16E-Instruct-FP4)

No Hugging Face authentication token is required to download these weights.

Note on Quantization Choice:
For Hopper, FP8 offers the best performance for most workloads. For Blackwell, NVFP4 provides additional memory savings and throughput gains, but may require tuning to maintain accuracy on certain tasks.

## Prerequisites

- OS: Linux
- Drivers: CUDA Driver 575 or above
- GPU: Blackwell architecture or Hopper Architecture
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html)

## Deployment Steps

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

Prepare the config YAML file to configure vLLM. Below shows the recommended config files for Blackwell and Hopper architectures, respectively. These config files have also been uploaded to the vLLM recipe repository. The explanation of each config is shown in the "Configs and Parameters" section.

`Llama4_Blackwell.yaml`
```
kv-cache-dtype: fp8
compilation-config: '{"pass_config":{"enable_fi_allreduce_fusion":true,"enable_attn_fusion":true,"enable_noop":true},"custom_ops":["+quant_fp8","+rms_norm"],"cudagraph_mode":"FULL_DECODE_ONLY","splitting_ops":[]}'
async-scheduling: true
no-enable-prefix-caching: true
max-num-batched-tokens: 8192
max-model-len: 10240
```

`Llama4_Hopper.yaml`
```
kv-cache-dtype: fp8
async-scheduling: true
no-enable-prefix-caching: true
max-num-batched-tokens: 8192
max-model-len: 10240
```

### Launch the vLLM Server

Below is an example command to launch the vLLM server with Llama-4-Scout-17B-16E-Instruct-FP4/FP8 model.

`launch_server.sh`
```
# Set up a few environment variables for better performance for Blackwell architecture.
# They will be removed when the performance optimizations have been verified and enabled by default.
COMPUTE_CAPABILITY=$(nvidia-smi -i 0 --query-gpu=compute_cap --format=csv,noheader)
if [ "$COMPUTE_CAPABILITY" = "10.0" ]; then
    # Set AR+Norm fusion thresholds
    export VLLM_FLASHINFER_ALLREDUCE_FUSION_THRESHOLDS_MB='{"2":32,"4":32,"8":8}'
    # Use FlashInfer FP8/FP4 MoE
    export VLLM_USE_FLASHINFER_MOE_FP8=1
    export VLLM_USE_FLASHINFER_MOE_FP4=1
    # Optionally, set this env var for low concurrency scenarios to reduce MoE latency.
    # export VLLM_FLASHINFER_MOE_BACKEND=latency
    # Use FP4 for Blackwell architecture
    # Change this to FP8 to run FP8 on Blackwell architecture
    DTYPE="FP4"
    # Select the config file for Blackwell architecture.
    YAML_CONFIG="Llama4_Blackwell.yaml"
else
    # Use FP8 for Hopper architecture
    DTYPE="FP8"
    # Select the config file for Hopper architecture.
    YAML_CONFIG="Llama4_Hopper.yaml"
fi

# Launch the vLLM server
vllm serve nvidia/Llama-4-Scout-Instruct-$DTYPE \
  --config ${YAML_CONFIG} \
  --tensor-parallel-size 1 \
  --max-num-seqs 512 &
```

After the server is set up, the client can now send prompt requests to the server and receive results.

### Configs and Parameters

You can specify the IP address and the port that you would like to run the server with using these flags:

- `host`: IP address of the server. By default, it uses 127.0.0.1.
- `port`: The port to listen to by the server. By default, it uses port 8000.

Below are the config flags that we do not recommend changing or tuning with:

- `kv-cache-dtype`: Kv-cache data type. We recommend setting it to `fp8` for best performance.
- `compilation-config`: Configuration for vLLM compilation stage. We recommend setting it to `'{"pass_config":{"enable_fi_allreduce_fusion":true,"enable_attn_fusion":true,"enable_noop":true},"custom_ops":["+quant_fp8","+rms_norm"],"cudagraph_mode":"FULL_DECODE_ONLY","splitting_ops":[]}'` to enable all the necessary fusions for the best performance on Blackwell architecture. However, this feature is not supported on Hopper architecture yet.
- `async-scheduling`: Enable asynchronous scheduling to reduce the host overheads between decoding steps. We recommend always adding this flag for best performance.
- `no-enable-prefix-caching` Disable prefix caching. We recommend always adding this flag if running with synthetic dataset for consistent performance measurement.

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


## Validation & Expected Behavior

### Basic Test

After the vLLM server is set up and shows `Application startup complete`, you can send requests to the server

`run_basic_test.sh`
```
curl http://0.0.0.0:8000/v1/completions -H "Content-Type: application/json" -d '{ "model": "nvidia/Llama-4-Scout-17B-16E-Instruct-FP8", "prompt": "San Francisco is a", "max_tokens": 20, "temperature": 0 }'
```

Here is an example response, showing that the vLLM server returns "*vibrant and eclectic city that offers something for everyone. From iconic landmarks to cultural attractions, here are some...*", completing the input sequence with up to 20 tokens.

```
{"id":"cmpl-11f390a191304552b233ad103d9c7f69","object":"text_completion","created":1754989296,"model":"nvidia/Llama-4-Scout-17B-16E-Instruct-FP4","choices":[{"index":0,"text":" vibrant and eclectic city that offers something for everyone. From iconic landmarks to cultural attractions, here are some","logprobs":null,"finish_reason":"length","stop_reason":null,"prompt_logprobs":null}],"service_tier":null,"system_fingerprint":null,"usage":{"prompt_tokens":5,"total_tokens":25,"completion_tokens":20,"prompt_tokens_details":null},"kv_transfer_params":null}
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
base_url=http://0.0.0.0:8000/v1/completions,\
model=nvidia/Llama-4-Scout-17B-16E-Instruct-FP4,\
tokenized_requests=False,tokenizer_backend=None,\
num_concurrent=128,timeout=120,max_retries=5
```

Here is an example accuracy result with the nvidia/Llama-4-Scout-17B-16E-Instruct-FP4 model on one B200 GPU:

```
local-completions (base_url=http://0.0.0.0:8000/v1/completions,model=nvidia/Llama-4-Scout-17B-16E-Instruct-FP4,tokenized_requests=False,tokenizer_backend=None,num_concurrent=128,timeout=120,max_retries=5), gen_kwargs: (None), limit: None, num_fewshot: None, batch_size: 1
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.8946|±  |0.0085|
|     |       |strict-match    |     5|exact_match|↑  |0.8779|±  |0.0090|
```

### Benchmarking Performance

To benchmark the performance, you can use the `vllm bench serve` command.

`run_performance.sh`
```
vllm bench serve \
  --host 0.0.0.0 \
  --port 8000 \
  --model nvidia/Llama-4-Scout-17B-16E-Instruct-FP4 \
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

The two main tunable configs for  Llama 4 Scout are the `--tensor-parallel-size` (TP) and `--max-num-seqs` (BS). How they affect the throughput and latencies can be summarized as the following:

- At the same BS, higher TP typically results in lower latencies but also lower throughput.
- At the same TP size, higher BS typically results in higher throughput but worse latencies, but the maximum BS is limited by the amount of available GPU memory for the kv-cache after the weights are loaded.
- Therefore, increasing TP (which would lower the throughput at the same BS) may allow higher BS to run (which would increase the throughput), and the net throughput gain/loss depends on models and configurations.

Note that the statements above assume that the concurrency setting on the client side, like the `--max-concurrency` flag in the performance benchmarking command, matches the `--max-num-seqs` (BS) setting on the server side.

Below are the recommended configs for different throughput-latency scenarios on B200 GPUs:

- **Max Throughput**: Set TP to 1 for FP4 model and 2 for FP8 model, and increase BS to the maximum possible value without exceeding KV cache capacity.
- **Min Latency**: Set TP to 4 or 8, and set BS to a small value (like `8`) that meets the latency requirements.
- **Balanced**: Set TP to 2 and set BS to 128.

Finally, another minor tunable config is the `--max-num-batched-tokens` flag which controls how many tokens can be batched together within a forward iteration. We recommend setting this to `8192` which works well for most cases. Increasing it to `16384` may result in slightly higher throughput and lower TTFT latencies, with a more uneven distribution of the TPOT latencies since some output tokens may be generated with more prefill-stage tokens in the same batches.
