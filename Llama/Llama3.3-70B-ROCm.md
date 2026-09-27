# Quick Start Recipe for Llama 3.3 70B on vLLM - AMD MI355

## Introduction

This quick start recipe provides step-by-step instructions for running the Llama 3.3-70B Instruct model using vLLM with FP8 and FP4 quantization, optimized for AMD GPUs(MI355). It covers the complete setup required; from accessing model weights and preparing the software environment to configuring vLLM parameters, launching the server, and validating inference output.

The recipe is intended for developers and practitioners seeking high-throughput or low-latency inference using ROCm's accelerated stack—building a docker image with vLLM for model serving, FlashInfer for optimized ROCm kernels, and ModelOpt to enable FP8 and MXFP4 quantized execution.

## Access & Licensing

### License

To use Llama 3.3-70B, you must first agree to Meta’s Llama 3 Community License (https://ai.meta.com/resources/models-and-libraries/llama-downloads/). AMD’s quantized versions (FP8 and FP4) are built on top of the base model and are available for research and commercial use under the same license.

### Weights

You only need to download one version of the model weights, depending on the precision in use:

- FP8 model for MI355: [amd/Llama-3.3-70B-Instruct-FP8-KV](https://huggingface.co/amd/Llama-3.3-70B-Instruct-FP8-KV)
- FP4 model for MI355: [amd/Llama-3.3-70B-Instruct-MXFP4-Preview](https://huggingface.co/amd/Llama-3.3-70B-Instruct-MXFP4-Preview)

No Hugging Face authentication token is required to download these weights.

Note on Quantization Choice:
For MI355, FP8 offers the best performance for most workloads. MXFP4 provides additional memory savings and throughput gains, but may require tuning to maintain accuracy on certain tasks.

## Prerequisites

- OS: Linux
- GPU: MI355
- [ROCm docker setup](https://docs.vllm.ai/en/stable/getting_started/installation/gpu.html#set-up-using-docker)

## Deployment Steps

### Pull Docker Image

Pull the rocm/vllm-dev:nightly docker image.

`pull_image`
```
docker pull rocm/vllm-dev:nightly # to get the latest image
```

### Run Docker Container

Run the docker container using the docker image `rocm/vllm-dev:nightly`.

`run_container`
```
docker run -it --rm \
--network=host \
--group-add=video \
--ipc=host \
--cap-add=SYS_PTRACE \
--security-opt seccomp=unconfined \
--device /dev/kfd \
--device /dev/dri \
-v <path/to/your/models>:/app/models \
-e HF_HOME="/app/models" \
-e HF_TOKEN="$HF_TOKEN" \
rocm/vllm-dev:nightly
```

Note: You can mount additional directories and paths using the `-v <local_path>:<path>` flag if needed, such as mounting the downloaded weight paths.

The `-e HF_TOKEN="$HF_TOKEN" -e HF_HOME="$HF_HOME"` flags are added so that the models are downloaded using your HuggingFace token and the downloaded models can be cached in $HF_HOME. Refer to [HuggingFace documentation](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables#hfhome) for more information about these environment variables and refer to [HuggingFace Quickstart guide](https://huggingface.co/docs/huggingface_hub/en/quick-start#authentication) about steps to generate your HuggingFace access token.

### Prepare the AITER and vLLM packages

We suggest to install the latest vLLM and AITER to leverage all the optimizations available on ROCm plarforms.

`install AITER`

```
# uninstall aiter & vllm
pip uninstall -y aiter vllm
# clone aiter repo
git clone https://github.com/ROCm/aiter.git
cd aiter
git checkout dev/perf
git submodule sync && git submodule update --init --recursive
python3 setup.py install
```

`install vLLM`
```
# clone vllm
cd .. && git clone https://github.com/vllm-project/vllm.git
cd vllm
PYTORCH_ROCM_ARCH="gfx950" python3 setup.py develop
```

(Optional) Recommended to install when loading weight from local storage.
   - Install from wheel (only have python 3.10 wheel for now)

     ```shell
     python3 -m pip install https://github.com/EmbeddedLLM/fastsafetensors-rocm/releases/download/v0.1.15-rocm7-preview/fastsafetensors-0.1.15-cp310-cp310-linux_x86_64.whl
     ```

   - If you are using other python version follow the following steps (this works for ROCm 6.4.3 and ROCm 7.0):

     ```shell
     git clone https://github.com/EmbeddedLLM/fastsafetensors-rocm.git
     cd fastsafetensors-rocm
     python3 setup.py develop
     ```

   Add `--load-format fastsafetensors` to the `vllm serve` command to enable this feature.
   A 5-mins readup about fastsafetensors can be found here <https://github.com/EmbeddedLLM/fastsafetensors-rocm/blob/blog/blog_fastsafetensors_amd.md>

### Launch the vLLM Server

Below is an example command to launch the vLLM server with Llama-3.3-70B-Instruct-FP4/FP8 model.

`launch_server_llama_fp8.sh`
```
export SAFETENSORS_FAST_GPU=1
export VLLM_ROCM_USE_AITER=1
export VLLM_USE_V1=1
export NCCL_DEBUG=WARN
export VLLM_RPC_TIMEOUT=1800000

vllm serve amd/Llama-3.3-70B-Instruct-FP8-KV/ \
    --tensor-parallel-size 1 \
    --max-num-batched-tokens 32768 \
    --port 8000 \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --disable-log-requests \
    --gpu_memory_utilization 0.9 \
    --async-scheduling \
    --load-format safetensors \
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE", "custom_ops": ["-rms_norm", "-quant_fp8", "-silu_and_mul"]}' \
    --kv-cache-dtype fp8 \
```

`launch_server_llama_fp4.sh`
```
export SAFETENSORS_FAST_GPU=1
export VLLM_ROCM_USE_AITER=1
export VLLM_USE_V1=1
export NCCL_DEBUG=WARN
export VLLM_RPC_TIMEOUT=1800000
export VLLM_ROCM_USE_AITER_FP4_ASM_GEMM=1

vllm serve amd/Llama-3.3-70B-Instruct-MXFP4-Preview \
    --tensor-parallel-size 1 \
    --max-num-batched-tokens 32768 \
    --port 8000 \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --disable-log-requests \
    --gpu_memory_utilization 0.9 \
    --async-scheduling \
    --load-format safetensors \
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
    --kv-cache-dtype fp8 \
```

After the server is set up, the client can now send prompt requests to the server and receive results.

### Configs and Parameters

You can specify the IP address and the port that you would like to run the server with using these flags:

- `host`: IP address of the server. By default, it uses 127.0.0.1.
- `port`: The port to listen to by the server. By default, it uses port 8000.

Below are the config flags that we do not recommend changing or tuning with:

- `kv-cache-dtype`: Kv-cache data type. We recommend setting it to `fp8` for best performance.
- `compilation-config`: Configuration for vLLM compilation stage. For amd/Llama-3.3-70B-Instruct-FP8-KV, we recommend setting it to `'{"cudagraph_mode":"FULL_AND_PIECEWISE", "custom_ops": ["-rms_norm", "-quant_fp8", "-silu_and_mul"]}'` to enable all the necessary fusions for the best performance on MI355. For amd/Llama-3.3-70B-Instruct-MXFP4-Preview, we recommend setting it to `'{"cudagraph_mode":"FULL_AND_PIECEWISE"}'` to enable all the necessary fusions for the best performance on MI355.
- `async-scheduling`: Enable asynchronous scheduling to reduce the host overheads between decoding steps. We recommend always adding this flag for best performance.
- `no-enable-prefix-caching` Disable prefix caching. We recommend always adding this flag if running with synthetic dataset for consistent performance measurement.

Below are a few tunable parameters you can modify based on your serving requirements:

- `tensor-parallel-size`: Tensor parallelism size. Increasing this will increase the number of GPUs that are used for inference.
  - Set this to `1` to achieve the best throughput per GPU, and set this to `2`, `4`, or `8` to achieve better per-user latencies.
- `max-num-batched-tokens`: Maximum number of tokens per batch.
  - We recommend setting this to `32768`. Increasing this value may have slight performance improvements if the sequences have long input sequence lengths.


## Validation & Expected Behavior

### Basic Test

After the vLLM server is set up and shows `Application startup complete`, you can send requests to the server

### Verify Accuracy

When the server is still running, we can run accuracy tests using lm_eval tool.

`run_accuracy.sh`
```
# Install lm_eval that is compatible with the latest vLLM
pip3 install lm-eval[api]

# Run lm_eval
lm_eval \
    --model local-completions \
    --tasks gsm8k \
    --model_args model="$model",base_url=http://127.0.0.1:8000/v1/completions \
    --batch_size 100 \
```

Here is an example accuracy result with the amd/Llama-3.3-70B-Instruct-FP8-KV/ model on one MI355 GPU:

```
local-completions (model=/data/pretrained-models/amd/Llama-3.3-70B-Instruct-FP8-KV/,base_url=http://127.0.0.1:6789/v1/completions), gen_kwargs: (None), limit: None, num_fewshot: None, batch_size: 100
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9052|±  |0.0081|
|     |       |strict-match    |     5|exact_match|↑  |0.8575|±  |0.0096|
```

Here is an example accuracy result with the amd/Llama-3.3-70B-Instruct-MXFP4-Preview model on one MI355 GPU:

```
local-completions (model=/data/pretrained-models/amd/Llama-3.3-70B-Instruct-MXFP4-Preview/,base_url=http://127.0.0.1:6789/v1/completions), gen_kwargs: (None), limit: None, num_fewshot: None, batch_size: 100
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.8954|±  |0.0084|
|     |       |strict-match    |     5|exact_match|↑  |0.8317|±  |0.0103|
```

### Benchmarking Performance

To benchmark the performance, you can use the `vllm bench serve` command.

`run_performance.sh`
```
input_tokens=8192
output_tokens=1024
max_concurrency=64
num_prompts=128

# model="/data/pretrained-models/amd/Llama-3.3-70B-Instruct-FP8-KV/"
# model="/data/pretrained-models/amd/Llama-3.3-70B-Instruct-MXFP4-Preview/"

vllm bench serve \
    --host localhost \
    --port 8000 \
    --model ${model} \
    --dataset-name random \
    --random-input-len ${input_tokens} \
    --random-output-len ${output_tokens} \
    --max-concurrency ${max_concurrency} \
    --num-prompts ${num_prompts} \
    --percentile-metrics ttft,tpot,itl,e2el \
    --ignore-eos \
    --seed 123 \
```

Explanations for the flags:

- `--dataset-name`: Which dataset to use for benchmarking. We use a `random` dataset here.
- `--random-input-len`: Specifies the average input sequence length.
- `--random-output-len`: Specifies the average output sequence length.
- `--ignore-eos`: Disables early returning when eos (end-of-sentence) token is generated. This ensures that the output sequence lengths match our expected range.
- `--max-concurrency`: Maximum number of in-flight requests. We recommend matching this with the `--max-num-seqs` flag used to launch the server.
- `--num-prompts`: Total number of prompts used for performance benchmarking. We recommend setting it to at least five times of the `--max-concurrency` to measure the steady state performance.

### Interpreting Performance Benchmarking Output 

Sample output by the `vllm bench serve` command:

`amd/Llama-3.3-70B-Instruct-FP8-KV` TP1 8k/1k conc=64 performance on MI355
```
============ Serving Benchmark Result ============
Successful requests:                     128       
Failed requests:                         0         
Maximum request concurrency:             64        
Benchmark duration (s):                  149.97    
Total input tokens:                      1048448   
Total generated tokens:                  131072    
Request throughput (req/s):              0.85      
Output token throughput (tok/s):         873.98    
Peak output token throughput (tok/s):    1600.00   
Peak concurrent requests:                70.00     
Total Token throughput (tok/s):          7864.95   
---------------Time to First Token----------------
Mean TTFT (ms):                          11639.66  
Median TTFT (ms):                        7543.86   
P99 TTFT (ms):                           31797.74  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          61.81     
Median TPOT (ms):                        66.91     
P99 TPOT (ms):                           72.17     
---------------Inter-token Latency----------------
Mean ITL (ms):                           61.81     
Median ITL (ms):                         42.31     
P99 ITL (ms):                            1068.18   
----------------End-to-end Latency----------------
Mean E2EL (ms):                          74875.13  
Median E2EL (ms):                        74736.83  
P99 E2EL (ms):                           101320.40 
==================================================
```

`amd/Llama-3.3-70B-Instruct-MXFP4-Preview` TP1 8k/1k conc=64 performance on MI355
```
============ Serving Benchmark Result ============
Successful requests:                     128       
Failed requests:                         0         
Maximum request concurrency:             64        
Benchmark duration (s):                  131.16    
Total input tokens:                      1048448   
Total generated tokens:                  131072    
Request throughput (req/s):              0.98      
Output token throughput (tok/s):         999.30    
Peak output token throughput (tok/s):    1728.00   
Peak concurrent requests:                76.00     
Total Token throughput (tok/s):          8992.71   
---------------Time to First Token----------------
Mean TTFT (ms):                          10128.30  
Median TTFT (ms):                        6526.57   
P99 TTFT (ms):                           26033.25  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          54.10     
Median TPOT (ms):                        57.28     
P99 TPOT (ms):                           62.95     
---------------Inter-token Latency----------------
Mean ITL (ms):                           54.10     
Median ITL (ms):                         39.05     
P99 ITL (ms):                            411.51    
----------------End-to-end Latency----------------
Mean E2EL (ms):                          65475.58  
Median E2EL (ms):                        65305.68  
P99 E2EL (ms):                           84989.10  
==================================================
```

Explanations for key metrics:

- `Median Time to First Token (TTFT)`: The typical time elapsed from when a request is sent until the first output token is generated.
- `Median Time Per Output Token (TPOT)`: The typical time required to generate each token after the first one.
- `Median Inter-Token Latency (ITL)`: The typical time delay between the completion of one token and the completion of the next.
- `Median End-to-End Latency (E2EL)`: The typical total time from when a request is submitted until the final token of the response is received.
- `Output token throughput`: The rate at which the system generates the output (generated) tokens.
- `Total Token Throughput`: The combined rate at which the system processes both input (prompt) tokens and output (generated) tokens.
