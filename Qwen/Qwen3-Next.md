# Qwen3-Next Usage Guide

[Qwen3-Next](https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd&from=research.latest-advancements-list) is an advanced large language model created by the Qwen team from Alibaba Cloud. It features several key improvements:

* A hybrid attention mechanism
* A highly sparse Mixture-of-Experts (MoE) structure
* Training-stability-friendly optimizations
* A multi-token prediction mechanism for faster inference

## Installing vLLM

```bash
uv venv
source .venv/bin/activate
uv pip install -U vllm --torch-backend auto
```

## Launching Qwen3-Next with vLLM

You can use 4x H200/H20 or 4x A100/A800 GPUs to launch this model.

### Basic Multi-GPU Setup

```bash
vllm serve Qwen/Qwen3-Next-80B-A3B-Instruct \
  --tensor-parallel-size 4 \
  --served-model-name qwen3-next 

```

If you encounter `torch.AcceleratorError: CUDA error: an illegal memory access was encountered`, you can add `--compilation_config.cudagraph_mode=PIECEWISE` to the startup parameters to resolve this issue. This IMA error may occur in Data Parallel (DP) mode.


### For FP8 model

We can accelerate the performance on SM100 machines using the FP8 FlashInfer TRTLLM MoE kernel.

```bash
VLLM_USE_FLASHINFER_MOE_FP8=1 \
VLLM_FLASHINFER_MOE_BACKEND=latency \
VLLM_USE_DEEP_GEMM=0 \
VLLM_USE_TRTLLM_ATTENTION=0 \
VLLM_ATTENTION_BACKEND=FLASH_ATTN \
vllm serve Qwen/Qwen3-Next-80B-A3B-Instruct-FP8 \
--tensor-parallel-size 4

```

For SM90/SM100 machines, we can enable `fi_allreduce_fusion` as follows:

```bash
vllm serve Qwen/Qwen3-Next-80B-A3B-Instruct-FP8 \
--tensor-parallel-size 4 \
--compilation_config.pass_config.enable_fi_allreduce_fusion true \
--compilation_config.pass_config.enable_noop true

```

### Advanced Configuration with MTP

`Qwen3-Next` also supports Multi-Token Prediction (MTP in short), you can launch the model server with the following arguments to enable MTP.

```bash
vllm serve Qwen/Qwen3-Next-80B-A3B-Instruct  \
--tokenizer-mode auto  --gpu-memory-utilization 0.8 \
--speculative-config '{"method": "qwen3_next_mtp", "num_speculative_tokens": 2}' \
--tensor-parallel-size 4 --no-enable-chunked-prefill 
```

The `speculative-config` argument configures speculative decoding settings using a JSON format. The method "qwen3_next_mtp" specifies that the system should use Qwen3-Next's specialized multi-token prediction method. The `"num_speculative_tokens": 2` setting means the model will speculate 2 tokens ahead during generation.


## Performance Metrics

### Benchmarking

We use the following script to demonstrate how to benchmark `Qwen/Qwen3-Next-80B-A3B-Instruct`.

```bash
vllm bench serve \
  --backend vllm \
  --model Qwen/Qwen3-Next-80B-A3B-Instruct \
  --served-model-name qwen3-next \
  --endpoint /v1/completions \
  --dataset-name random \
  --random-input 2048 \
  --random-output 1024 \
  --max-concurrency 10 \
  --num-prompt 100 
```

## Usage Tips

### Tune MoE kernel

When starting the model service, you may encounter the following warning in the server log(Suppose the GPU is `NVIDIA_H20-3e`):

```shell
(VllmWorker TP2 pid=47571) WARNING 09-09 15:47:25 [fused_moe.py:727] Using default MoE config. Performance might be sub-optimal! Config file not found at ['/vllm_path/vllm/model_executor/layers/fused_moe/configs/E=512,N=128,device_name=NVIDIA_H20-3e.json']
```

You can use [benchmark_moe](https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_moe.py) to perform MoE Triton kernel tuning for your hardware. Once tuning is complete, a JSON file with a name like `E=512,N=128,device_name=NVIDIA_H20-3e.json` will be generated. You can specify the directory containing this file for your deployment hardware using the environment variable `VLLM_TUNED_CONFIG_FOLDER`, like:

```shell
VLLM_TUNED_CONFIG_FOLDER=your_moe_tuned_dir vllm serve Qwen/Qwen3-Next-80B-A3B-Instruct \
  --tensor-parallel-size 4 \
  --served-model-name qwen3-next 

```

You should see the following information printed in the server log. This indicates that the tuned MoE configuration has been loaded, which will improve the model service performance.

```shell
(VllmWorker TP2 pid=60498) INFO 09-09 16:23:07 [fused_moe.py:720] Using configuration from /your_moe_tuned_dir/E=512,N=128,device_name=NVIDIA_H20-3e.json for MoE layer.
```

### Data Parallel Deployment

vLLM supports multi-parallel groups. You can refer to [Data Parallel Deployment documentation](https://docs.vllm.ai/en/latest/serving/data_parallel_deployment.html) and try parallel combinations that are more suitable for this model.

### Function calling

vLLM also supports calling user-defined functions. Make sure to run your Qwen3-Next models with the following arguments.

```bash
vllm serve ... --tool-call-parser hermes --enable-auto-tool-choice
```

### Known limitations

- Qwen3-Next currently does not support automatic prefix caching.

  
## AMD GPU Support
Recommended approaches by hardware type are:


MI300X/MI325X/MI355X 

Please follow the steps here to install and run Qwen3-Next models on AMD MI300X/MI325X/MI355X GPU.

### Step 1: Prepare Docker Environment
Pull the latest vllm docker:
```shell
docker pull vllm/vllm-openai-rocm:v0.14.1
```
Launch the ROCm vLLM docker: 
```shell
docker run -it --ipc=host --network=host --privileged --cap-add=CAP_SYS_ADMIN --device=/dev/kfd --device=/dev/dri --device=/dev/mem --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v $(pwd):/work -e SHELL=/bin/bash  --name Qwen3-Next  vllm/vllm-openai-rocm:v0.14.1
```
### Step 2: Log in to Hugging Face
Log in to your Hugging Face account:
```shell
hf auth login
```

### Step 3: Start the vLLM server

Run the vllm online serving


```shell
VLLM_ROCM_USE_AITER=1 \
SAFETENSORS_FAST_GPU=1 \
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
vllm serve Qwen/Qwen3-Next-80B-A3B-Instruct \
--tensor-parallel-size 4 \
--max-model-len 32768  \
--no-enable-prefix-caching \
--trust-remote-code 
```




### Step 4: Run Benchmark
Open a new terminal and run the following command to execute the benchmark script inside the container.
```shell
docker exec -it Qwen3-Next  vllm bench serve \
  --model "Qwen/Qwen3-Next-80B-A3B-Instruct" \
  --dataset-name random \
  --random-input-len 8192 \
  --random-output-len 1024 \
  --request-rate 10000 \
  --num-prompts 16 \
  --ignore-eos \
  --trust-remote-code 
```
