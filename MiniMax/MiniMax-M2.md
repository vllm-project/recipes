# MiniMax-M2.1/M2 Usage Guide

[MiniMax-M2.1](https://huggingface.co/MiniMaxAI/MiniMax-M2.1) and [MiniMax-M2](https://huggingface.co/MiniMaxAI/MiniMax-M2) are advanced large language models created by [MiniMax](https://www.minimax.io/). They offer the following highlights:

* Superior Intelligence – Ranks #1 among open-source models globally across mathematics, science, coding, and tool use.
* Advanced Coding – Excels at multi-file edits, coding-run-fix loops, and test-validated repairs. Strong performance on SWE-Bench and Terminal-Bench tasks.
* Agent Performance – Plans and executes complex toolchains across shell, browser, and code environments. Maintains traceable evidence and recovers gracefully from errors.
* Efficient Design – 10B activated parameters (230B total) enables lower latency, cost, and higher throughput for interactive and batched workloads.

## Supported Models

This guide applies to the following models. You only need to update the model name during deployment. The following examples use **MiniMax-M2**:

- [MiniMaxAI/MiniMax-M2.1](https://huggingface.co/MiniMaxAI/MiniMax-M2.1)
- [MiniMaxAI/MiniMax-M2](https://huggingface.co/MiniMaxAI/MiniMax-M2)

## Installing vLLM

- If you encounter corrupted output when using vLLM to serve these models, you can upgrade to the nightly version (ensure it is a version after commit [cf3eacfe58fa9e745c2854782ada884a9f992cf7](https://github.com/vllm-project/vllm/commit/cf3eacfe58fa9e745c2854782ada884a9f992cf7))

```bash
uv venv
source .venv/bin/activate
uv pip install -U vllm --extra-index-url https://wheels.vllm.ai/nightly
```

## Launching MiniMax-M2.1/M2 with vLLM

You can use 4x H200/H20 or 4x A100/A800 GPUs to launch this model.

run tensor-parallel like this:

```bash
vllm serve MiniMaxAI/MiniMax-M2 \
  --tensor-parallel-size 4 \
  --tool-call-parser minimax_m2 \
  --reasoning-parser minimax_m2_append_think  \
  --enable-auto-tool-choice
```

Note that TP8 is not supported. To run the model with >4 GPUs, please use DP+EP:

```bash
vllm serve MiniMaxAI/MiniMax-M2 \
  --data-parallel-size 8 \
  --enable-expert-parallel \
  --tool-call-parser minimax_m2 \
  --reasoning-parser minimax_m2_append_think  \
  --enable-auto-tool-choice
```

If you encounter `torch.AcceleratorError: CUDA error: an illegal memory access was encountered`, you can add `--compilation-config "{\"cudagraph_mode\": \"PIECEWISE\"}"` to the startup parameters to resolve this issue. 

```bash
vllm serve MiniMaxAI/MiniMax-M2 \
  --tensor-parallel-size 4 \
  --tool-call-parser minimax_m2 \
  --reasoning-parser minimax_m2_append_think  \
  --enable-auto-tool-choice \
  --compilation-config "{\"cudagraph_mode\": \"PIECEWISE\"}"
```

To run the model in responsesAPI that natively supports thinking, run it with the minimax_m2 reasoning parser:
```bash
vllm serve MiniMaxAI/MiniMax-M2 \
  --tensor-parallel-size 4 \
  --tool-call-parser minimax_m2 \
  --reasoning-parser minimax_m2 \
  --enable-auto-tool-choice
```

## Performance Metrics


### Benchmarking

We use the following script to demonstrate how to benchmark MiniMaxAI/MiniMax-M2`.

```bash
vllm bench serve \
  --backend vllm \
  --model MiniMaxAI/MiniMax-M2 \
  --endpoint /v1/completions \
  --dataset-name random \
  --random-input 2048 \
  --random-output 1024 \
  --max-concurrency 10 \
  --num-prompt 100 
```


If successful, you should see output similar to the following (TP 4 on NVIDIA_H20-3e *4) :

```
============ Serving Benchmark Result ============
Successful requests:                     100       
Failed requests:                         0         
Maximum request concurrency:             10        
Benchmark duration (s):                  851.51    
Total input tokens:                      204800    
Total generated tokens:                  98734     
Request throughput (req/s):              0.12      
Output token throughput (tok/s):         115.95    
Peak output token throughput (tok/s):    130.00    
Peak concurrent requests:                20.00     
Total Token throughput (tok/s):          356.46    
---------------Time to First Token----------------
Mean TTFT (ms):                          520.98    
Median TTFT (ms):                        523.86    
P99 TTFT (ms):                           1086.48   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          82.82     
Median TPOT (ms):                        82.90     
P99 TPOT (ms):                           84.28     
---------------Inter-token Latency----------------
Mean ITL (ms):                           82.78     
Median ITL (ms):                         82.18     
P99 ITL (ms):                            83.48 
```

## Using Tips

### DeepGEMM Usage

vLLM has DeepGEMM enabled by default, follow the [setup instructions](https://github.com/vllm-project/vllm/blob/v0.11.0/benchmarks/kernels/deepgemm/README.md#setup) to install it.

### AMD GPU Support

Please follow the steps here to install and run MiniMax-M2 models on AMD MI300X GPU.
### Step 1: Prepare Docker Environment
Pull the latest vllm docker:
```shell
docker pull rocm/vllm-dev:nightly
```
Launch the ROCm vLLM docker: 
```shell
docker run -it --ipc=host --network=host --privileged --cap-add=CAP_SYS_ADMIN --device=/dev/kfd --device=/dev/dri --device=/dev/mem --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v $(pwd):/work -e SHELL=/bin/bash  --name MiniMax-M2 rocm/vllm-dev:nightly 
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
VLLM_ALL2ALL_BACKEND="allgather_reducescatter" vllm serve MiniMaxAI/MiniMax-M2 \
  --data-parallel-size 8 \
  --max-model-len 32768 \
  --enable-expert-parallel \
  --tool-call-parser minimax_m2 \
  --reasoning-parser minimax_m2_append_think  \
  --enable-auto-tool-choice	\
  --disable-nccl-for-dp-synchronization \
  --trust-remote-code
```



### Step 4: Run Benchmark
Open a new terminal and run the following command to execute the benchmark script inside the container.
```shell
docker exec -it MiniMax-M2 vllm bench serve \
  --model "MiniMaxAI/MiniMax-M2" \
  --dataset-name random \
  --random-input-len 8000 \
  --random-output-len 1000 \
  --request-rate 10000 \
  --num-prompts 16 \
  --ignore-eos
```
