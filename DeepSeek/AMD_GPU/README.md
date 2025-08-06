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
rocm/vllm-dev:nightly
```
#### Step 2
  Huggingface login
```shell
   huggingface-cli login 
```   
#### Step 3
##### FP8

Run the vllm online serving
Sample Command
```shell
SAFETENSORS_FAST_GPU=1 VLLM_ROCM_USE_AITER=1 VLLM_USE_V1=1 vllm serve deepseek-ai/DeepSeek-R1 -tp 8 --max-model-len 32768 --block-size 1 --max_seq_len_to_capture 32768 --no-enable-prefix-caching --max-num-batched-tokens 32768 --gpu-memory-utilization 0.95 --trust-remote-code
```
#### Step 4 
Open a new terminal, enter into the running docker and run the following benchmark script.
```shell
docker exec -it vllm_DS /bin/bash 
python3 /vllm-workspace/benchmarks/benchmark_serving.py --model deepseek-ai/DeepSeek-R1 --dataset-name random --ignore-eos --num-prompts 500  --max-concurrency 128 --random-input-len 3200 --random-output-len 800  --percentile-metrics ttft,tpot,itl,e2el
```
