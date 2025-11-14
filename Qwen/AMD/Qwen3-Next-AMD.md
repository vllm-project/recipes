#### Step by Step Guide
Please follow the steps here to install and run Qwen3-Next-80B-A3B-Instruct models on AMD MI300X GPU.
#### Step 1
Pull the latest vllm docker:
```shell
docker pull rocm/vllm-dev:nightly
```
Launch the Rocm-vllm docker: 
```shell
docker run -d -it --ipc=host --network=host --privileged --cap-add=CAP_SYS_ADMIN --device=/dev/kfd --device=/dev/dri --device=/dev/mem --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v /:/work -e SHELL=/bin/bash  --name Qwen3-next rocm/vllm-dev:nightly
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
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm serve Qwen/Qwen3-Next-80B-A3B-Instruct --tensor-parallel-size 4 --max-model-len 32768  --no-enable-prefix-caching 
```
#### Step 4 
Open a new terminal, enter into the running docker and run the following benchmark script.
```shell
docker exec -it Qwen3-next /bin/bash 
python3 /vllm-workspace/benchmarks/benchmark_serving.py --model Qwen/Qwen3-Next-80B-A3B-Instruct --dataset-name random --ignore-eos --num-prompts 500  --max-concurrency 128 --random-input-len 3200 --random-output-len 800  --percentile-metrics ttft,tpot,itl,e2el
```
