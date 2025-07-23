# Qwen3-Coder Usage Guide

## Installing vLLM

[Qwen3-Coder](https://github.com/QwenLM/Qwen3-Coder) is an advanced large language model created by the Qwen team from Alibaba Cloud. vLLM already supports Qwen3-Coder, and `tool-call` functionality will be available in the next release version(0.10.0). You can install vLLM with `tool-call` support using the following method:

```bash
conda create -n myenv python=3.12 -y
conda activate myenv
export VLLM_COMMIT=4594fc3b281713bd3d7634405b4a1393af40d294 # Use full commit hash from the main branch
pip install https://wheels.vllm.ai/${VLLM_COMMIT}/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl
```

## Launching Qwen3-Coder with vLLM

### Serving on H20 GPUs (141GB × 8)

**BF16 Model**

```bash
vllm serve Qwen/Qwen3-Coder-480B-A35B-Instruct \
  --tensor-parallel-size 8 \
  --max-model-len 32000 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3-coder
```

**FP8 Model**

```bash
vllm serve Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 \
  --enable-expert-parallel \
  --data-parallel-size 8 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3-coder
```

## Performance Metrics

We launched `Qwen3-Coder-480B-A35B-Instruct-FP8` using vLLM and evaluated its performance using  [EvalPlus](https://github.com/evalplus/evalplus). The results are displayed below:

| Dataset | Test Type | Pass@1 Score |
|-----------|-----------|--------------|
| HumanEval | Base tests | 0.939 |
| HumanEval+ | Base + extra tests | 0.902 |
| MBPP | Base tests | 0.918 |
| MBPP+ | Base + extra tests | 0.794 |

## Using Tips

### BF16 Models
- **Context Length Limitation**: A single H20 node cannot serve the orgional context length(262144). You can reduce the `max-model-len` to work within memory constraints.

### FP8 Models
- **Tensor Parallelism Issue**: When using `tensor-parallel-size 8`, the following failures are expected. Switch to data-parallel mode using `--data-parallel-size`. 
- **Additional Resources**: Refer to the [Data Parallel Deployment documentation](https://docs.vllm.ai/en/latest/serving/data_parallel_deployment.html) for more parallelism groups.

```shell
(VllmWorker rank=5 pid=3914996) ERROR 07-22 20:16:11 [multiproc_executor.py:511]   File "/vllm/vllm/model_executor/models/qwen3_moe.py", line 336, in <lambda>
(VllmWorker rank=5 pid=3914996) ERROR 07-22 20:16:11 [multiproc_executor.py:511]     lambda prefix: Qwen3MoeDecoderLayer(config=config,
(VllmWorker rank=5 pid=3914996) ERROR 07-22 20:16:11 [multiproc_executor.py:511]                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(VllmWorker rank=5 pid=3914996) ERROR 07-22 20:16:11 [multiproc_executor.py:511]   File "/vllm/vllm/model_executor/models/qwen3_moe.py", line 278, in __init__
(VllmWorker rank=5 pid=3914996) ERROR 07-22 20:16:11 [multiproc_executor.py:511]     self.mlp = Qwen3MoeSparseMoeBlock(config=config,
(VllmWorker rank=5 pid=3914996) ERROR 07-22 20:16:11 [multiproc_executor.py:511]                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(VllmWorker rank=5 pid=3914996) ERROR 07-22 20:16:11 [multiproc_executor.py:511]   File "/vllm/vllm/model_executor/models/qwen3_moe.py", line 113, in __init__
(VllmWorker rank=5 pid=3914996) ERROR 07-22 20:16:11 [multiproc_executor.py:511]     self.experts = FusedMoE(num_experts=config.num_experts,
(VllmWorker rank=5 pid=3914996) ERROR 07-22 20:16:11 [multiproc_executor.py:511]                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(VllmWorker rank=5 pid=3914996) ERROR 07-22 20:16:11 [multiproc_executor.py:511]   File "/vllm/vllm/model_executor/layers/fused_moe/layer.py", line 773, in __init__
(VllmWorker rank=5 pid=3914996) ERROR 07-22 20:16:11 [multiproc_executor.py:511]     self.quant_method.create_weights(layer=self, **moe_quant_params)
(VllmWorker rank=5 pid=3914996) ERROR 07-22 20:16:11 [multiproc_executor.py:511]   File "/vllm/vllm/model_executor/layers/quantization/fp8.py", line 573, in create_weights
(VllmWorker rank=5 pid=3914996) ERROR 07-22 20:16:11 [multiproc_executor.py:511]     raise ValueError(
(VllmWorker rank=5 pid=3914996) ERROR 07-22 20:16:11 [multiproc_executor.py:511] ValueError: The output_size of gate's and up's weight = 320 is not divisible by weight quantization block_n = 128.
```

### Tool Calling
- **Enable Tool Calls**: Add `--tool-call-parser qwen3-coder` to enable tool call parsing functionality, please refer to: [tool_calling](https://docs.vllm.ai/en/latest/features/tool_calling.html)

## Roadmap

- [ ] Add benchmark results


## Additional Resources

- [EvalPlus](https://github.com/evalplus/evalplus)
- [Qwen3-Coder](https://github.com/QwenLM/Qwen3-Coder)
- [vLLM Documentation](https://docs.vllm.ai/)
