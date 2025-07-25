# Qwen3-Coder Usage Guide

[Qwen3-Coder](https://github.com/QwenLM/Qwen3-Coder) is an advanced large language model created by the Qwen team from Alibaba Cloud. vLLM already supports Qwen3-Coder, and `tool-call` functionality will be available in vLLM v0.10.0 and higher You can install vLLM with `tool-call` support using the following method:

## Installing vLLM

```bash
uv venv
source .venv/bin/activate
uv pip install -U vllm --torch-backend auto
```

## Launching Qwen3-Coder with vLLM

### Serving on 8xH200 (or H20) GPUs (141GB × 8)

**BF16 Model**

```bash
vllm serve Qwen/Qwen3-Coder-480B-A35B-Instruct \
  --tensor-parallel-size 8 \
  --max-model-len 32000 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
```

**FP8 Model**

```bash
vllm serve Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 \
  --enable-expert-parallel \
  --data-parallel-size 8 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
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
ERROR [multiproc_executor.py:511]   File "/vllm/vllm/model_executor/models/qwen3_moe.py", line 336, in <lambda>
ERROR [multiproc_executor.py:511]     lambda prefix: Qwen3MoeDecoderLayer(config=config,
ERROR [multiproc_executor.py:511]                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ERROR [multiproc_executor.py:511]   File "/vllm/vllm/model_executor/models/qwen3_moe.py", line 278, in __init__
ERROR [multiproc_executor.py:511]     self.mlp = Qwen3MoeSparseMoeBlock(config=config,
ERROR [multiproc_executor.py:511]                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ERROR [multiproc_executor.py:511]   File "/vllm/vllm/model_executor/models/qwen3_moe.py", line 113, in __init__
ERROR [multiproc_executor.py:511]     self.experts = FusedMoE(num_experts=config.num_experts,
ERROR [multiproc_executor.py:511]                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ERROR [multiproc_executor.py:511]   File "/vllm/vllm/model_executor/layers/fused_moe/layer.py", line 773, in __init__
ERROR [multiproc_executor.py:511]     self.quant_method.create_weights(layer=self, **moe_quant_params)
ERROR [multiproc_executor.py:511]   File "/vllm/vllm/model_executor/layers/quantization/fp8.py", line 573, in create_weights
ERROR [multiproc_executor.py:511]     raise ValueError(
ERROR [multiproc_executor.py:511] ValueError: The output_size of gate's and up's weight = 320 is not divisible by weight quantization block_n = 128.
```

### Tool Calling
- **Enable Tool Calls**: Add `--tool-call-parser qwen3_coder` to enable tool call parsing functionality, please refer to: [tool_calling](https://docs.vllm.ai/en/latest/features/tool_calling.html)

## Roadmap

- [ ] Add benchmark results


## Additional Resources

- [EvalPlus](https://github.com/evalplus/evalplus)
- [Qwen3-Coder](https://github.com/QwenLM/Qwen3-Coder)
- [vLLM Documentation](https://docs.vllm.ai/)
