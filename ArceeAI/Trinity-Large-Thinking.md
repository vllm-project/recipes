# Trinity-Large-Thinking Usage Guide

[Trinity-Large-Thinking](https://huggingface.co/arcee-ai/Trinity-Large-Thinking) is Arcee AI's reasoning-focused Trinity Large checkpoint. It is a sparse Mixture-of-Experts model designed for long-horizon planning, tool use, and multi-step agent workflows.

This guide describes how to run Trinity-Large-Thinking with vLLM for reasoning and tool-calling workloads. It focuses on the parts of the deployment that are specific to Trinity:

- extracting `<think>...</think>` traces into the OpenAI-compatible `reasoning` field
- enabling automatic tool use with structured `tool_calls`
- preserving reasoning across multi-turn agent loops so the model retains its working context

## Supported Model

This guide applies to [arcee-ai/Trinity-Large-Thinking](https://huggingface.co/arcee-ai/Trinity-Large-Thinking).

We recommend **vLLM 0.11.1 or newer**. Trinity-Large-Thinking uses the `AfmoeForCausalLM` architecture, which is supported by current vLLM builds.

Trinity-Large-Thinking emits explicit reasoning traces inside `<think>...</think>` blocks. For multi-turn chat and agentic tool loops, those reasoning tokens are part of the model's effective working state and should be preserved across turns.

## Installing vLLM

```bash
uv venv
source .venv/bin/activate
uv pip install -U vllm openai --torch-backend auto
```

## Launching Trinity-Large-Thinking with vLLM

We recommend starting from the Trinity-specific flags below, then adding the parallelism settings that match your hardware.

```bash
vllm serve arcee-ai/Trinity-Large-Thinking \
  --dtype bfloat16 \
  --reasoning-parser deepseek_r1 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
```

If you have already downloaded the checkpoint locally, you can serve the local path directly:

```bash
vllm serve /path/to/Trinity-Large-Thinking \
  --dtype bfloat16 \
  --reasoning-parser deepseek_r1 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
```

### Why these flags matter

- `--reasoning-parser deepseek_r1` extracts Trinity's `<think>...</think>` block into `message.reasoning`.
- `--enable-auto-tool-choice` allows the model to decide when to call a tool.
- `--tool-call-parser qwen3_coder` converts Trinity's tool-call output into structured OpenAI-style `tool_calls`.
- `--dtype bfloat16` matches the recommended serving setup for this checkpoint.

### Deployment Notes

- Trinity-Large-Thinking is a very large sparse MoE checkpoint. We recommend multi-GPU parallelism for production deployments.
- If you do not need the full long-context configuration, set `--max-model-len` lower to reduce KV-cache pressure.
- Add your standard cluster flags as needed, such as `--tensor-parallel-size`, `--data-parallel-size`, or `--enable-expert-parallel`.

## Validation Request

The following request verifies that both reasoning extraction and tool calling are configured correctly:

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
model = client.models.list().data[0].id

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
            },
            "required": ["location"],
        },
    },
}]

response = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "user", "content": "What is the weather in Paris right now?"}
    ],
    tools=tools,
    tool_choice="auto",
)

msg = response.choices[0].message
reasoning = getattr(msg, "reasoning", None) or getattr(
    msg, "reasoning_content", None
)

print("reasoning:", reasoning)
print("content:", msg.content)
print("tool_calls:", msg.tool_calls)
```

If the deployment is configured correctly, you should see:

- non-empty `reasoning`
- either a final answer in `content` or a structured entry in `tool_calls`

## Preserving Reasoning in Multi-Turn Agent Loops

The most important Trinity-specific integration requirement is to pass the assistant's reasoning back into later turns.

When appending an assistant response to conversation history:

```python
msg = response.choices[0].message
reasoning = getattr(msg, "reasoning", None) or getattr(
    msg, "reasoning_content", None
)

assistant_msg = {
    "role": "assistant",
    "content": msg.content or "",
}

if reasoning:
    assistant_msg["reasoning"] = reasoning

if msg.tool_calls:
    assistant_msg["tool_calls"] = [
        {
            "id": tc.id,
            "type": "function",
            "function": {
                "name": tc.function.name,
                "arguments": tc.function.arguments,
            },
        }
        for tc in msg.tool_calls
    ]

messages.append(assistant_msg)
```

We recommend following these rules consistently:

- Pass reasoning back as `reasoning`, even if your client library exposes it as `reasoning_content`.
- Keep `content` as an empty string on tool-only turns instead of `null`.
- Append the assistant message before appending the tool result message.
- Use the chat endpoint (`/v1/chat/completions`) when you need structured reasoning output.

## Troubleshooting

### No reasoning appears in responses

- Make sure you started the server with `--reasoning-parser deepseek_r1`.
- Use `/v1/chat/completions`, not `/v1/completions`.

### Tool calls come back as plain text

- Make sure both `--enable-auto-tool-choice` and `--tool-call-parser qwen3_coder` are enabled.
- Verify that you are passing OpenAI-style tool definitions in the request.

### The model loses coherence after a few tool turns

- Check that you are preserving `reasoning` on assistant turns.
- Do not replace tool-only assistant `content` with `null`.

### Out-of-memory during startup or long conversations

- Lower `--max-model-len`.
- Increase model parallelism for your deployment.
- Use a local checkpoint path if you want to control exactly which files are loaded.
