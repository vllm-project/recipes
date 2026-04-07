import json
from typing import Any

import aiohttp
import modal

# ---------------------------------------------------------------------------
# Container image
# ---------------------------------------------------------------------------
vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.9.1")
    .uv_pip_install("transformers==5.5.0")  # required for Gemma 4 as of vllm 0.9.1
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})  # faster model transfers
)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
MODEL_NAME = "google/gemma-4-26B-A4B-it"
MODEL_REVISION = "47b6801b24d15ff9bcd8c96dfaea0be9ed3a0301"

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

# ---------------------------------------------------------------------------
# Performance knob
# ---------------------------------------------------------------------------
# Set FAST_BOOT=True when iterating on config or if cold-starts are frequent.
# Set FAST_BOOT=False (default) for production — enables Torch compilation and
# CUDA graph capture for lower latency and higher throughput.
FAST_BOOT = False

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
app = modal.App("vllm-gemma4")

N_GPU = 1
MINUTES = 60  # seconds
VLLM_PORT = 8000


@app.function(
    image=vllm_image,
    gpu=f"H200:{N_GPU}",
    scaledown_window=15 * MINUTES,
    timeout=10 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
@modal.concurrent(max_inputs=100)
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
def serve():
    import subprocess

    cmd = [
        "vllm", "serve", MODEL_NAME,
        "--revision", MODEL_REVISION,
        "--served-model-name", MODEL_NAME,
        "--host", "0.0.0.0",
        "--port", str(VLLM_PORT),
        "--uvicorn-log-level=info",
        "--async-scheduling",
        "--enforce-eager" if FAST_BOOT else "--no-enforce-eager",
        "--tensor-parallel-size", str(N_GPU),
        "--limit-mm-per-prompt", json.dumps({"image": 0, "video": 0, "audio": 0}),
        "--enable-auto-tool-choice",
        "--reasoning-parser", "gemma4",
        "--tool-call-parser", "gemma4",
    ]

    print(*cmd)
    subprocess.run(cmd)


# ---------------------------------------------------------------------------
# Local test entrypoint  (`modal run gemma4_modal.py`)
# ---------------------------------------------------------------------------
@app.local_entrypoint()
async def test(test_timeout=10 * MINUTES, content=None, twice=True):
    url = await serve.get_web_url.aio()

    system_prompt = {
        "role": "system",
        "content": "You are a helpful assistant.",
    }
    if content is None:
        content = "Explain the singular value decomposition in two sentences."

    messages = [system_prompt, {"role": "user", "content": content}]

    async with aiohttp.ClientSession(base_url=url) as session:
        print(f"Running health check for server at {url}")
        async with session.get("/health", timeout=test_timeout - 1 * MINUTES) as resp:
            assert resp.status == 200, f"Health check failed: {resp.status}"
        print("Health check passed.")

        await _send_request(session, MODEL_NAME, messages)

        if twice:
            messages[1]["content"] = "What is the capital of France?"
            await _send_request(session, MODEL_NAME, messages)


async def _send_request(
    session: aiohttp.ClientSession, model: str, messages: list
) -> None:
    payload: dict[str, Any] = {
        "messages": messages,
        "model": model,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}

    print(f"\nSending: {messages[-1]['content']}")
    async with session.post("/v1/chat/completions", json=payload, headers=headers) as resp:
        resp.raise_for_status()
        async for raw in resp.content:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            line = line[len("data: "):]
            chunk = json.loads(line)
            delta = chunk["choices"][0]["delta"]
            content = (
                delta.get("content")
                or delta.get("reasoning")
                or delta.get("reasoning_content")
            )
            if content:
                print(content, end="", flush=True)
    print()
