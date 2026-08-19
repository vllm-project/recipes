import json
import os
from typing import Any

import aiohttp
import modal

# LFM2.5 is a first-class vLLM architecture as of vLLM 0.23.0 (Lfm2ForCausalLM /
# Lfm2MoeForCausalLM / Lfm2VlForConditionalGeneration) — no --trust-remote-code needed.
vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.23.0")  # pins the version the recipe is validated against
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})  # faster model transfers
)

# Override via env to deploy a different LFM2.5 model or GPU, e.g.:
#   MODEL=LiquidAI/LFM2.5-8B-A1B GPU=H100 modal run LiquidAI/lfm25-modal.py
# LFM2.5 models are small, so a cheap GPU is plenty for the dense / VL checkpoints; size up to
# an L4/A10G (24 GB) or larger for the 8B-A1B MoE (all ~8B of experts stay resident).
MODEL_NAME = os.environ.get("MODEL", "LiquidAI/LFM2.5-1.2B-Instruct")
GPU = os.environ.get("GPU", "L4")

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

FAST_BOOT = False

app = modal.App("example-lfm2-5-vllm-inference")

N_GPU = 1
MINUTES = 60  # seconds
VLLM_PORT = 8000


@app.function(
    image=vllm_image,
    gpu=f"{GPU}:{N_GPU}",
    scaledown_window=15 * MINUTES,  # how long should we stay up with no requests?
    timeout=10 * MINUTES,  # how long should we wait for container start?
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
@modal.concurrent(  # how many requests can one replica handle? tune carefully!
    max_inputs=100,
)
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
def serve():
    import subprocess

    cmd = [
        "vllm",
        "serve",
        MODEL_NAME,
        "--served-model-name",
        "llm",
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--uvicorn-log-level=info",
        # LFM2.5 tool calling: surface Pythonic <|tool_call_start|>...<|tool_call_end|> as tool_calls
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "lfm2",
    ]

    # enforce-eager disables both Torch compilation and CUDA graph capture; default keeps them on.
    cmd += ["--enforce-eager" if FAST_BOOT else "--no-enforce-eager"]

    # LFM2.5 models fit on a single GPU, so tensor parallelism gives no real speedup — keep
    # N_GPU=1 unless you specifically want to experiment with a multi-GPU node.
    cmd += ["--tensor-parallel-size", str(N_GPU)]

    print(*cmd)

    subprocess.Popen(" ".join(cmd), shell=True)


@app.local_entrypoint()
async def test(test_timeout=10 * MINUTES, content=None):
    url = await serve.get_web_url.aio()

    if content is None:
        content = "What is C. elegans? Answer in one sentence."
    messages = [{"role": "user", "content": content}]  # OpenAI chat format

    async with aiohttp.ClientSession(base_url=url) as session:
        print(f"Running health check for server at {url}")
        async with session.get("/health", timeout=test_timeout - 1 * MINUTES) as resp:
            up = resp.status == 200
        assert up, f"Failed health check for server at {url}"
        print(f"Successful health check for server at {url}")

        print(f"Sending a sample message to {url}", *messages, sep="\n\t")
        await _send_request(session, "llm", messages)


async def _send_request(session: aiohttp.ClientSession, model: str, messages: list) -> None:
    # `stream=True` tells an OpenAI-compatible backend to stream chunks
    payload: dict[str, Any] = {"messages": messages, "model": model, "stream": True}
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}

    async with session.post("/v1/chat/completions", json=payload, headers=headers) as resp:
        async for raw in resp.content:
            resp.raise_for_status()
            line = raw.decode().strip()
            if not line or line == "data: [DONE]":
                continue
            if line.startswith("data: "):  # SSE prefix
                line = line[len("data: ") :]

            chunk = json.loads(line)
            assert chunk["object"] == "chat.completion.chunk"  # or something went horribly wrong
            delta = chunk["choices"][0]["delta"]
            content = delta.get("content") or delta.get("reasoning_content")
            if content:
                print(content, end="")
    print()
