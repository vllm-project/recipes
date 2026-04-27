---
name: serve
description: Use when the user asks to deploy / launch / serve / run a model with vLLM (e.g. "deploy Qwen3.5-0.8B", "serve Llama-3.1-8B on this box", "spin up vLLM for DeepSeek-V3.2"). Detects the local hardware, looks up the recipe in this repo's YAML database, synthesises the `vllm serve` command, runs it (Docker or pip), and verifies the OpenAI endpoint is live. Does NOT invent flags — only fields from the recipe + taxonomy + strategies.
---

# Serve a model with vLLM (deploy agent)

This skill turns a user request like *"deploy Qwen3.5-0.8B"* into a working vLLM server on the local box. It is the runtime sibling of `add-recipe`: that skill **writes** recipes, this one **executes** them.

## Hard rules

1. **Never invent flags.** Every `vllm serve` argument must come from the recipe's `base_args`, the chosen variant's `extra_args`, the chosen strategy's parallel flags, or the matching `hardware_overrides[<gen>]`. If a knob the user wants isn't in those, stop and tell them — do not guess.
2. **One recipe per launch.** If `models/<org>/<repo>.yaml` does not exist locally, do not synthesise from the model card. Tell the user "no recipe — run `/add-recipe <org>/<repo>` first" and stop.
3. **Hardware match must be explicit.** If the detected GPU is not in `taxonomy.yaml`, refuse to pick parallelism heuristically — surface the gap to the user and offer to add the profile.
4. **Refuse multi-tenant collisions.** Before launching, check that nothing else holds port 8000 and the named container slot. Don't kill an existing server unless the user says so.

## End-to-end flow

```
user: "deploy Qwen3.5-0.8B"
  │
  ▼
1. Resolve model_id ──────► models/<org>/<repo>.yaml (must exist)
2. Detect hardware  ──────► taxonomy.yaml profile id (must match)
3. Pick variant      ──────► default unless user names one
4. Pick strategy     ──────► single_node_tp on 1-GPU; honour user override only if in compatible_strategies
5. Synthesise command ────► assemble base_args + variant.extra_args + hardware_overrides + strategy parallel flag
6. Pick runtime      ──────► docker_image > pip wheel; on aarch64 always Docker (vllm/vllm-openai:<tag>-aarch64-cu130-ubuntu2404)
7. Launch            ──────► `docker run -d --name vllm-<slug> ...` and stream logs to /tmp/vllm-<slug>.log
8. Verify            ──────► poll http://localhost:8000/v1/models until 200 (cap 600s; abort on container exit)
9. Report            ──────► print the curl one-liner for the user to test, plus log path + container name
```

## 1. Resolve the model id

The user may give:

- a HF id `<org>/<repo>` → use as-is
- a bare name like `Qwen3.5-0.8B` → grep `models/*/*.yaml` for `slug:` or filename match; if more than one match, ask
- a URL `https://huggingface.co/<org>/<repo>` → strip the prefix

Then assert `models/<org>/<repo>.yaml` exists. If not, **stop**.

## 2. Detect hardware

Run `scripts/detect-hardware.sh`. It echoes one line:

```
profile=<taxonomy_id> gpu_count=<N> driver=<X.Y> cuda=<X.Y> arch=<x86_64|aarch64>
```

Mapping (single source of truth — keep in sync with `taxonomy.yaml`):

| `nvidia-smi` name contains | profile        | notes                           |
|----------------------------|----------------|---------------------------------|
| `GB10`                     | `dgx_spark`    | 1× Blackwell + Grace, ~120 GB unified |
| `H100`                     | `h100`         |                                 |
| `H200`                     | `h200`         |                                 |
| `B200`                     | `b200`         |                                 |
| `GB200`                    | `gb200`        |                                 |
| `B300`                     | `b300`         |                                 |
| `MI300X`/`MI325X`/`MI355X` | `mi300x` etc.  | brand=AMD                       |

If the detected name doesn't match, print the raw `nvidia-smi` line and stop.

## 3. Pick variant

- Default to `variants.default`.
- If the user said "fp8" / "nvfp4" / "int4", pick the matching variant key only if it exists in the recipe.
- If a variant has its own `model_id`, that supersedes the recipe's top-level `model_id` (e.g. `nvidia/Llama-3.1-8B-Instruct-NVFP4`).

VRAM check: compare `variant.vram_minimum_gb` against `taxonomy.hardware_profiles[<profile>].vram_gb`. If short, don't fail — print the deficit and ask whether to proceed (the user may know they're paging to host memory on Spark).

## 4. Pick strategy

- 1 GPU → `single_node_tp` with `--tensor-parallel-size 1`. Always.
- N GPUs single node → `single_node_tp` with `--tensor-parallel-size N` unless the recipe lists a TEP/DEP option **and** the user asked for it.
- Multi-node → out of scope for this skill (point to `multi_node_tp_pp` recipe guide).
- Refuse strategy if it's not in `compatible_strategies`.

## 5. Synthesise the command

Order of args (deterministic):

```
vllm serve <effective_model_id>
  --tensor-parallel-size <N>
  <recipe.model.base_args ...>
  <variant.extra_args ...>
  <hardware_overrides[<generation>].extra_args ...>
  <user-requested feature args, only if feature key exists in recipe.features>
```

Env: union of `recipe.model.base_env`, `variant.extra_env`, `hardware_overrides[<generation>].extra_env`.

Print the command before running. The user should be able to read it and recognise every flag.

## 6. Pick runtime

Decision table:

| Arch    | Recipe `docker_image` set? | Action                                                             |
|---------|----------------------------|--------------------------------------------------------------------|
| x86_64  | yes                        | `docker run` with that tag                                         |
| x86_64  | no                         | `uv pip install vllm` in `.venv` then `vllm serve` (or fallback to `vllm/vllm-openai:latest-x86_64-cu130-ubuntu2404`) |
| aarch64 | yes                        | `docker run` with that tag if it has an arm64 manifest, else fall to next row |
| aarch64 | no                         | `docker run vllm/vllm-openai:latest-aarch64-cu130-ubuntu2404`      |

Always mount `~/.cache/huggingface` so weights persist between launches.

If the recipe sets `nightly_required: true`, swap the pip install line for the nightly wheel index per the recipe schema.

## 7. Launch

Use `scripts/launch.sh`. It accepts the synthesised invocation as args and handles:

- `docker run -d --gpus all --ipc=host --name vllm-<slug> -p 8000:8000 -v ~/.cache/huggingface:/root/.cache/huggingface -e HF_TOKEN=${HF_TOKEN:-} <image> <vllm-serve-args>`
- Tee stdout/stderr to `/tmp/vllm-<slug>.log`
- If a container with the same name already exists: tell the user; do not auto-`rm`.

## 8. Verify

```bash
scripts/verify.sh <slug>          # polls /v1/models and watches docker logs for crashes; exits 0 on first 200
```

Cap at 600 s. On timeout: pull the last 50 lines of the log, summarise the error class (OOM, missing kernel, weight-download failure), and stop.

## 9. Report back

On success, print:

```
✓ vLLM is serving <model_id> on http://localhost:8000
  container : vllm-<slug>
  log file  : /tmp/vllm-<slug>.log
  test it   : curl http://localhost:8000/v1/models
              curl http://localhost:8000/v1/chat/completions \
                -H "Content-Type: application/json" \
                -d '{"model":"<model_id>","messages":[{"role":"user","content":"hi"}]}'
  stop it   : docker rm -f vllm-<slug>
```

## Worked example: Qwen3.5-0.8B on DGX Spark

```
Resolved      : Qwen/Qwen3.5-0.8B
Recipe        : models/Qwen/Qwen3.5-0.8B.yaml (slug: qwen3.5-0.8b)
Hardware      : profile=dgx_spark gpu_count=1 cuda=13.0 arch=aarch64
Variant       : default (bf16, 2 GB VRAM minimum) — fits in ~120 GB unified
Strategy      : single_node_tp, --tensor-parallel-size 1
Runtime       : docker (aarch64) → vllm/vllm-openai:latest-aarch64-cu130-ubuntu2404
Command:
  docker run -d --gpus all --ipc=host --name vllm-qwen3.5-0.8b \
    -p 8000:8000 -v ~/.cache/huggingface:/root/.cache/huggingface \
    -e HF_TOKEN=${HF_TOKEN:-} vllm/vllm-openai:latest-aarch64-cu130-ubuntu2404 \
    --model Qwen/Qwen3.5-0.8B --tensor-parallel-size 1 --trust-remote-code
```

## Failure modes — what to do

| Symptom                                           | Action                                                                 |
|---------------------------------------------------|------------------------------------------------------------------------|
| `models/<org>/<repo>.yaml` missing                | Stop. Suggest `/add-recipe <org>/<repo>`.                              |
| Detected GPU not in `taxonomy.yaml`               | Stop. Print the `nvidia-smi` line. Offer to add the profile.          |
| Variant not in recipe                             | List the variants present and ask which.                               |
| Container crashes during pull/load                | Print last 50 log lines; classify (network / OOM / unsupported kernel).|
| `/v1/models` never returns 200 within 600 s       | Same — and surface the model-loading progress lines from the log.      |
