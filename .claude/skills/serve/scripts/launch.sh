#!/usr/bin/env bash
# Launches a vLLM server in Docker. The skill (SKILL.md) is responsible for
# synthesising the vllm-serve args from the recipe; this script just runs them.
#
# Usage:
#   launch.sh <slug> <image> -- <vllm-serve-args...>
#
# Example:
#   launch.sh qwen3.5-0.8b vllm/vllm-openai:latest-aarch64-cu130-ubuntu2404 -- \
#     --model Qwen/Qwen3.5-0.8B --tensor-parallel-size 1 --trust-remote-code
set -euo pipefail

slug="${1:?slug required}"; shift
image="${1:?image required}"; shift
[[ "${1:-}" == "--" ]] || { echo "expected -- before vllm args" >&2; exit 64; }
shift

name="vllm-${slug}"
log="/tmp/${name}.log"

if docker ps -a --format '{{.Names}}' | grep -qx "$name"; then
  echo "container $name already exists — refusing to clobber" >&2
  echo "  docker logs $name      (inspect)" >&2
  echo "  docker rm -f $name     (remove and re-run)" >&2
  exit 65
fi

if ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq '(^|:)8000$'; then
  echo "port 8000 already in use — free it before launching" >&2
  exit 66
fi

mkdir -p "$HOME/.cache/huggingface"

# `--ipc=host` matches the upstream vllm/vllm-openai docs (avoids the default
# 64 MB /dev/shm cap which breaks tensor sharing).
docker run -d \
  --gpus all \
  --ipc=host \
  --name "$name" \
  -p 8000:8000 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  "$image" "$@" >/dev/null

echo "container : $name"
echo "image     : $image"
echo "log       : $log"
echo "follow    : docker logs -f $name"

# Stream into the on-host log file so the skill can grep it cheaply.
nohup bash -c "docker logs -f '$name' &>'$log'" >/dev/null 2>&1 &
echo "pid       : $!"
