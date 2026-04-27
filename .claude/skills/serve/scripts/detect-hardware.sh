#!/usr/bin/env bash
# Echoes a single line consumed by SKILL.md step 2:
#   profile=<id> gpu_count=<N> driver=<X.Y> cuda=<X.Y> arch=<x86_64|aarch64>
# Exits 1 if no NVIDIA GPU is visible. Exits 2 if the GPU is not in the mapping.
set -uo pipefail   # not -e: head/awk on long pipes can SIGPIPE when we only need the first line

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "no-nvidia-smi" >&2
  exit 1
fi

names=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)
name=$(printf '%s\n' "$names" | sed -n '1p' | tr -d '\r' | xargs)
count=$(printf '%s\n' "$names" | grep -c .)
driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | sed -n '1p' | tr -d ' \r')
cuda=$(nvidia-smi 2>/dev/null | grep -oE 'CUDA Version: *[0-9]+\.[0-9]+' | head -1 | grep -oE '[0-9]+\.[0-9]+' || true)
arch=$(uname -m)

# Map nvidia-smi name → taxonomy profile id. Keep aligned with taxonomy.yaml.
profile=""
case "$name" in
  *GB10*)        profile="dgx_spark" ;;
  *H100*)        profile="h100" ;;
  *H200*)        profile="h200" ;;
  *GB200*)       profile="gb200" ;;
  *GB300*)       profile="gb300" ;;
  *B200*)        profile="b200" ;;
  *B300*)        profile="b300" ;;
  *MI300X*)      profile="mi300x" ;;
  *MI325X*)      profile="mi325x" ;;
  *MI355X*)      profile="mi355x" ;;
  *)             profile="" ;;
esac

if [[ -z "$profile" ]]; then
  echo "unknown-gpu name='$name'" >&2
  exit 2
fi

echo "profile=$profile gpu_count=$count driver=$driver cuda=$cuda arch=$arch name='$name'"
