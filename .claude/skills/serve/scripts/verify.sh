#!/usr/bin/env bash
# Polls http://localhost:8000/v1/models until 200, capped at 600s. Aborts early
# if the container has exited.
#
# Usage: verify.sh <slug> [timeout_seconds]
set -euo pipefail

slug="${1:?slug required}"
timeout="${2:-600}"
name="vllm-${slug}"
deadline=$(( $(date +%s) + timeout ))

while (( $(date +%s) < deadline )); do
  state=$(docker inspect -f '{{.State.Status}}' "$name" 2>/dev/null || echo "missing")
  if [[ "$state" != "running" ]]; then
    echo "container state=$state — aborting" >&2
    docker logs --tail 50 "$name" 2>&1 || true
    exit 1
  fi
  if curl -fsS -m 2 http://localhost:8000/v1/models >/dev/null 2>&1; then
    echo "OK"
    curl -fsS http://localhost:8000/v1/models
    exit 0
  fi
  sleep 5
done

echo "timeout after ${timeout}s — last log lines:" >&2
docker logs --tail 50 "$name" 2>&1 || true
exit 124
