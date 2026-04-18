#!/bin/bash
# Helper: fetch HF model metadata as compact JSON for migration scripts.
# Usage: ./scripts/hf-info.sh moonshotai/Kimi-K2.5
#
# Extracts the key fields needed for recipe YAML:
#   - id, author, last_modified
#   - config.architectures, config.model_type
#   - safetensors.parameters (total + dtype breakdown)
#   - pipeline_tag, tags
#   - library_name

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <hf_org/repo>" >&2
  exit 1
fi

MODEL="$1"
HF_BIN="${HF_BIN:-~/.local/bin/hf}"

"$HF_BIN" models info "$MODEL" 2>/dev/null | python3 -c "
import json, sys
d = json.load(sys.stdin)
cfg = d.get('config', {}) or {}
st = d.get('safetensors', {}) or {}
params = st.get('parameters', {}) or {}
total = st.get('total', 0)
print(json.dumps({
    'id': d.get('id'),
    'author': d.get('author'),
    'last_modified': d.get('last_modified'),
    'architectures': cfg.get('architectures', []),
    'model_type': cfg.get('model_type'),
    'pipeline_tag': d.get('pipeline_tag'),
    'library_name': d.get('library_name'),
    'tags': d.get('tags', []),
    'total_parameters': total,
    'param_dtypes': params,
    'disabled': d.get('disabled'),
    'gated': d.get('gated'),
    'likes': d.get('likes'),
    'downloads': d.get('downloads'),
}, indent=2))
"
