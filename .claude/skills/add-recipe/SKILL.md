---
name: add-recipe
description: Use when the user asks to add, contribute, or create a new vLLM recipe in this repo (e.g. "add a recipe for Qwen/Qwen3-XYZ", "create a recipe for huggingface.co/org/model"). Walks through fetching HF metadata, authoring the YAML at models/<hf_org>/<hf_repo>.yaml, picking variants/strategies, validating, and committing.
---

# Add a new vLLM recipe

Recipes are YAML files at `models/<hf_org>/<hf_repo>.yaml`. The path mirrors HuggingFace (`huggingface.co/<hf_org>/<hf_repo>`), and the site/API are generated at build time from these files + `taxonomy.yaml` + `strategies/*.yaml`.

## End-to-end steps

1. **Confirm the HF id.** You need the exact `<org>/<repo>` string. If the user gave a URL, strip the `https://huggingface.co/` prefix.
2. **Fetch model metadata.** Run `bash scripts/hf-info.sh <org>/<repo>` to pull `config.json` / `params.json`. Extract:
   - `architecture`: `moe` if `num_experts`, `num_local_experts`, `moe.num_experts`, or a `*MoE*` architecture name is present. Otherwise `dense`.
   - `parameter_count`: total params (e.g. `"671B"`, `"70B"`). Use HF model card or the sum of shard sizes.
   - `active_parameters`: for MoE, the activated-per-token count (e.g. `"37B"` on DeepSeek-V3.2). For dense, equal to `parameter_count`.
   - `context_length`: `max_position_embeddings` from `config.json` (for VL models, from `text_config.max_position_embeddings`).
3. **Read the README — don't skip this.** Run `curl -sL "https://huggingface.co/<org>/<repo>/resolve/main/README.md"` and scan the install / serve / usage sections in full. Configs are not enough; model authors put load-bearing requirements in prose. Mine the README for:
   - **`min_vllm_version` / `nightly_required`** — phrases like "install vllm nightly", "requires nightly wheels", or an install snippet using `--extra-index-url https://wheels.vllm.ai/nightly` mean `min_vllm_version: "nightly"` + `nightly_required: true`. A specific tag like "vLLM >= 0.12.0" sets that version. Don't default to `0.11.0` when the README says otherwise.
   - **`dependencies:`** — any pip line beyond `vllm` itself: version pins (`mistral_common >= 1.11.1`, `transformers >= 5.4.0`), extras (`vllm[audio]`), source installs (`pip install git+...`), DeepGEMM pins, etc. Pin them even when the README says "auto-installed" — users on stale wheel caches need an explicit upgrade path. Each entry needs a one-line `note` saying *why*.
   - **Parser flags for `features:`** — `--tool-call-parser <name>`, `--reasoning-parser <name>`, `--enable-auto-tool-choice`. Use the exact parser name the README specifies.
   - **Companion / draft repos** — EAGLE / MTP / Eagle3 heads, NVFP4 quants, instruct vs base. Wire as `spec_decoding` feature (draft pointer in `--speculative-config`) or a sibling variant with `model_id:` override. Copy the recommended `--speculative-config` JSON verbatim from the README.
   - **Recommended serve flags** — `--tensor-parallel-size`, `--gpu_memory_utilization`, `--max_num_batched_tokens`, `--max_num_seqs` go into the guide's launch command and into variant `extra_args` when they're variant-specific.
   - **Hardware guidance / sampling defaults** — "recommended on 8xH200" lines inform variant `description` + `vram_minimum_gb`; recommended `temperature` / `top_p` / `reasoning_effort` go in the guide's Client Usage block.
4. **Create the YAML.** Write `models/<hf_org>/<hf_repo>.yaml` following the schema below. Only include sections the model needs; leave `features: {}`, `opt_in_features: []`, `hardware_overrides: {}`, `strategy_overrides: {}` empty if not applicable.
5. **Register the provider (if new).** If `<hf_org>` isn't already in `src/lib/providers.js`, add an entry with `display_name` and the logo path `/providers/<hf_org>.png` (or `.jpeg`). Logos get downloaded by `scripts/fetch-provider-logos.mjs` on the next build.
6. **Validate.** Run `node scripts/build-recipes-api.mjs`. It must print `✓ JSON API: N models, 7 strategies` with no errors.
7. **Commit.** Follow the user's earlier feedback (no kill-and-rebuild of dev server; syntax-check only).

## YAML schema (top-level fields, in order)

```yaml
meta:
  title: "..."                    # display name (e.g. "DeepSeek-V3.2")
  slug: "..."                     # lowercase-kebab (legacy, keep consistent with title)
  provider: "..."                 # human-readable org label (e.g. "DeepSeek")
  description: "..."              # one-sentence summary
  date_updated: YYYY-MM-DD        # today's date, or the date the recipe was authored
  difficulty: beginner|intermediate|advanced
  tasks:                          # one or more of: text, multimodal, omni, embedding
    - text
  performance_headline: "..."     # optional pithy line for cards
  related_recipes: []             # optional list of "<org>/<repo>" ids
  # Optional. Tri-state:
  #   `verified`    — you've run this recipe on this GPU end-to-end (green ✓).
  #   `unsupported` — not yet runnable here today (compat gap, missing kernel,
  #                   upstream blocker). Pill disabled in UI with "Not yet
  #                   supported" tooltip. May flip later — revisit on updates.
  #   absent        — silent default, assumed to work. Don't mark "untested".
  hardware:
    h200: verified
    mi355x: verified
    # mi300x: unsupported    # e.g. when a required kernel/feature is missing

model:
  model_id: "<hf_org>/<hf_repo>"  # MUST match the filename path
  min_vllm_version: "0.11.0"      # string, e.g. "0.12.0"
  # Optional — pin the Docker image shown in Install → Docker. Two forms:
  #   docker_image: "vllm/vllm-openai:glm51"      # string pins NVIDIA only;
  #                                               # AMD/TPU still use brand defaults
  #   docker_image:                               # object pins per-brand
  #     nvidia: "vllm/vllm-openai:gemma4"
  #     amd:    "vllm/vllm-openai-rocm:gemma4"
  #     tpu:    "vllm/vllm-tpu:gemma4"
  # Missing keys fall back to `:latest` for that brand (vllm/vllm-openai,
  # vllm/vllm-openai-rocm, vllm/vllm-tpu). Use the object form when CUDA /
  # ROCm / TPU ship different pinned tags for the same recipe.
  docker_image: ""
  # Optional — set true when `min_vllm_version` hasn't shipped as a stable
  # release yet. Swaps the default pip command to nightly wheels
  # (https://wheels.vllm.ai/nightly/cu130) and adds a yellow "nightly" pill
  # to the Install header. Manual install.pip overrides still win.
  nightly_required: false
  # Optional — control the Install block's pip/Docker tabs. Each key accepts
  # `false` (hide the tab entirely) OR an object `{ command?, note? }` to
  # override the generated one-liner and/or show a note above it.
  #   install.pip: false                 → no wheel available, Docker only
  #   install.docker: false              → no published image, pip only
  #   install.pip.command: "..."         → replace the pip command
  #   install.pip.note: "..."            → one-liner above the code block
  # Tab ORDER follows the YAML key order — put `docker` first to make it the
  # default tab when Docker is the recommended install path.
  install:
    pip:
      command: ""
      note: ""
    docker:
      note: ""
  architecture: dense|moe
  parameter_count: "30B"          # string with suffix (B or T)
  active_parameters: "30B"        # same as parameter_count for dense models
  context_length: 131072          # integer (tokens)
  base_args: []                   # flags always needed (trust-remote-code, etc.)
  base_env: {}                    # env vars always needed

# Optional — only if the recipe needs extra pip installs beyond `uv pip install -U vllm`.
# Rendered as an "extra install" block above the vllm serve command.
dependencies:
  - note: "Why you need it (one line)"
    command: 'uv pip install -U "vllm[audio]"'
    optional: false               # omit or false for required; true to mark optional

features:
  tool_calling:                   # flip any of these pills; recipe chooses naming
    description: "..."
    args: ["--enable-auto-tool-choice", "--tool-call-parser", "<name>"]
  reasoning:
    description: "..."
    args: ["--reasoning-parser", "<name>"]
  spec_decoding:                  # USE spec_decoding, NOT mtp — unified key for
    description: "..."            # MTP / Eagle3 / ERNIE-MTP / etc.
    args: ["--speculative-config", '{"method":"mtp","num_speculative_tokens":1}']

opt_in_features:                  # features that default OFF (users tick them on)
  - spec_decoding                 # spec decoding is opt-in unless the model docs insist

variants:
  default:                        # ALWAYS include a `default` variant
    precision: bf16|fp8|nvfp4|fp4|int4|int8|awq|gptq|mxfp4
    vram_minimum_gb: <integer>    # params × bytes × 1.2 (see formula below)
    description: "..."
  fp8:                            # optional extra variants
    model_id: "<optional override>"   # only if the quantized variant is a different HF repo
    precision: fp8
    vram_minimum_gb: <integer>
    description: "..."
    extra_args: []
    extra_env: {}

compatible_strategies:            # subset of the 7 in strategies/*.yaml
  - single_node_tp                # always include this as a baseline
  - single_node_tep               # for MoE
  - single_node_dep               # for MoE
  - multi_node_tp
  - multi_node_dep                # for MoE
  - multi_node_tep                # for MoE
  - pd_cluster                    # only if the recipe documents PD

hardware_overrides:               # optional per-generation flags
  hopper:    { extra_args: [], extra_env: {} }
  blackwell: { extra_args: [], extra_env: {} }
  amd:       { extra_args: [], extra_env: {} }

strategy_overrides:               # optional per-strategy tweaks
  single_node_tp:
    tp: 1                         # optional — default TP size for this strategy.
                                  # Lets a small model run below full-node TP
                                  # (e.g. Gemma 4 fits on 1 GPU → tp: 1). Omit
                                  # to default to the node's gpu_count. Clamped
                                  # to [1, gpu_count]. When effective TP <
                                  # gpu_count the UI shows a "using N of M GPUs"
                                  # hint under the Hardware pill. TEP/DEP and
                                  # multi-node ignore `tp:` (topology requires
                                  # full pool).
    extra_args: []
    extra_env: {}

guide: |                          # markdown, rendered as the Guide accordion
  ## Overview
  ...
  ## Prerequisites
  ...
  ## Launch command
  ...
  ## Benchmarking
  ...
  ## References
  - [Model card](https://huggingface.co/<hf_org>/<hf_repo>)
```

## VRAM formula

`vram_minimum_gb = ceil(params × bytes_per_param × 1.2)` where params is the **total** parameter count (MoE includes inactive experts — they still live in VRAM).

| Precision | Bytes/param |
|-----------|-------------|
| bf16, fp16 | 2 |
| fp8, int8, awq, gptq (8-bit) | 1 |
| int4, nvfp4, fp4, mxfp4 (4-bit) | 0.5 |

Example: a 70B BF16 model → `70 × 2 × 1.2 = 168 GB`. Round up.

If the variant is `model_id`-overridden and the override is a different base model with its own param count (e.g. a distilled FP4 checkpoint), use the override's parameter count — verify it via HF.

## Naming and conventions

- **Feature keys**: prefer `tool_calling`, `reasoning`, `spec_decoding`. Don't use `mtp` — it's been renamed across the repo.
- **Strategy list**: MoE recipes usually support every strategy; dense recipes are limited to `single_node_tp` and `multi_node_tp` (TEP/DEP require MoE).
- **Variants**: quantized variants reuse the base name (`fp8`, `nvfp4`, `int4`). If the quantized checkpoint is authored by someone else (e.g. `nvidia/*-NVFP4`), set `model_id:` inside the variant.
- **Tasks**: `omni` means served via vLLM-Omni (offline Python, no `vllm serve`). The command builder hides the serve block for omni recipes — just put the Python usage in `guide:`.

## Validation checklist

Before committing:

1. `node scripts/build-recipes-api.mjs` succeeds and the new recipe appears in the line count.
2. `node -e "const d = require('./public/<hf_org>/<hf_repo>.json'); console.log(d.model.parameter_count, d.variants.default.vram_minimum_gb)"` prints sensible values.
3. The YAML top-level key order matches the schema above — downstream tools don't care, but reviewers scan for it.

## Commit

Stage **only** the new recipe (and providers.js if edited):

```bash
git add models/<hf_org>/<hf_repo>.yaml src/lib/providers.js
git commit -m "Add <hf_org>/<hf_repo> recipe"
```

Do not stage `public/` (it's generated) or the design docs.
