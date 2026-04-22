# Contributing a new recipe

This repo hosts structured YAML recipes for serving open-weight models with
vLLM. Each recipe renders on [recipes.vllm.ai](https://recipes.vllm.ai/) as an
interactive command builder (pick hardware, variant, strategy → copy the exact
`vllm serve` command) and is exported as static JSON under `public/` for
agents/tools to consume.

One recipe = one YAML file at `models/<hf_org>/<hf_repo>.yaml`. The path
mirrors HuggingFace so the site URL `/<hf_org>/<hf_repo>` matches
`huggingface.co/<hf_org>/<hf_repo>` exactly.

## Quick start

1. Fork + clone the repo.
2. Create `models/<hf_org>/<hf_repo>.yaml` using the schema below.
3. Validate: `node scripts/build-recipes-api.mjs` — must print
   `✓ JSON API: N models, 8 strategies` with no errors.
4. Preview locally: `pnpm install && pnpm dev` → open
   `http://localhost:3000/<hf_org>/<hf_repo>`.
5. Open a PR.

## Fetching HuggingFace metadata

Before authoring, pull the model's HF config to get architecture, parameter
count, and context length correct:

```bash
bash scripts/hf-info.sh <hf_org>/<hf_repo>
```

This dumps `config.json` / `params.json`. Look at:

- **`architectures`** — names like `*MoE*`, `*Mixtral*`, presence of
  `num_experts` / `num_local_experts` / `moe.num_experts` → set
  `architecture: moe`. Otherwise `dense`.
- **`max_position_embeddings`** (or `text_config.max_position_embeddings` for
  VL models) → `context_length`.
- **`torch_dtype` / `quantization_config`** → `precision` on the default
  variant (bf16, fp16, fp8, nvfp4, int4, etc.).

## YAML schema (authoritative)

Top-level keys, in this order:

```yaml
meta:
  title: "DeepSeek-V3.2"                          # display name
  slug: "deepseek-v3.2"                           # kebab-case; keep consistent with title
  provider: "DeepSeek"                            # human-readable label
  description: "…"                                # one-sentence summary
  date_updated: 2026-04-20                        # today's date (or last touch)
  difficulty: intermediate                        # beginner | intermediate | advanced
  tasks:
    - text                                        # one or more of: text, multimodal, omni, embedding
  performance_headline: "…"                       # optional pithy line for cards
  related_recipes: ["deepseek-ai/DeepSeek-V3.1"]  # optional list of "<org>/<repo>" ids
  # Optional — only add GPUs you've ACTUALLY tested end-to-end.
  # Missing GPUs are assumed to work silently (we don't flag "untested").
  hardware:
    h200: verified
    mi355x: verified

model:
  model_id: "deepseek-ai/DeepSeek-V3.2"           # MUST match the filename path
  min_vllm_version: "0.18.0"                      # earliest vLLM release that supports this model
  docker_image: "vllm/vllm-openai:v0.18.0"        # optional — override the Docker image:tag shown in Install → Docker. String form pins NVIDIA only (AMD/TPU still use their brand defaults). For per-brand pins, use the object form:
                                                  #   docker_image:
                                                  #     nvidia: "vllm/vllm-openai:gemma4"
                                                  #     amd:    "vllm/vllm-openai-rocm:gemma4"
                                                  #     tpu:    "vllm/vllm-tpu:gemma4"
                                                  # Missing keys fall back to `:latest` for that brand (vllm/vllm-openai, vllm/vllm-openai-rocm, vllm/vllm-tpu).
  nightly_required: true                          # optional — set when `min_vllm_version` hasn't shipped yet. Swaps the default pip command to `uv pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly/cu130` and adds a "nightly" pill to the Install header. Users on CUDA 12.9 are pointed to the /cu129 index via the note.
  # Optional — control the Install block's pip/Docker tabs. Each key accepts
  # either `false` (hide the tab entirely) or an object with { command?, note? }.
  # Use `note` for a one-liner above the code block; use `command` to fully
  # replace the generated install script (e.g. nightly wheel, pinned version).
  #
  # Tab ORDER follows the YAML key order. Put `docker` before `pip` to make
  # Docker the first tab and the default when the block opens (useful when the
  # recipe is easier via Docker — e.g. a pinned image vs. a multi-step pip).
  install:
    docker:                                       # listed first → Docker is the default tab
      note: "Recommended: the pinned image ships all dependencies"
    pip:
      command: "uv pip install vllm --pre --index-url https://wheels.vllm.ai/nightly"
      note: "nightly wheel required — stable release does not yet support this model"
  architecture: moe                               # dense | moe
  parameter_count: "671B"                         # total params with B/T suffix
  active_parameters: "37B"                        # same as parameter_count for dense models
  context_length: 163840                          # integer (tokens)
  supports_dcp: true                              # optional — MLA-attention models (DeepSeek, Kimi-K2 family)
  base_args:                                      # flags always needed
    - "--trust-remote-code"
  base_env:                                       # env vars always needed
    VLLM_USE_FLASHINFER_MOE_FP8: "1"

# Optional — extra install steps beyond `uv pip install -U vllm`.
# Rendered as a code block above the vllm serve command.
dependencies:
  - note: "DeepGEMM required for FP8 MoE kernels"
    command: "uv pip install git+https://github.com/deepseek-ai/DeepGEMM.git@v2.1.1.post3 --no-build-isolation"
  - note: "Set VLLM_USE_DEEP_GEMM=0 to skip DeepGEMM (recommended on H20)"
    command: "export VLLM_USE_DEEP_GEMM=0"
    optional: true

features:
  tool_calling:                                   # names are conventions, not required keys
    description: "Enable …"
    args: ["--enable-auto-tool-choice", "--tool-call-parser", "<name>"]
  reasoning:
    description: "…"
    args: ["--reasoning-parser", "<name>"]
  spec_decoding:                                  # USE spec_decoding, NOT mtp — unified key for MTP / Eagle3 / ERNIE-MTP
    description: "…"
    args: ["--speculative-config", '{"method":"mtp","num_speculative_tokens":1}']

opt_in_features:                                  # features that default OFF (users tick them on)
  - spec_decoding                                 # spec decoding is opt-in unless docs insist

variants:
  default:                                        # ALWAYS include a `default` variant
    precision: fp8                                # bf16|fp8|nvfp4|fp4|int4|int8|awq|gptq|mxfp4
    vram_minimum_gb: 805                          # params × bytes × 1.2 (see formula below)
    description: "…"
  nvfp4:                                          # optional additional variants
    model_id: "nvidia/*-NVFP4"                    # only if the quantized checkpoint is a different HF repo
    precision: nvfp4
    vram_minimum_gb: 403
    extra_args: ["--kv-cache-dtype", "fp8"]
    extra_env: { VLLM_USE_FLASHINFER_MOE_FP4: "1" }

compatible_strategies:                            # subset of the 8 strategy ids in strategies/*.yaml
  - single_node_tp                                # always include as baseline
  - single_node_tep                               # for MoE
  - single_node_dep                               # for MoE
  - multi_node_tp
  - multi_node_tp_pp                              # TP within node + PP across — vLLM's recommended multi-node default
  - multi_node_dep                                # for MoE
  - multi_node_tep                                # for MoE
  - pd_cluster                                    # only if the recipe documents PD

# Optional per-generation tweaks.
# Key by `hopper` / `blackwell` / `amd` for per-family overrides, OR use
# `nvidia` as a brand-wide fallback that applies to every NVIDIA GPU when
# no generation-specific block is present.
hardware_overrides:
  blackwell:
    extra_args: ["--attention-backend", "FLASHINFER_MLA"]
    extra_env: {}
  amd:
    extra_args: []
    extra_env:
      VLLM_ROCM_USE_AITER: "1"

# Optional per-strategy tweaks. Rare. PD recipes use this to shape each pool.
#
# For `pd_cluster`, each role (prefill / decode) accepts:
#   nodes        — how many dedicated nodes the pool uses (default 1)
#   parallelism  — "tp" (tensor-parallel pool) | "dep" (data-parallel + expert-parallel)
#                  default "tp"
#   tp           — TP width per DP rank when parallelism=dep (default 1);
#                  unused for parallelism=tp (the whole pool is one TP group)
#   vllm_args    — extra flags appended to this role's `vllm serve`
#   env          — extra env vars for this role
#   parallel_flag — (rare) override the auto-selected flag for parallelism=tp
#
# The Nodes row in the UI exposes `nodes` as per-pool inputs; the values here
# set the defaults. Kimi-K2.5 on GB200 illustrates the DEP-PD pattern:
strategy_overrides:
  pd_cluster:
    prefill:
      nodes: 1
      parallelism: dep
      tp: 1                                     # DP = nodes × gpus_per_node / tp = 4
      vllm_args: ["--enforce-eager"]
      env: {}
    decode:
      nodes: 4
      parallelism: dep
      tp: 1                                     # DP = 16 on 4x GB200
      vllm_args: ["--compilation-config", '{"cudagraph_mode":"FULL_DECODE_ONLY"}']
      env: {}
  # Default TP-per-role PD (simpler pattern, one engine per pool):
  # pd_cluster:
  #   prefill: { nodes: 1, parallelism: tp }   # TP = gpus_per_node
  #   decode:  { nodes: 1, parallelism: tp }
  #
  # Cap TP below the node's gpu_count for small models that fit on fewer GPUs
  # (Gemma 4 runs on 1 GPU even on an 8-GPU node):
  # single_node_tp:
  #   tp: 1                                    # clamped to [1, gpu_count];
  #                                            # UI shows "using N of M GPUs"
  #                                            # under the Hardware pill when
  #                                            # effective TP < gpu_count.
  #                                            # Only read for single_node_tp —
  #                                            # TEP/DEP and multi-node ignore.

# Markdown guide. Rendered on the recipe page under the command builder.
# Covers Overview, Prerequisites, Launch, Benchmarking, Troubleshooting, References.
guide: |
  ## Overview
  ...

  ## Prerequisites
  - **Hardware**: 8x H200
  - **vLLM**: >= 0.18.0

  ## Launching the Server
  ```bash
  vllm serve deepseek-ai/DeepSeek-V3.2 --tensor-parallel-size 8 ...
  ```

  ## References
  - [Model card](https://huggingface.co/deepseek-ai/DeepSeek-V3.2)
```

## VRAM formula

`vram_minimum_gb = ceil(params × bytes_per_param × 1.2)` — uses **total**
params (MoE counts inactive experts; they still live in VRAM).

| Precision | Bytes per param |
|-----------|-----------------|
| bf16, fp16 | 2 |
| fp8, int8, awq, gptq | 1 |
| int4, nvfp4, fp4, mxfp4 | 0.5 |

Examples:
- 70B BF16 → `70 × 2 × 1.2 = 168 GB`
- 671B FP8 → `671 × 1 × 1.2 = 805 GB`
- 235B NVFP4 → `235 × 0.5 × 1.2 = 141 GB`

If a `model_id`-override variant points to a different base (e.g., a distilled
FP4 checkpoint), use **that** model's parameter count, not the base.

## Naming conventions

- Use **`spec_decoding`** for any speculative decoding feature (MTP / Eagle3 /
  ERNIE-MTP / qwen3_next_mtp / step3p5_mtp). Don't use `mtp` — the key was
  renamed across all recipes.
- Dense models typically only get `single_node_tp` + `multi_node_tp` +
  `multi_node_tp_pp`. MoE models can have all 8 strategies.
- Quantized variants reuse the precision name (`fp8`, `nvfp4`, `int4`). If the
  checkpoint is authored by someone else (e.g. `nvidia/*-NVFP4`), set
  `model_id:` inside the variant.
- Tasks: `omni` means the model is served via vLLM-Omni (offline Python, not
  `vllm serve`). The command builder hides the serve block for omni recipes
  and just surfaces the guide.

## Provider logo (new organizations only)

If `<hf_org>` isn't already in `src/lib/providers.js`, add an entry:

```js
"my-new-org": { display_name: "My New Org", logo: "/providers/my-new-org.png" },
```

The build script fetches the logo automatically from the HuggingFace avatar at
build time:

```bash
node scripts/fetch-provider-logos.mjs
```

## Hardware support levels

Only **`verified`** is a meaningful signal — GPUs you've actually tested with
this recipe end-to-end. The site renders a green ✓ on those pills and shows
"Verified on NVIDIA H200" above the command block when one is selected.

GPUs **not** in `meta.hardware` render silently with no badge — the common
case is "should work but nobody's written it down", and we don't flag that as
a warning. Just add the entry when you test. Don't fill the list with
speculative guesses.

## Testing your recipe

After saving the YAML, run:

```bash
# Fast validation: all YAMLs parse + JSON API builds
node scripts/build-recipes-api.mjs

# Full preview
pnpm install
pnpm dev
# Open http://localhost:3000/<hf_org>/<hf_repo>
```

Click through Hardware / Variant / Strategy / Nodes pills and confirm the
generated `vllm serve` command matches what you'd expect. The Verify / Bench
popovers should open without clipping.

## Commit style

Stage **only** the new recipe YAML (and `providers.js` if you added an org):

```bash
git add models/<hf_org>/<hf_repo>.yaml
# plus src/lib/providers.js if you added a new provider
git commit -m "Add <hf_org>/<hf_repo> recipe"
```

Do not stage:

- `public/` — generated at build time
- `node_modules/`
- `site/` — the legacy MkDocs output

## Questions

- Open an issue at https://github.com/vllm-project/recipes/issues
- Or start a discussion at https://github.com/vllm-project/recipes/discussions

If you use Claude Code, the repo ships a skill at
`.claude/skills/add-recipe/SKILL.md` that walks the same process
interactively.
