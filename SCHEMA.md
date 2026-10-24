# Recipe YAML schema

The authoritative reference for a recipe YAML, `models/<hf_org>/<hf_repo>.yaml`,
whichever repository the file lives in (`recipes/` in vllm-project/vllm,
vllm-omni, or this repo). It is versioned with the renderer that reads it:
change it in the same pull request as any code that adds or changes a field.
[CONTRIBUTING.md](CONTRIBUTING.md), [CLAUDE.md](CLAUDE.md) and the
[add-recipe skill](.agents/skills/add-recipe/SKILL.md) link here rather than
repeating it.

Two tools enforce and assist it:

- **Machine-readable schema** — [`/recipe.schema.json`](https://recipes.vllm.ai/recipe.schema.json),
  generated from `taxonomy.yaml` and `strategies/` by `src/lib/recipe-schema.js`. Every build checks
  every recipe against it: unknown or misspelled keys, wrong types, and ids outside the vocabulary are
  errors (the renderer would otherwise ignore them silently). Point your editor at it for completion
  and inline errors — with the YAML language server (VS Code's Red Hat YAML extension), start the file
  with `# yaml-language-server: $schema=https://recipes.vllm.ai/recipe.schema.json`.
- **Starter template** — [`templates/recipe.yaml`](templates/recipe.yaml): the required fields
  plus the common ones (text or multimodal, dense or MoE), with `TODO`s to fill in.

- [Annotated example](#annotated-example) — every field in order, with inline notes.
- [Field reference](#field-reference) — semantics, precedence, and rendering consequences.
- [VRAM formula](#vram-formula)

## Annotated example

Top-level keys, in this order:

```yaml
meta:
  title: "DeepSeek-V3.2"                          # display name
  slug: "deepseek-v3.2"                           # kebab-case; keep consistent with title
  provider: "DeepSeek"                            # human-readable label
  description: "…"                                # one-sentence summary
  date_added: 2026-04-20                          # initial catalog addition; never change
  date_updated: 2026-04-20                        # latest material recipe change
  tasks:
    - text                                        # one or more of: text, multimodal, omni, embedding
  performance_headline: "…"                       # optional pithy line for cards
  related_recipes: ["deepseek-ai/DeepSeek-V3.1"]  # optional list of "<org>/<repo>" ids
  # Optional — `verified` only for GPUs you've ACTUALLY tested end-to-end;
  # `supported` lists a restricted profile without that claim; `unsupported`
  # marks a confirmed blocker. Missing GPUs are assumed to work silently.
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
    VLLM_USE_DEEP_GEMM: "0"                       # MoE kernel selection itself is done via --moe-backend args
  default_frontend: rust                          # optional — python (default) | rust; preselects the Frontend pill
  rust_frontend: unsupported                      # optional — only when Rust is confirmed to fail (e.g. a parser it hasn't registered); explain why in the guide

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
  long_context:                                   # RoPE/YaRN context extension — always opt-in (see below)
    description: "…"
    args: ["--hf-overrides", '{"rope_parameters":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}}',
           "--max-model-len", "131072"]
    env:                                          # optional — env set only while the feature is on
      VLLM_ALLOW_LONG_MAX_MODEL_LEN: "1"          # needed when the window exceeds max_position_embeddings

opt_in_features:                                  # features that default OFF (users tick them on)
  - spec_decoding                                 # spec decoding is opt-in unless docs insist
  - long_context                                  # always opt-in: static YaRN costs short-prompt quality

variants:
  default:                                        # ALWAYS include a `default` variant
    precision: fp8                                # bf16|fp8|nvfp4|fp4|int4|int8|awq|gptq|mxfp4
    vram_minimum_gb: 805                          # params × bytes × 1.2 (see formula below)
    description: "…"
  nvfp4:                                          # optional additional variants
    model_id: "nvidia/*-NVFP4"                    # only if the quantized checkpoint is a different HF repo
    precision: nvfp4
    vram_minimum_gb: 403
    extra_args: ["--kv-cache-dtype", "fp8", "--moe-backend", "flashinfer_cutlass"]

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
    extra_args: []
    extra_env: {}
  amd:
    extra_args: []
    extra_env:
      VLLM_ROCM_USE_AITER: "1"

# A variant's `hardware_overrides.<exact_gpu_id>` block may also declare that an
# install path does not exist on that GPU. Use it only when the pinned image
# carries kernels that are not in vLLM main, so the flags this hardware
# generates would abort on an official wheel — not merely because Docker is the
# easier route. The pip tab disappears on that GPU alone; every other hardware
# keeps it.
#
#   variants:
#     default:
#       hardware_overrides:
#         dgx_spark_gb10:
#           docker_image: "eugr/spark-vllm-b12x:latest"
#           install:
#             pip: false

# Optional per-strategy tweaks. Rare. PD recipes use this to shape each pool.
#
# For `pd_cluster`, each role (prefill / decode) accepts:
#   nodes        — how many dedicated nodes the pool uses (default 1)
#   parallelism  — "tp" (tensor-parallel pool) | "dep" (data-parallel + expert-parallel)
#                  default "tp"; `nodes` and `parallelism` also take a
#                  hardware-keyed map (e.g. { default: tp, gb300: dep })
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

# Optional — tuned presets: named configurations qualified on one GPU (and
# optionally one strategy and some variants), e.g. a benchmark lane. Each is a
# delta on the baseline command; the builder's Preset row offers it next to
# Baseline, and its points (if any) in a Point row below.
presets:
  gb200_agentx_pd:
    label: "AgentX 1P1D"
    hardware: gb200                               # exact GPU id (required)
    strategy: pd_cluster                          # optional; absent = every strategy
    variants: [default]                           # optional; absent = every variant
    workload: "Agentic (InferenceX AgentX)"       # optional free-text tag
    default: true                                 # applied unless the user picks Baseline; omit = opt-in
    description: "…"
    source: { label: "InferenceX #4322", href: "https://github.com/SemiAnalysisAI/InferenceX/pull/4322", key: "<lane config key>", revision: b7eee97 }
    image:                                        # or a plain string (the vLLM image)
      vllm: "vllm/vllm-openai:nightly-e68efc68…"  # pins this GPU's docker command
      dynamo: "ffbbaaa6…"                         # shown with the preset; also pip: [...], model_revision
    select:                                       # builder choices the preset implies
      pd_router: dynamo                           # vllm | dynamo | smg (pd_cluster only)
      frontend: rust
      features: { tool_calling: false, spec_decoding: false }   # true/false, or a mode key
      dcp: 16
      kv_offload: simple                          # a taxonomy kv_offload id or kv_store id; off/false
    args: ["--decode-context-parallel-size", "16"]   # every node / both PD roles
    env: { NCCL_NET_PLUGIN: "none" }
    remove_args: ["--block-size"]                 # drop an inherited flag (and its value)
    json_args:                                    # edit keys inside a JSON-valued flag; null deletes
      --speculative-config: { num_speculative_tokens: 4 }
    roles:                                        # pd_cluster only
      decode: { nodes: 4, parallelism: tp, args: ["--max-num-batched-tokens", "1024"] }
    points:                                       # optional named operating points
      c16:
        label: "C16"
        concurrency: 16                           # informational
        default: true                             # else the first point
        roles: { decode: { args: ["--max-num-seqs", "32"] } }
      c32:
        label: "C32"
        roles: { decode: { args: ["--max-num-seqs", "64"] } }
    runbook: |                                    # prose only: evidence, reproduction, limits
      Services, frontend, per-concurrency tuning the builder can't express…

# Optional — configuration-scoped notes and known issues. Rendered below the
# builder and filtered by its selection: `when` keys AND together, values OR
# together (hardware: exact id / generation / brand / arch:<isa>; strategies;
# variants; features = while one of them is on).
notes:                                            # under the command; `when` required
  - title: "MNNVL environments"
    when: { hardware: [gb200, gb300] }
    body: |
      Add `NCCL_MNNVL_ENABLE=1`, `NCCL_CUMEM_ENABLE=1` and `NCCL_NVLS_ENABLE=1`.
troubleshooting:                                  # Troubleshooting tab; matching entries first
  - title: "DeepGEMM warmup too slow"
    when: { hardware: [nvidia] }                  # optional; absent = always relevant
    body: |
      Set `VLLM_USE_DEEP_GEMM=0` to skip the long warmup.

# Markdown guide. Rendered on the recipe page under the command builder.
# Covers Overview, Prerequisites, Client usage, References — not launch
# commands (the builder renders those) and not configuration-specific notes
# or known issues (use `notes` / `troubleshooting`).
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

### Multiple speculative-decoding methods

Keep the flat `features.spec_decoding.args` form when a recipe has one method.
When the same served checkpoint can use multiple mutually exclusive methods,
replace `args` with a `modes` map:

```yaml
features:
  spec_decoding:
    description: "Speculative decoding — pick a drafting method."
    default_mode: mtp
    modes:
      mtp:
        label: "MTP"
        description: "Native multi-token prediction head."
        args: ["--speculative-config", '{"method":"mtp","num_speculative_tokens":2}']
      eagle3:
        label: "EAGLE3"
        description: "Community-trained external draft model."
        args: ["--speculative-config", '{"method":"eagle3","model":"community/model-eagle3","num_speculative_tokens":5}']

opt_in_features: [spec_decoding]
```

`default_mode` is the recommended selection when the feature is enabled. A
mode can use `variants: [default, fp8]` to limit it to particular served
checkpoints and `hardware: { hopper: unsupported }` to disable it on a GPU
family. Per-mode `hardware_overrides.<generation>.args` can replace its config
for one hardware generation.

The served checkpoint and speculative method are separate axes: only a
variant's `model_id` changes `vllm serve <checkpoint>`; a mode contributes only
the speculative arguments. If a draft is baked into a separately served
checkpoint, model it as a variant and couple its recommended method with
`default_modes`:

```yaml
variants:
  fused_eagle3:
    model_id: "community/model-with-eagle3"
    label: "EAGLE3"
    precision: fp8
    vram_minimum_gb: 128
    default_modes:
      spec_decoding: eagle3
```

Declaring `default_modes` also enables that feature by default for the variant,
even when it appears in `opt_in_features`. The build validator rejects mixed
`args` + `modes`, unknown mode/variant references, and variants with no usable
mode.

### Long-context extension (`long_context`)

Add this feature only when the **model card documents a specific extension
config** — never invent a YaRN factor. Keep it in `opt_in_features`: every card
that ships one says the same thing, that static YaRN applies a constant scaling
factor and therefore costs quality on short prompts, and the wider window also
inflates the KV budget.

Two things to get right when translating a model card:

1. **Use `--hf-overrides`, not `--rope-scaling`.** Most cards still print
   `--rope-scaling '{...}'`, but that flag was removed from vLLM's `ModelConfig`
   by 0.26 — a recipe emitting it breaks on any current wheel, even when the
   recipe's `min_vllm_version` floor is old enough that the flag once existed.
2. **Use the `rope_parameters` key.** Transformers v5 renamed the config field
   `rope_scaling` → `rope_parameters`, and vLLM 0.29 reads
   `hf_config.rope_parameters`. On a VL/omni checkpoint the block nests under
   `text_config` — a top-level override silently misses the decoder's config.

So a card's `--rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}'`
becomes:

```yaml
  long_context:
    description: "YaRN RoPE scaling to the card's validated 131,072-token context. …"
    args:
      - "--hf-overrides"
      - '{"rope_parameters":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}}'
      - "--max-model-len"
      - "131072"
```

Add `env: { VLLM_ALLOW_LONG_MAX_MODEL_LEN: "1" }` whenever the target window
exceeds the checkpoint's `max_position_embeddings` (cards show it in the launch
line when it's needed). Features emit last and arg dedupe is last-wins, so both
flags cleanly shadow a `--max-model-len` already present in `base_args` — don't
edit `base_args` to make room.

Some cards describe an extension path that is **not** a flag toggle: a local
`config.json` swap, a removed engine (`VLLM_USE_V1=0`), or instructions given
only for another serving framework. Leave those out of the feature and describe
them in the `guide` instead.

Unreleased, source-only support: put the complete clone/checkout/build
sequence in `model.install.pip.command` and set `model.install.docker: false`.

## Field reference

Top-level keys, in this order:

- `meta` — title, slug, provider, description, date_added, date_updated, difficulty (deprecated — nothing reads it; omit it in new recipes), tasks[], performance_headline, related_recipes[], optional `default_hardware`, optional `default_variant`, optional `hardware: { gpu_id: verified | unsupported }`. `default_variant` is a map in the variant-`tp` keyspace (exact GPU id > generation > brand > `default`), e.g. `{ amd: mxfp4 }`: when the active variant can't run on the selected hardware, `pickFittingVariant` switches to this variant if it runs there, instead of the largest-footprint fit. It steers the command builder's hardware switch; the JSON API renders every variant separately and is unaffected. `date_added` is the immutable day the recipe first entered the catalog and drives the homepage's "Latest recipes" ranking; `date_updated` records the latest material recipe change. Hardware states: `verified` = tested end-to-end (green ✓ badge); `unsupported` = not yet runnable on this hardware today — compat gap / missing kernel / upstream blocker that could lift later (pill rendered but disabled + struck through, tooltip says "Not yet supported"); `supported` = declared without a verified claim — only needed for a `restricted` profile (DGX Spark/Station, RTX Pro, TPU, Xeon), which appears in the picker only when listed here; absent = silent default, assumed to work. Any other value is a schema error. Don't use `unsupported` for "untested" — only when you've confirmed it currently fails. `default_hardware` is a single GPU profile id (`b300`, `dgx_spark_gb10`, …) — the hardware the page and the JSON API open on. **Precedence: URL `?hardware=` > the user's stored global preference (set only by an explicit pill click) > `default_hardware` > the `pickDefaultHardware` heuristic (B200 for Blackwell-constrained variants, else H200).** A returning visitor's own GPUs therefore still win — this key only replaces the model-blind heuristic. It's validated and fail-open: an id that's unknown to the taxonomy, precision-incompatible with the resolved variant, marked `unsupported`, or a `restricted` profile the recipe doesn't declare in `meta.hardware` falls through silently to the heuristic. **Authoring rule: set it to the hardware the guide's numbers, the `performance_headline` and the `verified` badge were taken on — not to the biggest GPU.** Landing on a `verified` profile is what makes the command block's green "Verified on <hw>" badge the built-in explanation for the choice, so no extra banner is needed; pointing it at an unverified GPU loses that badge and is worse than leaving the key absent. Most important for the JSON API, which has no URL or localStorage — `default_hardware` is the only way an author steers `recommended_command` (everything else stays addressable under `by_hardware`).
- `model` — `model_id`, `min_vllm_version`, `architecture` (`dense` or `moe`), `parameter_count`, `active_parameters`, `context_length`, `base_args[]`, `base_env{}`, optional `default_frontend: python | rust` (defaults to Python), optional `rust_frontend: unsupported`. Frontends are defined once in `taxonomy.yaml → frontends` (`{ label, env? }`, pill order = key order); synthesis merges the selected frontend's `env` into every rendering — Rust's is `VLLM_USE_RUST_FRONTEND: "1"`, Python sets nothing. Keep that variable in the taxonomy entry, never in `base_env`; use `default_frontend` to express recipe defaults. Set `default_frontend: rust` for recipes that recommend it. The selector is saved per recipe and in `?frontend=python|rust`; the JSON API uses the same recipe default. `rust_frontend: unsupported` (the general form is `<frontend_id>_frontend`; Python is the baseline and can't be opted out) disables the Rust pill with a tooltip, and `resolveFrontend` falls back to Python for every path — URL, saved state, JSON API — so no rendering ever carries the Rust env for that recipe. Same rule as `meta.hardware: unsupported`: only when Rust is confirmed to fail today (typically a `--tool-call-parser`/`--reasoning-parser` the Rust frontend hasn't registered, which aborts startup), not for "untested"; put the reason in the guide, and remove the key once a vLLM release fixes it. The build rejects unknown frontend ids, any value other than `unsupported`, and `default_frontend` naming an unsupported frontend.
- `dependencies` (optional) — list of `{ command, note?, optional?, brand?, install_modes? }`. Rendered as the "extra install" block above `vllm serve`. Use this for DeepGEMM pins, `vllm[audio]`, pinned transformers commits, vllm-omni source installs, etc. `optional: true` dims the entry and excludes it from the block's "Copy all" button (the user must copy it explicitly). `brand: NVIDIA | AMD | Intel` (or array) hides the entry when the current hardware pill is a different brand — use for CUDA-only kernels (xformers, DeepGEMM) or ROCm-only wheels. Omit `brand` for platform-agnostic deps (diffusers from source, transformers commits, etc.). `install_modes` gates the entry by Install tab, and **absent means `[pip]`** — the block is pip-scoped by default because the Docker image already bundles the pip installs, so repeating them beside `docker run` is noise. `install_modes: [pip, docker]` opts an entry into both tabs; reserve it for **host-side prerequisites Docker cannot satisfy**, i.e. a gated checkpoint `hf download` must place on disk before `buildDockerRun` can bind-mount it (MiniMax-H3). A pip install never wants this. Keep the runnable fetch step here rather than as a `#` comment on the command — the block's "Copy all" yields executable lines, a comment does not.
- `features` — map of `{feature_key: { description, args[], env?, label? }}`. Naming convention: `tool_calling`, `reasoning`, `spec_decoding` (the unified key for MTP / Eagle3 / ERNIE-MTP — **don't use `mtp`**, it was renamed across all recipes), `long_context` (RoPE/YaRN context extension — see below). Each feature is a boolean toggle by default. Optional `label` overrides the pill text (default: title-cased key).
  - **`env{}`** — env vars set only while the feature is on. Emitted from `buildEnv` as the mirror of the feature `args` block in `buildArgs` (step 7): applied after every override layer so an enabled feature wins, and gated identically (strategy allowlist, Mooncake-companion skip, active mode) so a feature's env and args can never disagree about whether it's on. A `modes` entry may carry `env` too, and `hardware_overrides.<key>.env` REPLACES it the same way `args` does. Use it for features whose documented launch line is env + flags rather than flags alone — `long_context` needs `VLLM_ALLOW_LONG_MAX_MODEL_LEN: "1"` wherever the target window exceeds the checkpoint's `max_position_embeddings`. Propagates to every rendering: single-node, head + each worker on multi-node (one env map per deployment), both PD roles, and the Docker `-e` flags.
  - **`long_context`** — the context-extension toggle, ALWAYS in `opt_in_features`: static YaRN holds its scaling factor constant regardless of input length, so leaving it on costs short-prompt quality, and the longer window inflates the KV budget. Args are `--hf-overrides '<rope json>'` + `--max-model-len <extended>`; because features emit last and `dedupeArgs` is last-wins, both cleanly shadow a `--max-model-len` already in `base_args`. **Use `--hf-overrides`, never `--rope-scaling`** — the latter is what most model cards still print, but it was removed from vLLM's `ModelConfig` by 0.26, and transformers v5 renamed the config field `rope_scaling` → `rope_parameters` (vLLM 0.29 reads `hf_config.rope_parameters`). So a card's `--rope-scaling '{...}'` translates to `--hf-overrides '{"rope_parameters":{...}}'`, and on a VL/omni checkpoint the block nests under `text_config` (a top-level override silently misses the decoder's config). Only add this feature where the model card documents a specific extension config — don't invent a factor.
  - **Single-select sub-modes** — when one feature has several *mutually exclusive* methods (e.g. a model offering more than one speculative method: MTP / DSpark / DFlash), replace the feature's flat `args[]` with a `modes` map and it renders as a "pick one" segmented row under the feature toggle instead of a boolean. **A mode contributes only `args` (the `--speculative-config`) — the method, never the served checkpoint.** Shape: `{ description, default_mode?, modes: { <mode_key>: { label?, description?, args[], variants?, hardware_overrides?, hardware? } } }`. Only the selected mode's `args` are emitted, so exactly one `--speculative-config` is ever produced — no dedupe conflict between sibling methods, and no `disables_features` gymnastics. `default_mode` picks the initial mode (else the first key). A mode may restrict itself to specific checkpoints with `variants: [<variant_key>, …]` — e.g. the dspark method only exists on the fused DSpark checkpoint, so FP8/NVFP4 offer MTP only (a mode with no `variants` is available everywhere). **The row renders only when ≥2 modes are available for the current variant** — a single available method is left to the boolean feature toggle (no redundant one-pill row, no disabled cross-axis buttons). The effective mode is resolved by `resolveModeKey` (variant + hardware availability, then user pick → `variant.default_modes` → `feature.default_mode` → first available). Keep the boolean `args[]` form for recipes that ship a single method — the two forms coexist and single-method recipes need no change. Selected mode syncs to the URL as `?fmode=spec_decoding:dflash` (default omitted). Synthesis + UI helpers: `defaultModeFor`/`isModeSupported` in `command-synthesis.js`.
    - **Method vs. checkpoint — the two axes stay separate.** The **served checkpoint** (`vllm serve <X>`) is always owned by the **Variant** row (`variant.model_id`); a **mode** only picks the `--speculative-config` method. So: a method that adds an *externally-trained* draft (a separate speculator repo named **inside** the config `model`, e.g. GLM's `RedHatAI/GLM-5.2-speculator.dspark-preview`) is a pure **mode** — `<X>` is unchanged. A method that ships as a *fused checkpoint you serve instead* (the draft is baked in, `--speculative-config` carries **no** `model`, e.g. official `deepseek-ai/DeepSeek-V4-Pro-DSpark`) is a **variant** (its `model_id`) **plus** a mode for the config. Couple them with **`variant.default_modes: { <feature>: <mode> }`** (see `variants` below): selecting the DSpark variant flips `spec_decoding` to the dspark method by default, while the checkpoint still runs MTP if the user switches the mode back. Don't put `--speculative-config` in a variant's `extra_args` (last-wins dedupe would let the boolean/mode feature shadow it) — keep it in the mode.
    - **Per-mode hardware gating** — a mode may carry `hardware_overrides.<gen>.args` (per-generation arg swap, same semantics as feature-level) and a tri-state `hardware: { <gen_or_gpu_id>: unsupported }` to gate a mode to specific GPUs (e.g. a DFlash draft that only ships for Blackwell → `hardware: { hopper: unsupported }`); the pill is struck-through on incompatible hardware and the selection auto-falls-back to the first supported mode.
  - **Per-strategy gating** — a feature may declare `strategies: [<strategy_id>, …]` (allowlist; absent = every strategy). Gate a feature when its args collide with a strategy's own flags or its setup doesn't generalize (e.g. a `--kv-transfer-config` that would clobber pd_cluster's NixlConnector — features emit last, so last-wins dedupe would win). On excluded strategies the pill renders disabled ("Not available with this strategy"); the toggle state is preserved and synthesis independently skips the feature's args and companion. Helper: `isFeatureAllowedForStrategy` in `command-synthesis.js`. NB: KV offloading itself is NOT a per-recipe feature in this repo — it lives on the universal KV Offload row (taxonomy `kv_offload` + `kv_store/`), which reuses the same gating idea via `isKvOffloadAllowedForStrategy`.
  - **Companion process** — a feature may declare `companion: { label, description?, command }` for a helper that must run alongside `vllm serve` on the same node. While the feature is on, the single-node command block renders PD-style tabs (`vLLM Serve · <label>`); the companion tab shows the raw command (no docker wrap, no env prelude) with the description as a `#` comment line. Carried in the JSON API as `companions[]` on single-node renderings. The taxonomy's `kv_offload.<key>.companion` (LMCache's `lmcache server`) flows through the same machinery.
- `opt_in_features[]` — subset of feature keys that default OFF; others default ON. Typically `spec_decoding` is opt-in. (For a modes-feature, opt-in still governs the on/off toggle; the mode row only appears once the feature is on.)
- `variants` — must include a `default` variant. Each has `precision`, `vram_minimum_gb`, optional `description`, optional `model_id` override (for quantized checkpoints shipped under a different HF repo, e.g. `nvidia/*-NVFP4`), optional `extra_args`/`extra_env`, optional `label` (overrides the pill text, which otherwise shows `precision.toUpperCase()` — use it to disambiguate two variants of the same precision, e.g. a fused-spec checkpoint → `label: "DSpark"`), optional `default_modes: { <feature>: <mode> }` (steers a modes-feature's default mode when this variant is selected **and** auto-enables that feature even if it's in `opt_in_features` — e.g. the DSpark checkpoint variant turns `spec_decoding` on and defaults to the dspark method; see spec_decoding modes above), optional `min_vllm_version` / `nightly_required` (bump the recipe baseline for a variant that needs a newer vLLM — the Install block shows the higher version and, if `nightly_required`, switches to the nightly wheel index + tag; e.g. the DSpark checkpoint requires 0.25.0, currently nightly), optional `tp` (per-variant single-node TP override; either a bare number or a hardware-aware map keyed by exact GPU id, generation, brand, or `default`, e.g. `{ b300: 1, h200: 2 }`. Use it when a variant has documented per-GPU layouts or stricter sharding constraints — typically FP8-block-quantized siblings whose `moe_intermediate_size / TP` must stay divisible by `block_n=128`. Do **not** put `--tensor-parallel-size` inside `extra_args`; it gets silently overridden by the strategy emitter), optional `hardware_overrides` with **two disjoint keyspaces**: a **generation** key (`hopper`/`blackwell`/`amd`/`nvidia`) layers **ADDITIVELY** on top of the recipe-level `hardware_overrides.<gen>` baseline — applies to **every variant including `default`** (unlike the top-level variant `extra_args`, which is a non-default checkpoint delta; the generation HO is hardware-conditional config, so `default` uses it too). Use it for a per-variant hw delta that only *some* precisions want: e.g. the FP8 checkpoints of DeepSeek-V4 add `--moe-backend deep_gemm_mega_moe` (an FP8-only MoE kernel) on Blackwell while the NVFP4 variant adds nothing — the recipe-level `blackwell` block holds only the shared `--attention_config.use_fp4_indexer_cache=True`. Precedence: a per-strategy `strategy_overrides.<s>.hardware_overrides.<gen>` still **REPLACES** both the recipe baseline *and* the variant delta (e.g. `single_node_tp` dropping the mega-kernel), so strategy HO > (recipe HO + variant HO). The other keyspace, an **exact-GPU-id** key (`b200`, `mi355x`, …), is additive too (non-default variants only) and can also carry `vram_minimum_gb` (GPU-resident fit budget after CPU weight offload — replaces the variant's full `vram_minimum_gb` for `fitsSingleNode` on that GPU only; never bump taxonomy `vram_gb` with host RAM), `docker_image` (per-variant-per-GPU image pin), `docker_serve_command` (e.g. `"vllm serve"` — restated in the `docker run` CMD before the model id for a pinned image whose entrypoint is not `vllm serve`, such as NVIDIA-base community builds that run `nvidia_entrypoint.sh`; without it the model id is executed as a path) and `install: { pip: false }` — the per-GPU form of `model.install.pip: false`, declaring that no wheel can run this recipe on that GPU so the Install block hides the pip tab there (and the command block renders `docker run`) while leaving pip intact on every other hardware. Use it when the pinned image carries kernels that aren't in vLLM main, so the generated args would abort on an official wheel — DeepSeek-V4-Flash's B12X flags on DGX Spark. Not for "docker is easier here": a pip path that merely isn't the recommended one stays offered.
  - `precision_hardware` (optional, variant-level) — overrides the default hardware constraint implied by `precision`, using `{ brand?, generation? }`. Use only for exceptional checkpoints whose runtime fallback expands compatibility beyond the stored format, such as NVFP4 weights that vLLM dequantizes to BF16 on Hopper.
  - `supported_hardware` (optional, variant-level) — an eligibility allowlist of exact profile ids or `arch:<compute_arch>` selectors, where `compute_arch` is the GPU ISA recorded once in `taxonomy.yaml` (e.g. `sm90`, `sm100`, `sm103`, `sm120`, `gfx942`, `gfx950`). Eligibility is not a verified serving claim; only `meta.hardware.<id>: verified` makes that claim. Use an architecture selector only when the checkpoint and generated runtime path work across that ISA; combine it with exact ids for profiles that need special kernels, images, or topology. This lets a new same-ISA profile inherit eligible variants after its taxonomy entry is classified, without treating every Blackwell or AMD GPU as interchangeable. Omit the list when the precision constraint alone suffices. Keep exact ids for hardware-specific checkpoints or runtime paths, and use `meta.hardware.<id>: unsupported` only for a confirmed recipe-wide blocker. Restricted profiles still require an explicit `meta.hardware.<id>` declaration before they appear in the picker, including when selected through a URL.
- `compatible_strategies[]` — subset of the serving strategy ids. Dense models typically only get `single_node_tp` + `multi_node_tp`; MoE models can have every serving strategy. Membership doubles as the default support state per GPU: listed = supported on every hardware, unlisted = supported nowhere — `strategy_hardware` overrides either default per hardware, and a strategy granted `supported` there joins the offered set without being listed here (appended after the listed ones; helper `effectiveCompatibleStrategies` in `command-synthesis.js`). Do NOT list `kv_store_*_mooncake` ids here — KV-offload options (Simple + both Mooncake modes) are implicit on every non-omni recipe and feed the "KV Offload" row, not the Strategy row.
- `strategy_min_gpus` (optional) — minimum GPUs a strategy needs **for this model**. Either a bare number (one bar for every strategy) or a map keyed by strategy id (`{ multi_node_dep: 16, multi_node_tp: 8 }`) so one parallelism can demand more than another — e.g. Kimi-K3 runs TP/TEP from 8 GPUs but needs 16 for DEP. The map takes a second, disjoint keyspace: an **exact GPU id** (`h100: 32`, or nested per-strategy `h100: { multi_node_dep: 32 }`) layering a hardware-conditional floor on top of the strategy bars, **highest wins**. Use it when one GPU needs a bigger cluster than the recipe baseline — K3's ~1.68 TB of weights fits 2 H200 nodes but needs 4 on H100, so `h100: 32`, which also drops `single_node_tp` from the H100 pill row for free (8 GPUs can't clear a 32-GPU bar). Only exact GPU ids are read, never generations: GPUs of one generation differ in VRAM, which is exactly what moves the floor. Opt-in and fail-open: absent = no floor, so existing recipes are unaffected. A multi-node strategy stays offered and *grows the node count* to meet its bar when picked (the Nodes row gains the count that first clears it, e.g. 4 on a 4-GPU NVL tray); a single-node strategy can't scale, so it drops out when the floor exceeds one node's GPUs. Applies to PD pools too, keyed by pool parallelism (`tp`/`tep`/`dep` → the matching `multi_node_*` entry) — each pool is sized to its own floor. Helpers: `minGpusForStrategy` / `nodesForStrategy` / `isStrategyReachable` in `command-synthesis.js`, shared by the builder and the JSON API. Distinct from the (currently unread) `hardware_match.min_gpus` in `strategies/*.yaml`, which is a per-strategy structural minimum rather than a per-recipe one.
- `kv_cache_strategy_hardware` (optional) — per-strategy/per-GPU opt-OUT for the Mooncake pills: `{ <kv_store_id>: { <gpu_id>: unsupported } }`. Follows the repo's fail-open hardware convention: absent = assumed to work (pill enabled on any scalable GPU); `unsupported` disables the pill and makes the JSON API skip that strategy on that hardware. "Off" and "Simple" need no gating (Simple is defined once in `taxonomy.yaml → kv_offload`).
- `strategy_hardware` (optional) — per-serving-strategy hardware overrides, `{ <strategy_id>: { <hw_key>: unsupported | supported } }`, layered on the membership default from `compatible_strategies` (listed = supported everywhere, unlisted = nowhere). Each per-strategy map takes the variant-`tp` keyspace — **exact GPU id > generation > brand (lowercase) > `default`**, first key present wins, an explicit `default:` beating the membership baseline — so both directions are one line per hardware key: `unsupported` opts a listed strategy out of a GPU/generation/brand; `supported` opts an unlisted strategy in, which also adds it to the offered set (no `compatible_strategies` entry needed — e.g. Kimi-K3 offers `multi_node_tp_dp` only via `{ h200: supported, gb200: supported, gb300: supported }`). `unsupported` renders the Strategy pill disabled with a tooltip, keeps it out of the recommended/active fallback, and makes the JSON API skip that strategy on that GPU. Use `unsupported` when a listed strategy is genuinely unusable on specific hardware (e.g. multi-node pure-TP on Hopper, which has no cross-node NVLink) — not for "untested". Helpers: `isStrategySupportedOnHardware` + `effectiveCompatibleStrategies` in `command-synthesis.js` (shared by the builder and the JSON API).
- `hardware_overrides` — optional per-generation tweaks keyed by `hopper`, `blackwell`, `amd`, or the brand-wide `nvidia`. Generation-specific blocks take precedence over the NVIDIA fallback. Each supplies `{ extra_args[], extra_env{} }`.
- `decode_context_parallel` (optional) — `{ sizes, comm_backend?, match_tp? }` adds a **Decode Context Parallelism** control to the command builder: its own row for aggregated strategies, and a `DCP` line inside the **Nodes** row (beside the Prefill/Decode TP/TEP/DEP controls) under `pd_cluster`, where one shared pick covers both roles. `sizes` lists the DCP sizes qualified for the recipe, either a bare list or a hardware-keyed map of lists (exact GPU id > generation > brand > `default`), so a size is never offered on hardware it was not run on. The row offers **Off** plus each size that divides the tensor-parallel size of *every* role in the current topology (both pools under `pd_cluster`, the single engine otherwise), so one pick keeps every node tab consistent; a size that does not divide TP is never rendered. `match_tp: true` narrows that to sizes *equal* to TP, for recipes qualified only at DCP = TP. That is a statement about what the recipe has been run at, not a vLLM constraint, and it also hides the row when prefill and decode have different TP; leave it off for recipes qualified at other divisors. The default pick is the `--decode-context-parallel-size` the layered recipe args already carry (the qualified lane value) when the topology can run it, else Off. A pick rewrites that flag in every role's command (and adds it, with `comm_backend` as `--dcp-comm-backend`, when absent); **Off** drops both flags. Hardware with no `sizes` entry shows no row and is left untouched. The pick syncs to the URL as `?dcp=<size>` (`1` = Off) and is exposed on the resolved command as `dcp: { sizes, baseline, selected, tp }`.
- `strategy_overrides` — optional per-strategy tweaks (rare). For `pd_cluster` specifically: `prefill`/`decode` sub-objects set each pool's `nodes`, `parallelism` (`tp`|`tep`|`dep`, or a hardware-keyed map of them like `nodes`), and `tp`; `router.intra_node_data_parallel_size` pins the router flag that otherwise derives from the pools' `max(dp_local)`; plus optional `pool_modes: [<mode>, …]` — narrows the PD-pool parallelism pills INDEPENDENTLY of the Strategy row. PD-pool modes are otherwise derived from `compatible_strategies[]` suffixes (via `pdPoolModes`), so `pool_modes` is how you offer TP/TEP as standalone serving strategies while locking disaggregated PD to a subset (e.g. `pool_modes: [dep]` → both pools DEP-only, single mode hides the pills; synthesis snaps any resolved mode not in the set to the first allowed one, so the JSON API stays consistent too).
- `presets` (optional) — tuned presets, `{ <key>: { label, hardware, strategy?, variants?, workload?, description?, default?, source?, image?, select?, args?, env?, remove_args?, json_args?, roles?, points?, runbook? } }`: a named configuration qualified on one exact GPU — and, with `strategy` / `variants`, one serving strategy and some checkpoints — such as a benchmark lane. It is a **delta on the baseline command**, applied after every other override layer: `args`/`env` reach every node (both roles under PD); `roles.<prefill|decode>` adds per-role `args`/`env` (and `remove_args`/`json_args`) and sets that pool's `nodes` / `parallelism` on this GPU, so the lane's topology travels with its flags. Synthesis mounts the active preset into the exact-GPU layers (`strategy_overrides.<s>.hardware_overrides.<gpu>`, `strategy_overrides.pd_cluster.<role>.hardware_overrides.<gpu>` and the role's `nodes`/`parallelism` maps), so moving such a block into a `default: true` preset leaves every rendered command unchanged. **`remove_args`** drops a flag (and its value) whichever layer emitted it, base args and features included — for flags with no `--no-` form. **`json_args`** edits keys inside a JSON-valued flag (`--speculative-config`, `--compilation-config`, `--attention-config`, `--kv-transfer-config`, …): the edit deep-merges into the flag's last occurrence (adding the flag if absent), `null` deletes a key, and `_`/`-` spellings match alike — so a lane changes `num_speculative_tokens` without restating the feature's whole JSON, or sizes the Simple offload through its `--kv-transfer-config`. **`points`** are named operating points (one per client concurrency, or arms such as eval vs throughput), keyed by id because concurrency values can repeat: each has a `label`, optional `concurrency` / `description` / `default`, and the preset's own delta fields (`args`, `env`, `remove_args`, `json_args`, `roles` incl. `nodes`/`parallelism`, `select`, `image`), layered on the preset; a point can't change the hardware, strategy or variants (that is another preset). **`select`** names the builder choices the preset implies — `pd_router`, `frontend`, `features` (`true`/`false`, or a mode key, which also turns the feature on), `dcp`, `kv_offload` — and is applied when the preset (or point) becomes active, except for what a shared URL sets explicitly; the JSON API renders with them too. **`image`** (a string, or `{ vllm, dynamo?, pip?, model_revision? }`) pins the docker command on this GPU for the variants the preset covers; the rest is shown with the preset and published. When the selected hardware has presets, its pill says so ("preset" / "N presets") and the builder shows a Preset row (nested under Strategy) listing all of them — presets of another strategy or variant are tagged and switch it when picked — plus **Baseline**; the selection syncs to `?preset=<key>` and `?point=<key>` (`baseline` opts out). A `default: true` preset is selected on its GPU + strategy until the user picks Baseline, and the JSON API's renderings bake it in and name it as `preset` / `preset_point`; every preset is also rendered at each point under `/<hf_id>/presets/<key>.json`. At most one default per GPU + strategy + variants; `baseline` is a reserved key; `roles` needs `strategy: pd_cluster`. Mark a preset `default: true` only when it is the configuration this GPU should serve with by default — a benchmark lane with benchmark-only settings stays opt-in. Put what the builder can't express — services, multi-engine placement, site specifics, benchmark parameters, evidence — in `runbook` (markdown), rendered server-side on its card in the Presets tab (anchor `#preset-<key>`), never launch commands or builder instructions. Use a preset for an evidence-backed configuration with a `source`, not for general hardware support: kernels or flags every workload on that GPU needs stay in `hardware_overrides`. An overlay source may add presets whose `hardware` is in its scope. Helpers: `src/lib/recipe-presets.js`.
- `omni` (optional) — vllm-omni online-serving config. Required when `meta.tasks` includes `omni`. Shape: `{ serve_binary?, port?, task_label?, task_hint?, tasks: [...] }`. `tasks` accepts bare ids from the catalog (`t2i`, `i2i`, `t2v`, `i2v`, `ti2v`, `t2a`) or `{ id, key?, model_id?, vram_minimum_gb?, description?, extra_args?, curl?, benchmark? }` overrides — Wan2.2 uses overrides to swap the served checkpoint per task. `benchmark` is a static command override for diffusion or other non-LLM endpoints that cannot use the default `vllm bench serve` command. `serve_binary: "vllm-omni serve"` swaps the binary; currently unused, and needed only for recipes pinned below vLLM 0.20.0, where the `vllm` console-script does not yet delegate to vllm-omni on `--omni`. Catalog lives at `src/lib/omni-tasks.js`.
  `key` is the unique selector and URL identity when multiple request presets share one catalog `id`; absent `key` falls back to `id`. `task_label` and `task_hint` customize that selector row.
- `notes` (optional) — `[{ title?, when, body }]`: configuration-specific guidance, shown in the command block (beside the hardware note, titles listed and bodies expandable) only while `when` matches the builder's selection. `when` is `{ hardware?, strategies?, variants?, features? }`, each a non-empty list: keys AND together, values within a key OR together; `hardware` takes the hardware-map keyspace (exact GPU id, generation, lowercase brand, `arch:<compute_arch>`), `features` matches while one of those features is on. `when` is required — a note that applies to every configuration belongs in `guide`. Use it for what a reader of *this* command needs (an env to add on MNNVL, why a flag is set on Blackwell), not for flags the builder should emit itself.
- `troubleshooting` (optional) — `[{ title, when?, body }]`: known issues, rendered in the **Troubleshooting** tab (beside Guide and Presets): entries whose `when` matches (or that have none) first, the rest folded under "Other configurations"; the tab gets a dot when a conditional entry matches. `title` is the symptom (an error message, "warmup too slow"), and becomes the anchor `#issue-<slug>`. The build rejects `when.variants` / `when.features` naming a variant or feature the recipe doesn't have. Helpers: `src/lib/recipe-context.js`.
- `guide` — markdown string (`|`-block), rendered with `react-markdown` + `remark-gfm` + `rehype-slug`.

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

**Mixed-precision quants (NVFP4 / ModelOpt) — don't trust the bytes-per-param table.** NVIDIA ModelOpt NVFP4 checkpoints are *not* uniformly 4-bit: only the MLP linears drop to NVFP4 (W4A16), while attention linears + KV cache stay FP8 and embeddings/norms stay higher precision. `hf_quant_config.json` shows `quant_algo: MIXED_PRECISION` in this case. The pure `params × 0.5 × 1.2` formula then **underestimates** — e.g. `nvidia/Qwen3.6-27B-NVFP4` is ~21.9 GB on disk, not the 13.5 GB the table implies, so `27B × 0.5 × 1.2 = 17` is wrong (the weights alone exceed it). For any mixed-precision checkpoint, size from the **real weight footprint** instead:

```bash
# total_size is in bytes → GB; then × 1.2 for KV/activation overhead
curl -sL "https://huggingface.co/<org>/<repo>/resolve/main/model.safetensors.index.json" \
  | python3 -c "import json,sys; print(round(json.load(sys.stdin)['metadata']['total_size']/1e9*1.2))"
```

So `vram_minimum_gb = ceil(real_checkpoint_GB × 1.2)` (Qwen3.6-27B-NVFP4 → `ceil(21.9 × 1.2) = 27`). The bytes-per-param table stays correct for *uniform* quants (plain int4/awq/gptq/fp8, and full-model FP4).
