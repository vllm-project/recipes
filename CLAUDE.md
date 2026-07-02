# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Two overlapping things live here:

1. **A content repo** for vLLM recipes — structured YAML at `models/<hf_org>/<hf_repo>.yaml`, one per model. The path mirrors HuggingFace so the site URL `/<hf_org>/<hf_repo>` matches `huggingface.co/<hf_org>/<hf_repo>`.
2. **A Next.js 15 site** (App Router, React 19, Tailwind v4) that renders those YAMLs as an interactive command builder + static JSON API under `public/`. The older top-level Markdown directories (`DeepSeek/`, `Qwen/`, etc.) and `mkdocs.yml` are the legacy MkDocs site, kept for reference during migration.

The site is the canonical surface. New work goes into YAMLs + the Next.js app; the `.md` files are historical sources for the migrated YAMLs.

## Commands

```bash
pnpm dev                                # Next dev server (do not kill or rebuild as part of a task)
pnpm build                              # Regenerates JSON API + fetches logos/HF dates, then next build
pnpm lint                               # next lint (ESLint)
node scripts/build-recipes-api.mjs      # Rebuild the JSON API only — fastest validation that all YAMLs parse
node scripts/fetch-provider-logos.mjs   # Re-download HF org avatars to public/providers/
node scripts/fetch-hf-dates.mjs         # Refresh public/hf-dates.json (HF createdAt per model)
bash scripts/hf-info.sh <org>/<repo>    # Pull config.json / params.json for a model (used when authoring recipes)
```

There is no unit-test suite. The build-recipes-api script is the de-facto validator: if it prints `✓ JSON API: N models, 7 strategies` the YAML set is internally consistent.

**Dev-server convention:** don't kill or restart the dev server as part of a task, and don't run `pnpm build` for each change. Use `node scripts/build-recipes-api.mjs` for fast feedback on YAML edits; rely on Next's HMR for `src/` edits.

**Commit convention:** always use `git commit -s` (sign-off). The upstream `vllm-project/recipes` repo enforces DCO via the probot/dco GitHub App — commits without a `Signed-off-by:` trailer fail CI and block the PR. There's no GitHub-side auto-fix; the trailer has to be in the commit itself.

## Architecture

### Data pipeline

```
models/<org>/<repo>.yaml  ──┐
strategies/*.yaml          ─┼──► src/lib/*.js  ─► Next.js pages + JSON API in public/
taxonomy.yaml              ─┘       (SSG, no runtime DB)
```

- `taxonomy.yaml` — the controlled vocabulary: hardware profiles (GPUs/node + VRAM), tasks, strategy ids. Every recipe references ids from here.
- `strategies/*.yaml` — 7 deployment strategies: `single_node_{tp,tep,dep}`, `multi_node_{tp,tep,dep}`, `pd_cluster`. Each defines the vLLM flags and env for that parallelism pattern.
- `src/lib/recipes.js`, `strategies.js`, `taxonomy.js` — filesystem readers that parse YAML once per process.
- `src/lib/command-synthesis.js` — **the core of the site**. Pure functions that take `(recipe, variantKey, strategyName, hwProfileId, features, strategies, taxonomy, advancedArgs, nodeCount)` and return either a single `vllm serve` command, a pair of head+worker commands (multi-node), or a prefill/decode pair (pd_cluster). All parallel flag decisions, per-generation hardware overrides, and multi-node mp args (`--nnodes`/`--node-rank`/`--master-addr` for TP, `--data-parallel-*` for DEP) live here.
- `scripts/build-recipes-api.mjs` — writes `public/taxonomy.json`, `public/strategies/*.json`, and `public/<hf_org>/<hf_repo>.json` at build time so agents can consume recipes without the website.

### Next.js App Router

- `src/app/page.js` — homepage (provider grid, alphabetical).
- `src/app/[org]/page.js` — org overview, task-grouped sections.
- `src/app/[org]/[repo]/page.js` — **the recipe detail page**. SSG via `generateStaticParams` over all YAMLs. Renders header + `<CommandBuilder>` + guide markdown (via `react-markdown`, not MDX — MDX chokes on `<think>` and `<64K` literals in guide text).
- `src/app/[org]/layout.js` — wraps every recipe/org page with the left `<ModelSidebar>`, sorted by HF release date desc.
- `src/components/recipes/CommandBuilder.jsx` — the interactive piece. Client component. Holds state for `variant`, `hwId`, `strategyOverride`, `nodeCount`, `features`, `advanced`; syncs all of them to URL query params; renders Hardware / Variant / Strategy / Nodes / Features / Advanced rows + the sticky command block. Calls `resolveCommand` on every change.

### Recipe YAML schema (authoritative)

Top-level keys, in this order:

- `meta` — title, slug, provider, description, date_updated, difficulty, tasks[], performance_headline, related_recipes[], optional `hardware: { gpu_id: verified | unsupported }`. Tri-state: `verified` = tested end-to-end (green ✓ badge); `unsupported` = not yet runnable on this hardware today — compat gap / missing kernel / upstream blocker that could lift later (pill rendered but disabled + struck through, tooltip says "Not yet supported"); absent = silent default, assumed to work. Don't use `unsupported` for "untested" — only when you've confirmed it currently fails.
- `model` — `model_id`, `min_vllm_version`, `architecture` (`dense` or `moe`), `parameter_count`, `active_parameters`, `context_length`, `base_args[]`, `base_env{}`.
- `dependencies` (optional) — list of `{ command, note?, optional?, brand? }`. Rendered as the "extra install" block above `vllm serve`. Use this for DeepGEMM pins, `vllm[audio]`, pinned transformers commits, vllm-omni source installs, etc. `optional: true` dims the entry and excludes it from the block's "Copy all" button (the user must copy it explicitly). `brand: NVIDIA | AMD | Intel` (or array) hides the entry when the current hardware pill is a different brand — use for CUDA-only kernels (xformers, DeepGEMM) or ROCm-only wheels. Omit `brand` for platform-agnostic deps (diffusers from source, transformers commits, etc.).
- `features` — map of `{feature_key: { description, args[] }}`. Naming convention: `tool_calling`, `reasoning`, `spec_decoding` (the unified key for MTP / Eagle3 / ERNIE-MTP — **don't use `mtp`**, it was renamed across all recipes). Each feature is a boolean toggle by default.
  - **Single-select sub-modes** — when one feature has several *mutually exclusive* methods (e.g. a model offering more than one speculative method: MTP / DSpark / DFlash), replace the feature's flat `args[]` with a `modes` map and it renders as a "pick one" segmented row under the feature toggle instead of a boolean. **A mode contributes only `args` (the `--speculative-config`) — the method, never the served checkpoint.** Shape: `{ description, default_mode?, modes: { <mode_key>: { label?, description?, args[], hardware_overrides?, hardware? } } }`. Only the selected mode's `args` are emitted, so exactly one `--speculative-config` is ever produced — no dedupe conflict between sibling methods, and no `disables_features` gymnastics. `default_mode` picks the initial mode (else the first key). Keep the boolean `args[]` form for recipes that ship a single method — the two forms coexist and single-method recipes need no change. Selected mode syncs to the URL as `?fmode=spec_decoding:dflash` (default omitted). Synthesis + UI helpers: `defaultModeFor`/`isModeSupported` in `command-synthesis.js`.
    - **Method vs. checkpoint — the two axes stay separate.** The **served checkpoint** (`vllm serve <X>`) is always owned by the **Variant** row (`variant.model_id`); a **mode** only picks the `--speculative-config` method. So: a method that adds an *externally-trained* draft (a separate speculator repo named **inside** the config `model`, e.g. GLM's `RedHatAI/GLM-5.2-speculator.dspark-preview`) is a pure **mode** — `<X>` is unchanged. A method that ships as a *fused checkpoint you serve instead* (the draft is baked in, `--speculative-config` carries **no** `model`, e.g. official `deepseek-ai/DeepSeek-V4-Pro-DSpark`) is a **variant** (its `model_id`) **plus** a mode for the config. Couple them with **`variant.default_modes: { <feature>: <mode> }`** (see `variants` below): selecting the DSpark variant flips `spec_decoding` to the dspark method by default, while the checkpoint still runs MTP if the user switches the mode back. Don't put `--speculative-config` in a variant's `extra_args` (last-wins dedupe would let the boolean/mode feature shadow it) — keep it in the mode.
    - **Per-mode hardware gating** — a mode may carry `hardware_overrides.<gen>.args` (per-generation arg swap, same semantics as feature-level) and a tri-state `hardware: { <gen_or_gpu_id>: unsupported }` to gate a mode to specific GPUs (e.g. a DFlash draft that only ships for Blackwell → `hardware: { hopper: unsupported }`); the pill is struck-through on incompatible hardware and the selection auto-falls-back to the first supported mode.
- `opt_in_features[]` — subset of feature keys that default OFF; others default ON. Typically `spec_decoding` is opt-in. (For a modes-feature, opt-in still governs the on/off toggle; the mode row only appears once the feature is on.)
- `variants` — must include a `default` variant. Each has `precision`, `vram_minimum_gb`, optional `description`, optional `model_id` override (for quantized checkpoints shipped under a different HF repo, e.g. `nvidia/*-NVFP4`), optional `extra_args`/`extra_env`, optional `label` (overrides the pill text, which otherwise shows `precision.toUpperCase()` — use it to disambiguate two variants of the same precision, e.g. a fused-spec checkpoint → `label: "DSpark"`), optional `default_modes: { <feature>: <mode> }` (steers a modes-feature's default when this variant is selected — see spec_decoding modes above), optional `tp` (per-variant single-node TP override; use when a non-default variant has stricter sharding constraints than the default — typically FP8-block-quantized siblings whose `moe_intermediate_size / TP` must stay divisible by `block_n=128`. Do **not** put `--tensor-parallel-size` inside `extra_args`; it gets silently overridden by the strategy emitter).
- `compatible_strategies[]` — subset of the 7 strategy ids. Dense models typically only get `single_node_tp` + `multi_node_tp`; MoE models can have all seven.
- `hardware_overrides` — optional per-generation tweaks keyed by `hopper`, `blackwell`, `amd`. Each: `{ extra_args[], extra_env{} }`.
- `strategy_overrides` — optional per-strategy tweaks (rare).
- `omni` (optional) — vllm-omni online-serving config. Required when `meta.tasks` includes `omni`. Shape: `{ serve_binary?, port?, tasks: [...] }`. `tasks` accepts bare ids from the catalog (`t2i`, `i2i`, `t2v`, `i2v`, `ti2v`, `t2a`) or `{ id, model_id?, vram_minimum_gb?, description?, extra_args?, curl? }` overrides — Wan2.2 uses overrides to swap the served checkpoint per task. `serve_binary: "vllm-omni serve"` swaps the binary for handlers that don't ship in the `vllm` console-script (today: stable-audio-open). Catalog lives at `src/lib/omni-tasks.js`.
- `guide` — markdown string (`|`-block), rendered with `react-markdown` + `remark-gfm` + `rehype-slug`.

**VRAM formula**: `vram_minimum_gb = ceil(params × bytes × 1.2)`. Bytes per param: bf16/fp16=2, fp8/int8/awq/gptq=1, int4/nvfp4/fp4/mxfp4=0.5. For MoE, use total params (inactive experts still live in VRAM). **Exception — mixed-precision quants (NVFP4/ModelOpt, `hf_quant_config.json` → `quant_algo: MIXED_PRECISION`):** only MLP linears are 4-bit while attention/KV/embeddings stay FP8+, so the table underestimates. Size from the real checkpoint instead: `ceil(safetensors total_size_GB × 1.2)`.

### Non-obvious design decisions

- **Hardware is never blocked by VRAM.** Multi-node scales VRAM, so `command-synthesis.js` only uses VRAM as a display hint. Precision constraints (NVFP4/FP4 require Blackwell) are the only things that disable a hardware pill.
- **Default hardware is always H200** (or B200 for Blackwell-constrained variants). `pickDefaultHardware` prefers NVIDIA over AMD; `loadPreferences` deliberately ignores AMD entries from localStorage so first-page-load is always H200.
- **Omni recipes render `vllm serve --omni`** via a dedicated `resolveOmniCommand` path (no strategy / multi-node / pd). The command builder adds a Task pill row (T2I / I2I / T2V / I2V / TI2V / T2A) above Hardware, with task-specific cURL examples in the popover. Detection: `meta.tasks` includes `omni`. The `--omni` flag is auto-injected, so don't add it to `model.base_args`.
- **Multi-node uses vLLM mp (multiprocessing) backend**, not Ray. For TP/TEP every node runs the same command varying `--node-rank`; rank>0 adds `--headless`. For DEP, Node N adds `--data-parallel-start-rank = N × gpus_per_node`. The UI shows exactly two tabs (Head + Node 1) as a 2-node example.
- **Taxonomy hardware specs** follow [inferencex.semianalysis.com/gpu-specs](https://inferencex.semianalysis.com/gpu-specs): B200 SXM is 180 GB/GPU (not the nameplate 192, which is pre-ECC), GB200/GB300 are **4-GPU compute trays** (NVL4, the NVL72 rack unit), B300 is 268 GB/GPU.
- **Legacy Markdown under `DeepSeek/`, `Qwen/` etc. is read-only reference.** When migrating or updating a recipe, read those to extract flags and env, but write the YAML; don't edit the Markdown.
- **`public/` is generated** — don't commit it. `.gitignore` should cover it; if something slips through, leave it untracked.

## Contributing a new recipe

A dedicated skill lives at `.claude/skills/add-recipe/SKILL.md` and is auto-loaded when the user asks to add/create/contribute a recipe. It covers HF metadata extraction, the full schema, VRAM formula, provider-registry updates, and the validation/commit loop.
