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
- `dependencies` (optional) — list of `{ command, note?, optional? }`. Rendered as the "extra install" block above `vllm serve`. Use this for DeepGEMM pins, `vllm[audio]`, pinned transformers commits, vllm-omni source installs, etc.
- `features` — map of `{feature_key: { description, args[] }}`. Naming convention: `tool_calling`, `reasoning`, `spec_decoding` (the unified key for MTP / Eagle3 / ERNIE-MTP — **don't use `mtp`**, it was renamed across all recipes).
- `opt_in_features[]` — subset of feature keys that default OFF; others default ON. Typically `spec_decoding` is opt-in.
- `variants` — must include a `default` variant. Each has `precision`, `vram_minimum_gb`, optional `description`, optional `model_id` override (for quantized checkpoints shipped under a different HF repo, e.g. `nvidia/*-NVFP4`), optional `extra_args`/`extra_env`.
- `compatible_strategies[]` — subset of the 7 strategy ids. Dense models typically only get `single_node_tp` + `multi_node_tp`; MoE models can have all seven.
- `hardware_overrides` — optional per-generation tweaks keyed by `hopper`, `blackwell`, `amd`. Each: `{ extra_args[], extra_env{} }`.
- `strategy_overrides` — optional per-strategy tweaks (rare).
- `guide` — markdown string (`|`-block), rendered with `react-markdown` + `remark-gfm` + `rehype-slug`.

**VRAM formula**: `vram_minimum_gb = ceil(params × bytes × 1.2)`. Bytes per param: bf16/fp16=2, fp8/int8/awq/gptq=1, int4/nvfp4/fp4/mxfp4=0.5. For MoE, use total params (inactive experts still live in VRAM).

### Non-obvious design decisions

- **Hardware is never blocked by VRAM.** Multi-node scales VRAM, so `command-synthesis.js` only uses VRAM as a display hint. Precision constraints (NVFP4/FP4 require Blackwell) are the only things that disable a hardware pill.
- **Default hardware is always H200** (or B200 for Blackwell-constrained variants). `pickDefaultHardware` prefers NVIDIA over AMD; `loadPreferences` deliberately ignores AMD entries from localStorage so first-page-load is always H200.
- **Omni recipes skip the command builder entirely** — `CommandBuilder` returns an "Served via vLLM-Omni" notice instead, because these models use offline Python scripts, not `vllm serve`. Detection: `meta.tasks` includes `omni`.
- **Multi-node uses vLLM mp (multiprocessing) backend**, not Ray. For TP/TEP every node runs the same command varying `--node-rank`; rank>0 adds `--headless`. For DEP, Node N adds `--data-parallel-start-rank = N × gpus_per_node`. The UI shows exactly two tabs (Head + Node 1) as a 2-node example.
- **Taxonomy hardware specs** follow [inferencex.semianalysis.com/gpu-specs](https://inferencex.semianalysis.com/gpu-specs): B200 SXM is 180 GB/GPU (not the nameplate 192, which is pre-ECC), GB200/GB300 are **4-GPU compute trays** (NVL4, the NVL72 rack unit), B300 is 268 GB/GPU.
- **Legacy Markdown under `DeepSeek/`, `Qwen/` etc. is read-only reference.** When migrating or updating a recipe, read those to extract flags and env, but write the YAML; don't edit the Markdown.
- **`public/` is generated** — don't commit it. `.gitignore` should cover it; if something slips through, leave it untracked.

## Contributing a new recipe

A dedicated skill lives at `.claude/skills/add-recipe/SKILL.md` and is auto-loaded when the user asks to add/create/contribute a recipe. It covers HF metadata extraction, the full schema, VRAM formula, provider-registry updates, and the validation/commit loop.
