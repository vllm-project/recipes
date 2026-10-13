# Contributing a new recipe

This repo hosts structured YAML recipes for serving open-weight models with
vLLM. Each recipe renders on [recipes.vllm.ai](https://recipes.vllm.ai/) as an
interactive command builder (pick hardware, variant, strategy → copy the exact
`vllm serve` command) and is exported as static JSON under `public/` for
agents/tools to consume.

One recipe = one YAML file at `models/<hf_org>/<hf_repo>.yaml`. The path
mirrors HuggingFace so the site URL `/<hf_org>/<hf_repo>` matches
`huggingface.co/<hf_org>/<hf_repo>` exactly.

## Where a recipe lives

Recipes are kept in the repository of the project that serves the model; this
repo renders them (see "How recipes are maintained" in the
[README](README.md)). Open your change against the owning repo:

| Recipe | Repository and directory |
| --- | --- |
| Models served by vLLM | This repo → `models/<hf_org>/<hf_repo>.yaml` today; moving to `recipes/models/` in vllm-project/vllm ([#61302](https://github.com/vllm-project/vllm/pull/61302)) |
| Omni models (`meta.tasks` includes `omni`) | This repo → `models/<hf_org>/<hf_repo>.yaml`; moving to vllm-omni later |
| Ascend NPU content | Inline in the recipe; moving to `overlays/<hf_org>/<hf_repo>.yaml` in vllm-ascend later |

The schema ([SCHEMA.md](SCHEMA.md)) is the same everywhere. Changes to the schema, hardware
profiles (`taxonomy.yaml`), strategies, provider logos, or the site itself go
to this repo.

## Quick start

1. Fork + clone the owning repo (above) and this repo.
2. Create `models/<hf_org>/<hf_repo>.yaml` in the owning repo, starting from
   [`templates/recipe.yaml`](templates/recipe.yaml) and following
   [SCHEMA.md](SCHEMA.md). The template's first line points YAML-aware editors at
   the published schema for completion and inline errors.
3. Point this repo at your checkout with a gitignored `sources.local.yaml`
   (skip this when the recipe lives in this repo):

   ```yaml
   overrides:
     vllm: ~/vllm/recipes
   ```

4. Validate from this repo: `node scripts/build-recipes-api.mjs` — must print
   `✓ JSON API: N models, …` with no errors.
5. Preview locally: `pnpm install && pnpm dev` → open
   `http://localhost:3000/<hf_org>/<hf_repo>`.
6. Open a PR against the owning repo. Once it merges, the site picks it up
   automatically (the `bump-sources` workflow checks every 20 minutes).

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

## YAML schema

The schema — an annotated example of every field, the field reference, and
the VRAM formula — is in **[SCHEMA.md](SCHEMA.md)**, the single authoritative
copy (versioned with the renderer, so it is the same for every repository
that hosts recipes).

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

If `<hf_org>` isn't already in `src/lib/providers.js`, add an entry (in this
repo, even when the recipe itself lives in another repo):

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

After saving the YAML (with `sources.local.yaml` pointing at your checkout if
the recipe lives in another repo), run from this repo:

```bash
# Fetch the pinned commits of every remote source (first run / after a lock bump)
node scripts/sync-sources.mjs

# Fast validation: assembles every source, all YAMLs parse + JSON API builds
node scripts/build-recipes-api.mjs

# Or validate a source checkout exactly as its pull-request check does
# (strict assemble + JSON API build), without touching sources.local.yaml:
node scripts/validate-source.mjs --source vllm --dir ~/vllm/recipes

# Full preview
pnpm install
pnpm dev
# Open http://localhost:3000/<hf_org>/<hf_repo>
```

Click through Hardware / Variant / Strategy / Nodes pills and confirm the
generated `vllm serve` command matches what you'd expect. The Verify / Bench
popovers should open without clipping. While `pnpm dev` runs,
`pnpm content:watch` re-assembles when you add or remove files in a local
source.

## Commit style

In the owning repo, stage **only** the new recipe YAML, and sign off the
commit (vllm-project repos enforce DCO):

```bash
git add recipes/models/<hf_org>/<hf_repo>.yaml   # in vllm; omni recipes here: models/<hf_org>/<hf_repo>.yaml
git commit -s -m "Add <hf_org>/<hf_repo> recipe"
```

A new provider entry in `src/lib/providers.js` is a separate PR to this repo.

Do not stage:

- `public/`, `.content/`, `.sources/` — generated at build time
- `sources.local.yaml` — your local overrides
- `node_modules/`
- `site/` — the legacy MkDocs output

## Questions

- Open an issue at https://github.com/vllm-project/recipes/issues
- Or start a discussion at https://github.com/vllm-project/recipes/discussions

If you use Claude Code, the repo ships a skill at
`.claude/skills/add-recipe/SKILL.md` that walks the same process
interactively.
