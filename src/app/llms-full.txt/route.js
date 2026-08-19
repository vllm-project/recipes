// /llms-full.txt — full text of every recipe concatenated, with one
// `# Title` + `Source: <url>` block per section. Designed for AI-assistant
// retrieval-augmented use: a single fetch gives a model the entire recipes
// corpus rather than crawling page-by-page.
//
// Section format follows the emerging convention used by Mintlify, Vercel,
// and the docs.vllm.ai llms-full.txt. Each recipe section contains:
//   - Title heading
//   - Canonical URL (HTML)
//   - Markdown shadow URL (machine-readable)
//   - Publish/last-modified dates
//   - Model metadata
//   - Recommended `vllm serve` flags
//   - The recipe's full guide (markdown body)

import { getAllRecipes } from "@/lib/recipes";
import { siteUrl } from "@/lib/site-url";
import { hardwareLabel } from "@/lib/jsonld";

export const dynamic = "force-static";

function renderRecipeSection(recipe) {
  const canonical = `${siteUrl}/${recipe.hf_org}/${recipe.hf_repo}`;
  const markdown = `${canonical}.md`;
  const hw = hardwareLabel(recipe);
  const tasks = (recipe.meta?.tasks || []).join(", ");
  const provider = recipe.meta?.provider || recipe.hf_org;

  const header = [
    `# ${recipe.hf_id} on vLLM`,
    `Source: ${canonical}`,
    `Markdown: ${markdown}`,
    `Provider: ${provider}`,
    recipe.hf_released ? `Released: ${recipe.hf_released}` : null,
    recipe.meta?.date_updated ? `Updated: ${recipe.meta.date_updated}` : null,
    recipe.model?.architecture ? `Architecture: ${recipe.model.architecture}` : null,
    recipe.model?.parameter_count ? `Parameters: ${recipe.model.parameter_count}` : null,
    recipe.model?.context_length
      ? `Context length: ${recipe.model.context_length.toLocaleString()}`
      : null,
    recipe.model?.min_vllm_version ? `Min vLLM: ${recipe.model.min_vllm_version}` : null,
    tasks ? `Tasks: ${tasks}` : null,
    hw ? `Verified hardware: ${hw}` : null,
    recipe.meta?.description ? `Summary: ${recipe.meta.description}` : null,
  ]
    .filter(Boolean)
    .join("\n");

  // Variants and base args — the bits an AI assistant actually needs to
  // answer "how do I serve this?". The interactive command builder also
  // composes these at runtime, but exposing them in plain text means a
  // retriever can answer without crawling the JS bundle.
  const variantLines = [];
  if (recipe.variants && Object.keys(recipe.variants).length > 0) {
    variantLines.push("");
    variantLines.push("## Variants");
    for (const [name, v] of Object.entries(recipe.variants)) {
      const precision = v.precision ? `precision=${v.precision}` : "";
      const vram = v.vram_minimum_gb ? `vram_minimum_gb=${v.vram_minimum_gb}` : "";
      const modelId = v.model_id ? `model_id=${v.model_id}` : "";
      const bits = [precision, vram, modelId].filter(Boolean).join(", ");
      variantLines.push(`- ${name}: ${bits || "(default)"}`);
      if (v.description) variantLines.push(`  ${v.description}`);
    }
  }

  const baseArgs = recipe.model?.base_args || [];
  const baseEnv = recipe.model?.base_env || {};
  const argsLines = [];
  if (baseArgs.length > 0 || Object.keys(baseEnv).length > 0) {
    argsLines.push("");
    argsLines.push("## Recommended base flags");
    if (baseArgs.length > 0) {
      argsLines.push("```");
      argsLines.push(`vllm serve ${recipe.model?.model_id || recipe.hf_id} \\`);
      for (let i = 0; i < baseArgs.length; i++) {
        const trailing = i < baseArgs.length - 1 ? " \\" : "";
        argsLines.push(`  ${baseArgs[i]}${trailing}`);
      }
      argsLines.push("```");
    }
    if (Object.keys(baseEnv).length > 0) {
      argsLines.push("");
      argsLines.push("Environment:");
      argsLines.push("```");
      for (const [k, v] of Object.entries(baseEnv)) {
        argsLines.push(`${k}=${v}`);
      }
      argsLines.push("```");
    }
  }

  const guide = recipe.guide
    ? ["", "## Guide", "", recipe.guide.trim()].join("\n")
    : "";

  return [header, variantLines.join("\n"), argsLines.join("\n"), guide]
    .filter(Boolean)
    .join("\n");
}

export async function GET() {
  const recipes = getAllRecipes();

  const preamble = [
    "# vLLM Recipes — full text",
    "",
    `Concatenated full text of every recipe on ${siteUrl}, formatted for`,
    "AI-assistant retrieval. Each section is prefixed with its canonical URL,",
    "markdown shadow URL, model metadata, recommended flags, and (where",
    "present) the recipe's full guide body.",
    "",
    `Recipe count: ${recipes.length}`,
    `Generated at build time from the canonical YAML sources under \`models/\`.`,
    "",
  ].join("\n");

  const sections = recipes.map(renderRecipeSection);

  const body = [preamble, ...sections].join("\n\n---\n\n");

  return new Response(body, {
    headers: {
      "Content-Type": "text/plain; charset=utf-8",
      "Cache-Control": "public, max-age=3600, s-maxage=3600",
    },
  });
}
