// /llms.txt — curated hierarchical index of every recipe + supporting page,
// formatted per the emerging spec at https://llmstxt.org. Absolute URLs end
// in `.md`, pointing at the markdown shadow served alongside each recipe.
//
// This file is what AI assistants (Claude, ChatGPT, Perplexity, Gemini)
// retrieve when they want to know "what does recipes.vllm.ai cover and
// where do I find each topic?". Pair this with `/llms-full.txt` which
// concatenates the full text of every recipe for retrieval-augmented use.

import { getAllRecipes } from "@/lib/recipes";
import { getProviderDisplayName } from "@/lib/providers";
import { siteUrl } from "@/lib/site-url";

export const dynamic = "force-static";

function escapeMarkdownLinkText(text) {
  return String(text).replace(/[\[\]\\]/g, "\\$&");
}

export async function GET() {
  const recipes = getAllRecipes();

  // Group recipes by provider, preserving date_updated-descending order from
  // getAllRecipes(). Providers themselves are ordered by their newest recipe
  // — same heuristic the homepage uses.
  const groups = new Map();
  for (const recipe of recipes) {
    if (!groups.has(recipe.hf_org)) groups.set(recipe.hf_org, []);
    groups.get(recipe.hf_org).push(recipe);
  }

  const corePages = [
    ["Homepage", siteUrl, "Recipe catalogue: pick a model, get a working `vllm serve` command."],
    [`Browse all recipes`, `${siteUrl}/browse`, "Flat alphabetical list of every recipe."],
    [`JSON API`, `${siteUrl}/models.json`, "Machine-readable list of every recipe and its metadata."],
  ];

  const lines = [
    "# vLLM Recipes",
    "",
    "> Per-model serving recipes for vLLM: hardware-tuned `vllm serve` commands, flag explanations, and known pitfalls. Each recipe page also serves clean markdown at the same URL with a `.md` suffix.",
    "",
    "## Core pages",
    "",
    ...corePages.map(([title, url, desc]) => `- [${title}](${url}): ${desc}`),
    "",
    "## Related properties",
    "",
    `- [vLLM](https://vllm.ai): project homepage and blog.`,
    `- [vLLM documentation](https://docs.vllm.ai): full vLLM reference docs.`,
    `- [vLLM on GitHub](https://github.com/vllm-project/vllm): source code and issues.`,
    "",
    "## Recipes",
    "",
  ];

  for (const [org, recipesForOrg] of groups) {
    const providerName = getProviderDisplayName(org);
    lines.push(`### ${escapeMarkdownLinkText(providerName)}`);
    lines.push("");
    for (const r of recipesForOrg) {
      const url = `${siteUrl}/${r.hf_org}/${r.hf_repo}.md`;
      const summary = r.meta?.description
        ? r.meta.description.replace(/\s+/g, " ").trim()
        : `${r.hf_id} serving recipe.`;
      lines.push(`- [${escapeMarkdownLinkText(r.hf_id)}](${url}): ${summary}`);
    }
    lines.push("");
  }

  lines.push("## Feeds");
  lines.push("");
  lines.push(`- [Sitemap](${siteUrl}/sitemap.xml): canonical XML sitemap.`);
  lines.push(
    `- [llms-full.txt](${siteUrl}/llms-full.txt): full text of every recipe, concatenated for AI-assistant retrieval.`,
  );
  lines.push("");

  return new Response(lines.join("\n"), {
    headers: {
      "Content-Type": "text/plain; charset=utf-8",
      "Cache-Control": "public, max-age=3600, s-maxage=3600",
    },
  });
}
