// `/<hf_id>.md?<query>` and `/<hf_id>.json?<query>` (rewritten here by
// next.config.mjs): the configuration a recipe page URL selects, rendered on
// the server — as Markdown for AI assistants, or as the JSON API's rendering
// shape. The query string is the page's own, so any shared builder link has
// an agent-readable twin. A bare `/<hf_id>.json` stays the static file.

import { findVariantRedirect, getRecipeByHfId } from "@/lib/recipes";
import { loadStrategies } from "@/lib/strategies";
import { loadTaxonomy } from "@/lib/taxonomy";
import { resolveSelection } from "@/lib/recipe-render";
import { selectionMarkdown, selectionUrls } from "@/lib/recipe-markdown";

// Content only changes on deploy, which purges the CDN cache.
const CACHE = "public, max-age=300, s-maxage=86400, stale-while-revalidate=86400";

export async function GET(request, { params }) {
  const { format, org, repo } = await params;
  if (format !== "md" && format !== "json") return new Response("Not found", { status: 404 });
  const url = new URL(request.url);
  const query = new URLSearchParams(url.searchParams);
  // `<page URL>.md` puts the extension on the last param's value.
  for (const [k, v] of [...query]) {
    if (v.endsWith(`.${format}`)) query.set(k, v.slice(0, -format.length - 1));
  }
  let recipe = getRecipeByHfId(org, repo);
  if (!recipe) {
    // A variant's own checkpoint id renders its parent recipe at that variant.
    const v = findVariantRedirect(org, repo);
    if (!v) return new Response(`No recipe for ${org}/${repo}\n`, { status: 404, headers: { "Content-Type": "text/plain; charset=utf-8" } });
    recipe = v.parent;
    if (!query.get("variant")) query.set("variant", v.variantKey);
  }

  const strategies = loadStrategies();
  const taxonomy = loadTaxonomy();
  const resolved = resolveSelection(recipe, query, strategies, taxonomy);
  const urls = selectionUrls(url.origin, recipe.hf_id, query, resolved.selection.hardware);

  if (format === "json") {
    // Same top-level shape as the static `/<hf_id>/hw/<hw>.json` renderings,
    // plus the resolved selection and this configuration's other forms.
    return Response.json(
      { hf_id: recipe.hf_id, ...resolved.rendering, selection: resolved.selection, page_url: urls.page, markdown_url: urls.md },
      { status: resolved.rendering ? 200 : 422, headers: { "Cache-Control": CACHE } },
    );
  }
  return new Response(selectionMarkdown(recipe, resolved, { strategies, taxonomy, urls }), {
    headers: {
      "Content-Type": "text/markdown; charset=utf-8",
      "Cache-Control": CACHE,
      Link: `<${urls.page}>; rel="canonical"`,
    },
  });
}
