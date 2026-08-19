import { Suspense } from "react";
import { getAllRecipes } from "@/lib/recipes";
import { BrowseList } from "@/components/recipes/BrowseList";

export const metadata = {
  title: "Browse all recipes",
  description: "Filter every vLLM recipe by task, architecture, parameter size, precision, hardware, and provider.",
};

export default function BrowsePage() {
  const recipes = getAllRecipes();
  // Slim payload — drop the heavy `guide` markdown before sending to the
  // client, same shape the search box uses but with the fields BrowseList
  // needs for filtering and display.
  const slim = recipes.map((r) => {
    const v = r.variants?.default || {};
    return {
      hf_id: r.hf_id,
      hf_org: r.hf_org,
      hf_repo: r.hf_repo,
      hf_released: r.hf_released || null,
      meta: {
        title: r.meta?.title || r.hf_repo,
        provider: r.meta?.provider || r.hf_org,
        description: r.meta?.description || "",
        performance_headline: r.meta?.performance_headline || "",
        date_updated: r.meta?.date_updated || null,
        tasks: r.meta?.tasks || [],
        hardware: r.meta?.hardware || {},
      },
      model: {
        architecture: r.model?.architecture || "dense",
        parameter_count: r.model?.parameter_count || "",
        active_parameters: r.model?.active_parameters || null,
        context_length: r.model?.context_length || 0,
      },
      variant: {
        precision: v.precision || null,
        vram_minimum_gb: v.vram_minimum_gb || null,
      },
      // All distinct variant precisions, not just the default. NVFP4 / MXFP4
      // / INT4 / etc. typically ship as *non-default* variants (quantized
      // checkpoints), so a precision filter that only inspects the default
      // would miss them. Used by BrowseList for both counts and matching.
      precisions: [...new Set(
        Object.values(r.variants || {}).map((vv) => vv?.precision).filter(Boolean)
      )],
    };
  });

  return (
    <main className="max-w-[1480px] mx-auto px-4 sm:px-6 py-8">
      <header className="mb-5">
        <h1 className="text-xl font-semibold tracking-tight">Browse all recipes</h1>
        <p className="text-sm text-muted-foreground mt-1">
          Filter {slim.length} recipes by task, architecture, size, precision, and hardware.
        </p>
      </header>
      <Suspense fallback={<div className="text-sm text-muted-foreground py-8">Loading...</div>}>
        <BrowseList recipes={slim} />
      </Suspense>

      {/* Server-rendered crawlable index. BrowseList above is client-rendered,
          so without this the model pages have no internal links pointing at
          them and search engines can only discover them via the sitemap. A
          plain <a> list gives every recipe a real internal link from this
          indexable hub page. */}
      <nav aria-label="All recipes" className="mt-12 pt-6 border-t border-border">
        <h2 className="text-sm font-semibold text-muted-foreground mb-3">All recipes</h2>
        <ul className="columns-2 sm:columns-3 lg:columns-4 gap-x-6 text-sm">
          {slim
            .slice()
            .sort((a, b) => a.hf_id.localeCompare(b.hf_id))
            .map((r) => (
              <li key={r.hf_id} className="mb-1 break-inside-avoid">
                <a href={`/${r.hf_org}/${r.hf_repo}`} className="text-vllm-blue hover:underline">
                  {r.hf_id}
                </a>
              </li>
            ))}
        </ul>
      </nav>
    </main>
  );
}
