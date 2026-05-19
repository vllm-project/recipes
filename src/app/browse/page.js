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
    </main>
  );
}
