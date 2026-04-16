import { Suspense } from "react";
import { getAllRecipes } from "@/lib/recipes";
import { loadTaxonomy } from "@/lib/taxonomy";
import { SearchBox } from "@/components/recipes/SearchBox";
import { RecipeCardGrid } from "@/components/recipes/RecipeCardGrid";
import { ExternalLink } from "lucide-react";

export const metadata = {
  title: "vLLM Recipes",
  description: "How do I run model X on hardware Y for task Z? Pick a model, get a working vllm serve command.",
};

export default async function HomePage() {
  const recipes = getAllRecipes();
  const taxonomy = loadTaxonomy();

  return (
    <main className="max-w-6xl mx-auto px-4 sm:px-6 py-8">
      {/* ── Hero ── */}
      <section className="mb-10">
        <h2 className="text-2xl sm:text-3xl font-bold tracking-tight text-foreground">
          Deploy any model on any hardware
        </h2>
        <p className="text-muted-foreground mt-2 max-w-2xl">
          Community-maintained deployment recipes for vLLM. Each recipe provides a verified,
          copy-ready serving configuration with interactive hardware and strategy selection.
        </p>
        <div className="flex flex-wrap gap-3 mt-4">
          {[
            { label: "Supported Models", href: "https://docs.vllm.ai/en/latest/models/supported_models.html" },
            { label: "Supported Hardware", href: "https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html" },
            { label: "Documentation", href: "https://docs.vllm.ai" },
            { label: "Contribute", href: "https://github.com/vllm-project/recipes" },
          ].map(({ label, href }) => (
            <a
              key={label}
              href={href}
              className="inline-flex items-center gap-1 text-sm text-vllm-blue hover:text-vllm-blue-hover transition-colors"
            >
              {label}
              <ExternalLink size={12} />
            </a>
          ))}
        </div>
      </section>

      {/* ── Search ── */}
      <section className="mb-8">
        <Suspense>
          <SearchBox recipes={recipes} />
        </Suspense>
      </section>

      {/* ── Recipe catalog ── */}
      <section>
        <Suspense fallback={<div className="text-sm text-muted-foreground py-8">Loading recipes...</div>}>
          <RecipeCardGrid recipes={recipes} taxonomy={taxonomy} />
        </Suspense>
      </section>
    </main>
  );
}
