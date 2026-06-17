import { Suspense } from "react";
import { getAllRecipes } from "@/lib/recipes";
import { RecipeCardGrid } from "@/components/recipes/RecipeCardGrid";

export const metadata = {
  // `absolute` bypasses the layout's `%s | vLLM Recipes` template — without it
  // the homepage would render "vLLM Recipes | vLLM Recipes".
  title: { absolute: "vLLM Recipes — Deploy any model on any hardware with vLLM" },
  description: "How do I run model X on hardware Y? Pick a model, get a working vllm serve command.",
};

export default async function HomePage() {
  const recipes = getAllRecipes();

  return (
    <main className="max-w-[1480px] mx-auto px-4 sm:px-6 py-8">
      {/* ── Hero (compact) ── */}
      <header className="mb-6">
        <h1 className="sr-only">vLLM Recipes — Deploy any model on any hardware with vLLM</h1>
        <p className="text-sm text-muted-foreground max-w-3xl">
          Pick a model, adjust for your GPUs, copy the <code className="font-mono text-[12px] bg-muted/50 px-1 py-0.5 rounded">vllm serve</code> line that runs.{" "}
          Community-maintained recipes for NVIDIA H100/H200/B200/B300, Grace-Blackwell, and AMD MI300X/MI325X/MI355X.{" "}
          <a
            href="https://vllm.ai/#compatibility"
            target="_blank"
            rel="noopener noreferrer"
            className="text-vllm-blue hover:underline"
          >
            Full vLLM compatibility →
          </a>
        </p>
      </header>

      {/* ── Recipe catalog ── */}
      <Suspense fallback={<div className="text-sm text-muted-foreground py-8">Loading...</div>}>
        <RecipeCardGrid recipes={recipes} />
      </Suspense>
    </main>
  );
}
