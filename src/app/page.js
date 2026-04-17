import { Suspense } from "react";
import { getAllRecipes } from "@/lib/recipes";
import { RecipeCardGrid } from "@/components/recipes/RecipeCardGrid";

export const metadata = {
  title: "vLLM Recipes",
  description: "How do I run model X on hardware Y? Pick a model, get a working vllm serve command.",
};

export default async function HomePage() {
  const recipes = getAllRecipes();

  return (
    <main className="max-w-6xl mx-auto px-4 sm:px-6 py-8">
      {/* ── Hero (compact) ── */}
      <header className="mb-6">
        <h1 className="text-xl sm:text-2xl font-bold tracking-tight">
          How do I run <span className="text-vllm-blue">model X</span> on{" "}
          <span className="text-vllm-blue">hardware Y</span>?
        </h1>
        <p className="text-sm text-muted-foreground mt-1">
          Community-maintained recipes for deploying selected models on selected hardware with vLLM.{" "}
          For the full list of supported models and hardware, see{" "}
          <a
            href="https://vllm.ai/#compatibility"
            target="_blank"
            rel="noopener noreferrer"
            className="text-vllm-blue hover:underline"
          >
            vllm.ai/#compatibility
          </a>
          .
        </p>
      </header>

      {/* ── Recipe catalog ── */}
      <Suspense fallback={<div className="text-sm text-muted-foreground py-8">Loading...</div>}>
        <RecipeCardGrid recipes={recipes} />
      </Suspense>
    </main>
  );
}
