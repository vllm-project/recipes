import { getAllRecipes } from "@/lib/recipes";
import { ModelSidebar } from "@/components/recipes/ModelSidebar";

export default async function ModelsLayout({ children }) {
  const recipes = getAllRecipes();

  // Group by provider, sorted by count desc
  const byProvider = {};
  for (const r of recipes) {
    const p = r.meta.provider;
    if (!byProvider[p]) byProvider[p] = [];
    byProvider[p].push(r);
  }
  const sorted = Object.entries(byProvider).sort((a, b) => b[1].length - a[1].length);

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 flex gap-6">
      <ModelSidebar recipesByProvider={sorted} />
      <div className="flex-1 min-w-0">
        {children}
      </div>
    </div>
  );
}
