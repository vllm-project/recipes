import { getAllRecipes } from "@/lib/recipes";
import { ModelSidebar } from "@/components/recipes/ModelSidebar";

export default async function OrgLayout({ children }) {
  const recipes = getAllRecipes();

  // Group by HF org, sorted by count desc
  const byOrg = {};
  for (const r of recipes) {
    const org = r.hf_org || "unknown";
    if (!byOrg[org]) byOrg[org] = [];
    byOrg[org].push(r);
  }
  const sorted = Object.entries(byOrg).sort((a, b) => b[1].length - a[1].length);

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 flex gap-6">
      <ModelSidebar recipesByOrg={sorted} />
      <div className="flex-1 min-w-0">
        {children}
      </div>
    </div>
  );
}
