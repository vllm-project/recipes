import { getAllRecipes } from "@/lib/recipes";
import { ModelSidebar } from "@/components/recipes/ModelSidebar";

export default async function OrgLayout({ children }) {
  const recipes = getAllRecipes();

  // Group by HF org, sort alphabetically (case-insensitive) for predictable navigation
  const byOrg = {};
  for (const r of recipes) {
    const org = r.hf_org || "unknown";
    if (!byOrg[org]) byOrg[org] = [];
    byOrg[org].push(r);
  }
  const sorted = Object.entries(byOrg).sort((a, b) =>
    a[0].toLowerCase().localeCompare(b[0].toLowerCase())
  );
  // Sort models within each org alphabetically too
  for (const [, models] of sorted) {
    models.sort((a, b) => (a.hf_repo || "").localeCompare(b.hf_repo || ""));
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 flex gap-6">
      <ModelSidebar recipesByOrg={sorted} />
      <div className="flex-1 min-w-0">
        {children}
      </div>
    </div>
  );
}
