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
  // Sort models within each org by HF release date (newest first).
  // Fallback to repo-name alphabetical when date missing/equal.
  for (const [, models] of sorted) {
    models.sort((a, b) => {
      const da = a.hf_released ? new Date(a.hf_released).getTime() : 0;
      const db = b.hf_released ? new Date(b.hf_released).getTime() : 0;
      if (da !== db) return db - da;
      return (a.hf_repo || "").localeCompare(b.hf_repo || "");
    });
  }

  return (
    <div className="max-w-[1480px] mx-auto px-4 sm:px-6 flex gap-6">
      <ModelSidebar recipesByOrg={sorted} />
      <div className="flex-1 min-w-0">
        {children}
      </div>
    </div>
  );
}
