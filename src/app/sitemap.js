import { getAllRecipes } from "@/lib/recipes";
import { siteUrl } from "@/lib/site-url";

export default function sitemap() {
  const recipes = getAllRecipes();

  const recipeEntries = recipes.map((r) => ({
    url: `${siteUrl}/${r.hf_org}/${r.hf_repo}`,
    lastModified: r.meta.date_updated || undefined,
    changeFrequency: "weekly",
    priority: 0.8,
  }));

  const orgs = [...new Set(recipes.map((r) => r.hf_org))];
  const orgEntries = orgs.map((org) => {
    const latest = recipes
      .filter((r) => r.hf_org === org)
      .map((r) => r.meta.date_updated)
      .filter(Boolean)
      .sort()
      .at(-1);
    return {
      url: `${siteUrl}/${org}`,
      lastModified: latest || undefined,
      changeFrequency: "weekly",
      priority: 0.6,
    };
  });

  const homeLastModified = recipes
    .map((r) => r.meta.date_updated)
    .filter(Boolean)
    .sort()
    .at(-1);

  return [
    {
      url: siteUrl,
      lastModified: homeLastModified || undefined,
      changeFrequency: "daily",
      priority: 1.0,
    },
    ...orgEntries,
    ...recipeEntries,
  ];
}
