/**
 * Shared utilities — safe to import from client components (no fs/path/yaml).
 */

/**
 * URL for a recipe — matches HuggingFace org/repo path.
 * e.g. /deepseek-ai/DeepSeek-V3.2 (swap huggingface.co → recipes.vllm.ai)
 */
export function recipeHref(recipe) {
  if (recipe.hf_org && recipe.hf_repo) {
    return `/${recipe.hf_org}/${recipe.hf_repo}`;
  }
  // Fallback for recipes that somehow lack hf_id derivation
  const org = recipe.meta.provider.toLowerCase().replace(/\s+/g, "-");
  return `/${org}/${recipe.meta.title}`;
}

function dateTimestamp(value) {
  const timestamp = Date.parse(value || "");
  return Number.isFinite(timestamp) ? timestamp : 0;
}

/**
 * Newest catalog additions first. `date_updated` is only a compatibility
 * fallback for recipes authored before `date_added` became part of the schema.
 */
export function compareRecipesByDateAdded(a, b) {
  const aAdded = dateTimestamp(a.meta?.date_added || a.meta?.date_updated);
  const bAdded = dateTimestamp(b.meta?.date_added || b.meta?.date_updated);
  if (aAdded !== bAdded) return bAdded - aAdded;
  return (a.hf_id || "").localeCompare(b.hf_id || "");
}
