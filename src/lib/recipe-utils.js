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
