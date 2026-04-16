/**
 * Shared utilities for recipes — safe to import from both client and server components.
 * No fs/path/yaml imports here.
 */

export function recipeHref(recipe) {
  const provider = recipe.meta.provider.toLowerCase().replace(/\s+/g, "-");
  return `/${provider}/${recipe.meta.slug}`;
}
