/**
 * Shared utilities — safe to import from client components (no fs/path/yaml).
 */

// Prefer specialized capabilities over text, regardless of YAML tag order.
const PRIMARY_TASK_PRIORITY = ["omni", "embedding", "multimodal", "text"];

export function primaryRecipeTask(recipe) {
  const tasks = recipe.meta?.tasks || [];
  return PRIMARY_TASK_PRIORITY.find((task) => tasks.includes(task)) || tasks[0] || "other";
}

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

/*
 * Client-component projections of a recipe.
 *
 * Every prop a server component hands a client component is serialized into
 * the page's RSC payload, which ships inline in the HTML *and* again in every
 * `<Link>` prefetch of a page sharing that layout. Passing whole recipes (guide,
 * variants, presets, …) to the sidebar and the homepage grid made each page
 * ~2.7 MB raw / ~500 KB compressed. Pass these projections instead, and add a
 * field here when a client component starts reading it — a field left out
 * reads as `undefined` without any error.
 */

/** What `ModelSidebar` reads: link target, label, task icon. */
export function toSidebarEntry(r) {
  return {
    hf_id: r.hf_id,
    hf_org: r.hf_org,
    hf_repo: r.hf_repo,
    meta: { title: r.meta.title, provider: r.meta.provider, tasks: r.meta.tasks },
  };
}

/** What `RecipeCardGrid` reads: card text, badges and "latest" ordering. */
export function toCardEntry(r) {
  return {
    hf_id: r.hf_id,
    hf_org: r.hf_org,
    hf_repo: r.hf_repo,
    meta: {
      title: r.meta.title,
      provider: r.meta.provider,
      description: r.meta.description,
      performance_headline: r.meta.performance_headline,
      date_added: r.meta.date_added,
      date_updated: r.meta.date_updated,
      tasks: r.meta.tasks,
    },
    model: {
      architecture: r.model?.architecture,
      parameter_count: r.model?.parameter_count,
      active_parameters: r.model?.active_parameters,
      context_length: r.model?.context_length,
    },
    variants: { default: { precision: r.variants?.default?.precision } },
  };
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
