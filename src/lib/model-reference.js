// A variant with a different checkpoint must not inherit the base weights URL.
export function resolveModelReference(recipe, variantKey) {
  const base = recipe.model || {};
  const variant = recipe.variants?.[variantKey] || recipe.variants?.default || {};
  const modelId = variant.model_id || base.model_id;
  if (!modelId) return null;
  const explicit = variant.model_url ?? (modelId === base.model_id ? base.model_url : undefined);
  const url = new URL(explicit ?? `https://huggingface.co/${modelId}`);
  if (!["http:", "https:"].includes(url.protocol)) throw new Error("model_url must be an absolute HTTP(S) URL");
  const host = url.hostname.toLowerCase();
  const provider = ["huggingface.co", "www.huggingface.co"].includes(host) ? "Hugging Face"
    : ["modelscope.cn", "www.modelscope.cn"].includes(host) ? "ModelScope" : host;
  return { model_id: modelId, model_url: url.href, provider };
}

export function validateModelReferences(recipe) {
  const errors = [];
  for (const [field, entry] of [["model", recipe.model],
    ...Object.entries(recipe.variants || {}).map(([key, value]) => [`variants.${key}`, value])]) {
    if (entry?.model_url == null) continue;
    try {
      const url = new URL(entry.model_url);
      if (!["http:", "https:"].includes(url.protocol)) throw new Error();
    } catch {
      errors.push(`${field}.model_url must be an absolute HTTP(S) URL`);
    }
  }
  return errors;
}
