export function checkpointLink(recipe, variant, taxonomy) {
  const modelId = variant?.model_id || recipe?.model?.model_id;
  if (!modelId) return null;

  const hub = variant?.hub || "huggingface";
  const config = taxonomy?.model_hubs?.[hub];
  if (!config) return null;
  return { modelId, href: config.url.replace("{model_id}", modelId), provider: config.label, hub };
}

// A ModelScope selection is process-wide on existing vLLM builds. Warn when
// recipe arguments may independently name a draft or tokenizer repository.
export function hasExternalModelReference(recipe) {
  const visit = (value) => {
    if (Array.isArray(value)) {
      if (value.some((item) => typeof item === "string" && /^--tokenizer(?:=|$)/.test(item))) return true;
      for (let i = 0; i < value.length; i++) {
        const item = value[i];
        const config = item === "--speculative-config" ? value[i + 1]
          : typeof item === "string" && item.startsWith("--speculative-config=") ? item.slice("--speculative-config=".length) : null;
        if (typeof config === "string") {
          try { if (JSON.parse(config).model) return true; } catch { /* Validation is handled elsewhere. */ }
        }
      }
      return value.some(visit);
    }
    return value && typeof value === "object" && Object.values(value).some(visit);
  };
  return !!visit(recipe);
}

export function hasManualModelScopeEnv(value) {
  if (!value || typeof value !== "object") return false;
  return Object.hasOwn(value, "VLLM_USE_MODELSCOPE") || Object.values(value).some(hasManualModelScopeEnv);
}
