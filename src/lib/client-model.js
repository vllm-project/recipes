// Resolve the public API model name from emitted arguments, never draft JSON.
export function clientModelFor(recipe, variantKey, rendering = {}) {
  const argv = rendering.argv || rendering.headArgv || rendering.head_argv
    || rendering.prefill?.argv || rendering.vllm?.argv || rendering.vllm_argv || [];
  const index = argv.lastIndexOf("--served-model-name");
  if (index >= 0 && argv[index + 1]) return argv[index + 1];
  const inline = argv.findLast((arg) => arg.startsWith("--served-model-name="));
  if (inline) return inline.slice("--served-model-name=".length);
  return recipe.variants?.[variantKey]?.model_id || recipe.model?.model_id || recipe.hf_id;
}

export function fillClientModel(text, model) {
  return text.replaceAll("{{client_model}}", JSON.stringify(model));
}
