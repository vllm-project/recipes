export function checkpointLink(recipe, variant) {
  const modelId = variant?.model_id || recipe?.model?.model_id;
  if (!modelId) return null;

  const fallback = `https://huggingface.co/${modelId}`;
  let href = variant?.model_url || fallback;
  let host;
  try {
    const url = new URL(href);
    if (url.protocol !== "https:") throw new Error("HTTPS required");
    host = url.hostname;
  } catch {
    href = fallback;
    host = "huggingface.co";
  }
  const provider = host === "modelscope.cn" || host.endsWith(".modelscope.cn")
    ? "ModelScope"
    : host === "huggingface.co" ? "Hugging Face" : "Checkpoint";
  return { modelId, href, provider };
}
