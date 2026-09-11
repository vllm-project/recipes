// InferenceX (inferencex.semianalysis.com) embed helpers.
//
// A recipe opts in with `meta.inferencex: { model: <inferencex-model-slug> }`.
// The slug is the model's path on InferenceX (`/inference/<slug>`), which is not
// derivable from the HF id (DeepSeek-V4-Pro → `deepseek-v4`), so it is
// declared per recipe. The embed is locked to vLLM: this is the vLLM recipes
// site, so the chart only shows vLLM configurations regardless of what other
// engines InferenceX benchmarks.

export const INFERENCEX_BASE_URL = (
  process.env.NEXT_PUBLIC_INFERENCEX_URL || "https://inferencex.semianalysis.com"
).replace(/\/$/, "");

export const INFERENCEX_FRAMEWORK = "vllm";

// InferenceX ships a skin that re-tokens the embed with this site's palette
// and fonts (`theme=vllm-light` / `theme=vllm-dark`), so the chart reads as
// part of the page rather than an InferenceX screenshot.
export const INFERENCEX_SKIN = "vllm";

/** Normalize `meta.inferencex` (string shorthand or object) to `{ model, scenario? }`. */
export function resolveInferencex(meta) {
  const raw = meta?.inferencex;
  if (!raw) return null;
  const cfg = typeof raw === "string" ? { model: raw } : raw;
  if (!cfg.model || typeof cfg.model !== "string") return null;
  return { model: cfg.model, scenario: cfg.scenario || null };
}

/** Iframe src for the embedded chart. `theme` is "light" | "dark". */
export function inferencexEmbedUrl({ model, scenario }, theme = "light") {
  const params = new URLSearchParams({
    framework: INFERENCEX_FRAMEWORK,
    theme: `${INFERENCEX_SKIN}-${theme === "dark" ? "dark" : "light"}`,
  });
  if (scenario) params.set("scenario", scenario);
  return `${INFERENCEX_BASE_URL}/embed/model/${encodeURIComponent(model)}?${params}`;
}

/** Link to the full, interactive InferenceX dashboard for the model. */
export function inferencexModelUrl({ model }) {
  return `${INFERENCEX_BASE_URL}/inference/${encodeURIComponent(model)}`;
}

/** InferenceX AgentX page (the agentic-coding workload the embedded chart is measured on). */
export function inferencexAgentxUrl() {
  return `${INFERENCEX_BASE_URL}/agentx`;
}
