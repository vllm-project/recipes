/**
 * Provider metadata keyed by HuggingFace org.
 *
 * logo: path to local avatar served from /public/providers/.
 * Logos are downloaded at build time by scripts/fetch-provider-logos.mjs
 * (fetched from HuggingFace) — so the file URL always comes from our own
 * domain, making it globally accessible even where HF CDN is blocked.
 *
 * Extensions (png vs jpeg) match what HF serves for each org.
 */
export const PROVIDERS = {
  "deepseek-ai":     { display_name: "DeepSeek",                logo: "/providers/deepseek-ai.png" },
  "Qwen":            { display_name: "Qwen",                    logo: "/providers/Qwen.png" },
  "moonshotai":      { display_name: "Moonshot AI",             logo: "/providers/moonshotai.jpeg" },
  "meta-llama":      { display_name: "Meta",                    logo: "/providers/meta-llama.png" },
  "Google":          { display_name: "Google",                  logo: "/providers/Google.png" },
  "google":          { display_name: "Google",                  logo: "/providers/Google.png" },
  "microsoft":       { display_name: "Microsoft",               logo: "/providers/microsoft.png" },
  "mistralai":       { display_name: "Mistral AI",              logo: "/providers/mistralai.png" },
  "nvidia":          { display_name: "NVIDIA",                  logo: "/providers/nvidia.png" },
  "openai":          { display_name: "OpenAI",                  logo: "/providers/openai.png", invertInDark: true },
  "MiniMaxAI":       { display_name: "MiniMax",                 logo: "/providers/MiniMaxAI.jpeg" },
  "zai-org":         { display_name: "GLM (Z-AI)",              logo: "/providers/zai-org.png" },
  "baidu":           { display_name: "Ernie (Baidu)",           logo: "/providers/baidu.png" },
  "arcee-ai":        { display_name: "Arcee AI",                logo: "/providers/arcee-ai.png" },
  "inclusionAI":     { display_name: "inclusionAI",             logo: "/providers/inclusionAI.jpeg" },
  "OpenGVLab":       { display_name: "InternVL (OpenGVLab)",    logo: "/providers/OpenGVLab.jpeg" },
  "internlm":        { display_name: "InternLM",                logo: "/providers/internlm.png" },
  "jinaai":          { display_name: "Jina AI",                 logo: "/providers/jinaai.png" },
  "PaddlePaddle":    { display_name: "PaddlePaddle",            logo: "/providers/PaddlePaddle.png" },
  "ByteDance-Seed":  { display_name: "Seed (ByteDance)",        logo: "/providers/ByteDance-Seed.png" },
  "tencent":         { display_name: "Hunyuan (Tencent)",       logo: "/providers/tencent.png" },
  "XiaomiMiMo":      { display_name: "MiMo (Xiaomi)",            logo: "/providers/XiaomiMiMo.jpeg" },
  "Wan-AI":          { display_name: "Wan (Alibaba)",            logo: "/providers/Wan-AI.png" },
  "meituan-longcat": { display_name: "LongCat (Meituan)",        logo: "/providers/meituan-longcat.png" },
  "stabilityai":     { display_name: "Stability AI",             logo: "/providers/stabilityai.png" },
  "stepfun-ai":      { display_name: "StepFun",                  logo: "/providers/stepfun-ai.png" },
  "poolside":        { display_name: "Poolside",                 logo: "/providers/poolside.png" },
};

export function getProviderLogo(hfOrg) {
  return PROVIDERS[hfOrg]?.logo || null;
}

// Monochrome logos (e.g. OpenAI's pure-black mark) disappear on dark backgrounds.
// Flip them to white via `dark:invert` for those providers.
export function getProviderLogoClass(hfOrg) {
  return PROVIDERS[hfOrg]?.invertInDark ? "dark:invert" : "";
}

export function getProviderDisplayName(hfOrg) {
  return PROVIDERS[hfOrg]?.display_name || hfOrg;
}
