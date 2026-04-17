/**
 * Provider metadata keyed by HuggingFace org.
 * display_name: how to render the provider in the UI.
 * logo: avatar URL (usually from HF or GitHub).
 *
 * HF org names are case-sensitive (HF doesn't do canonical casing).
 */
export const PROVIDERS = {
  "deepseek-ai": {
    display_name: "DeepSeek",
    logo: "https://avatars.githubusercontent.com/u/148330874?s=200&v=4",
  },
  "Qwen": {
    display_name: "Qwen",
    logo: "https://qwenlm.github.io/favicon.png",
  },
  "moonshotai": {
    display_name: "Moonshot AI",
    logo: "https://avatars.githubusercontent.com/u/129152888?v=4",
  },
  "meta-llama": {
    display_name: "Meta",
    logo: "https://avatars.githubusercontent.com/u/69631?v=4",
  },
  "google": {
    display_name: "Google",
    logo: "https://avatars.githubusercontent.com/u/1342004?v=4",
  },
  "microsoft": {
    display_name: "Microsoft",
    logo: "https://avatars.githubusercontent.com/u/6154722?s=48&v=4",
  },
  "mistralai": {
    display_name: "Mistral AI",
    logo: "https://avatars.githubusercontent.com/u/132372032?v=4",
  },
  "nvidia": {
    display_name: "NVIDIA",
    logo: "https://avatars.githubusercontent.com/u/1728152?v=4",
  },
  "openai": {
    display_name: "OpenAI",
    logo: "https://avatars.githubusercontent.com/u/14957082?v=4",
  },
  "MiniMaxAI": {
    display_name: "MiniMax",
    logo: "https://github.com/MiniMax-AI/MiniMax-01/raw/main/figures/minimax.svg",
  },
  "zai-org": {
    display_name: "GLM (Z-AI)",
    logo: "https://raw.githubusercontent.com/zai-org/GLM-4.5/refs/heads/main/resources/logo.svg",
  },
  "baidu": {
    display_name: "Ernie (Baidu)",
    logo: "https://avatars.githubusercontent.com/u/13245940?v=4",
  },
  "arcee-ai": {
    display_name: "Arcee AI",
    logo: "https://cdn-avatars.huggingface.co/v1/production/uploads/6435718aaaef013d1aec3b8b/GZPnGkfMn8Ino6JbkL4fJ.png",
  },
  "inclusionAI": {
    display_name: "inclusionAI",
    logo: "https://avatars.githubusercontent.com/u/199075982?s=200&v=4",
  },
  "OpenGVLab": {
    display_name: "InternVL (OpenGVLab)",
    logo: "https://avatars.githubusercontent.com/u/135356492?s=200&v=4",
  },
  "internlm": {
    display_name: "InternLM",
    logo: "https://avatars.githubusercontent.com/u/135356492?s=200&v=4",
  },
  "jinaai": {
    display_name: "Jina AI",
    logo: "https://avatars.githubusercontent.com/u/60539444?s=200&v=4",
  },
  "PaddlePaddle": {
    display_name: "PaddlePaddle",
    logo: "https://avatars.githubusercontent.com/u/23534030?v=4",
  },
  "ByteDance-Seed": {
    display_name: "Seed (ByteDance)",
    logo: "https://avatars.githubusercontent.com/u/4158466?s=200&v=4",
  },
  "tencent": {
    display_name: "Tencent Hunyuan",
    logo: "https://avatars.githubusercontent.com/u/210980732?s=200&v=4",
  },
  "XiaomiMiMo": {
    display_name: "Xiaomi MiMo",
    logo: "https://avatars.githubusercontent.com/u/208276378?v=4",
  },
};

export function getProviderLogo(hfOrg) {
  return PROVIDERS[hfOrg]?.logo || null;
}

export function getProviderDisplayName(hfOrg) {
  return PROVIDERS[hfOrg]?.display_name || hfOrg;
}
