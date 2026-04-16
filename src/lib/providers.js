/**
 * Provider metadata: logos, display names.
 * Logo URLs extracted from README.md provider headings.
 */
export const PROVIDERS = {
  "DeepSeek": {
    logo: "https://avatars.githubusercontent.com/u/148330874?s=200&v=4",
  },
  "Qwen": {
    logo: "https://qwenlm.github.io/favicon.png",
  },
  "Moonshot AI": {
    logo: "https://avatars.githubusercontent.com/u/129152888?v=4",
  },
  "Meta": {
    logo: "https://avatars.githubusercontent.com/u/69631?v=4",
  },
  "Google": {
    logo: "https://avatars.githubusercontent.com/u/1342004?v=4",
  },
  "Microsoft": {
    logo: "https://avatars.githubusercontent.com/u/6154722?s=48&v=4",
  },
  "Mistral AI": {
    logo: "https://avatars.githubusercontent.com/u/132372032?v=4",
  },
  "NVIDIA": {
    logo: "https://avatars.githubusercontent.com/u/1728152?v=4",
  },
  "OpenAI": {
    logo: "https://avatars.githubusercontent.com/u/14957082?v=4",
  },
  "MiniMax": {
    logo: "https://github.com/MiniMax-AI/MiniMax-01/raw/main/figures/minimax.svg",
  },
  "GLM": {
    logo: "https://raw.githubusercontent.com/zai-org/GLM-4.5/refs/heads/main/resources/logo.svg",
  },
  "Ernie": {
    logo: "https://avatars.githubusercontent.com/u/13245940?v=4",
  },
  "Arcee AI": {
    logo: "https://cdn-avatars.huggingface.co/v1/production/uploads/6435718aaaef013d1aec3b8b/GZPnGkfMn8Ino6JbkL4fJ.png",
  },
  "inclusionAI": {
    logo: "https://avatars.githubusercontent.com/u/199075982?s=200&v=4",
  },
  "InternVL": {
    logo: "https://avatars.githubusercontent.com/u/135356492?s=200&v=4",
  },
  "InternLM": {
    logo: "https://avatars.githubusercontent.com/u/135356492?s=200&v=4",
  },
  "Jina AI": {
    logo: "https://avatars.githubusercontent.com/u/60539444?s=200&v=4",
  },
  "PaddlePaddle": {
    logo: "https://avatars.githubusercontent.com/u/23534030?v=4",
  },
  "Seed": {
    logo: "https://avatars.githubusercontent.com/u/4158466?s=200&v=4",
  },
  "Tencent-Hunyuan": {
    logo: "https://avatars.githubusercontent.com/u/210980732?s=200&v=4",
  },
  "Xiaomi MiMo": {
    logo: "https://avatars.githubusercontent.com/u/208276378?v=4",
  },
};

export function getProviderLogo(name) {
  return PROVIDERS[name]?.logo || null;
}
