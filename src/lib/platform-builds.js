// Exact hardware overrides the platform brand; values describe custom builds.
export function selectedVersion(model, hardware, profiles = {}) {
  const brand = profiles[hardware]?.brand;
  const build = model?.platform_builds?.[hardware] ?? model?.platform_builds?.[brand?.toLowerCase()];
  if (build) {
    if (brand === "Zhenwu") return { custom: true, label: `vLLM ${build}+` };
    return { custom: true, label: `vLLM ${build} · ${brand ? `${brand} ` : ""}custom build` };
  }
  return { custom: false, label: `vLLM ${model?.min_vllm_version}+` };
}
