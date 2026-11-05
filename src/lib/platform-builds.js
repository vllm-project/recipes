// Platform builds describe exact custom images, not upstream version floors.
export function platformBuildLabels(model, hardwareProfiles = {}) {
  const groups = new Map();
  for (const [id, build] of Object.entries(model?.platform_builds || {})) {
    const profile = hardwareProfiles[id];
    if (!profile || !build?.base_version || build.build_type !== "custom") continue;
    const brand = profile.brand || "";
    const key = JSON.stringify([brand, build.base_version, build.build_type]);
    if (!groups.has(key)) groups.set(key, { brand, version: build.base_version, hardware: [] });
    groups.get(key).hardware.push(profile.display_name || id);
  }
  return [...groups.values()].map(({ brand, version, hardware }) =>
    `${brand} ${hardware.join(" / ")} · vLLM ${version}-based custom build`.trim());
}

export function selectedVersion(model, hardware, profiles = {}) {
  const build = model?.platform_builds?.[hardware];
  if (build?.base_version && build.build_type === "custom") {
    const brand = profiles[hardware]?.brand;
    if (brand === "Zhenwu") return { custom: true, label: `vLLM ${build.base_version}+` };
    return { custom: true, label: `vLLM ${build.base_version} · ${brand ? `${brand} ` : ""}custom build` };
  }
  return { custom: false, label: `vLLM ${model?.min_vllm_version}+` };
}
