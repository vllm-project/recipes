// References are declared by individual presets and their points. Keep this
// traversal common to source ownership checks and recipe loading.
export function poolProfileRefs(recipe) {
  const refs = [];
  for (const [key, preset] of Object.entries(recipe?.presets || {})) {
    const at = `presets.${key}`;
    const collect = (selection, where) => {
      const id = selection?.kv_offload;
      if (typeof id === "string" && id.includes("/")) refs.push({ id, where, hardware: preset.hardware });
    };
    collect(preset?.select, `${at}.select.kv_offload`);
    for (const [point, value] of Object.entries(preset?.points || {})) {
      collect(value?.select, `${at}.points.${point}.select.kv_offload`);
    }
  }
  return refs;
}
