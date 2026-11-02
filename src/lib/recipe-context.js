// Configuration-scoped guide entries: `notes` and `troubleshooting` items that
// carry a `when` condition, so the page below the command builder can show
// what applies to the command the reader just built (and fold the rest away).
//
// when: { hardware: [hwKey], strategies: [id], variants: [key], features: [key] }
//   - keys AND together, values within a key OR together; absent = any.
//   - hardware values use the hardware-map keyspace: exact GPU id, generation,
//     lowercase brand, or `arch:<compute_arch>`.
//   - features matches while one of those features is on.

export const CONTEXT_KINDS = ["notes", "troubleshooting"];
const WHEN_KEYS = ["hardware", "strategies", "variants", "features"];

export function hardwareKeyMatches(key, hwId, hp) {
  if (!hwId) return false;
  if (key === hwId) return true;
  if (!hp) return false;
  if (key === hp.generation) return true;
  if (hp.brand && key === hp.brand.toLowerCase()) return true;
  return !!hp.compute_arch && key === `arch:${hp.compute_arch}`;
}

// sel: { hwId, strategy, variant, features: [key] }
export function whenMatches(when, sel, hardwareProfiles = {}) {
  if (!when) return true;
  if (!sel) return false;
  const hp = hardwareProfiles[sel.hwId];
  if (when.hardware?.length && !when.hardware.some((k) => hardwareKeyMatches(k, sel.hwId, hp))) return false;
  if (when.strategies?.length && !when.strategies.includes(sel.strategy)) return false;
  if (when.variants?.length && !when.variants.includes(sel.variant)) return false;
  if (when.features?.length && !when.features.some((f) => (sel.features || []).includes(f))) return false;
  return true;
}

// Short labels for a `when`, e.g. ["Hopper", "Blackwell", "Single-Node TP"].
export function whenLabels(when, { hardwareProfiles = {}, strategies = {}, recipe } = {}) {
  if (!when) return [];
  const hwLabel = (k) => {
    if (hardwareProfiles[k]) return hardwareProfiles[k].display_name || k;
    if (k.startsWith("arch:")) return k.slice(5);
    const brand = Object.values(hardwareProfiles).find((h) => h.brand?.toLowerCase() === k)?.brand;
    if (brand) return brand;
    return k.charAt(0).toUpperCase() + k.slice(1);
  };
  return [
    ...(when.hardware || []).map(hwLabel),
    ...(when.strategies || []).map((s) => strategies[s]?.display_name || s),
    ...(when.variants || []).map((v) => recipe?.variants?.[v]?.label || (recipe?.variants?.[v]?.precision || v).toUpperCase()),
    ...(when.features || []).map((f) => recipe?.features?.[f]?.label || f.replace(/_/g, " ").replace(/^./, (c) => c.toUpperCase())),
  ];
}

// Cross-field checks the JSON Schema can't express: referenced variants and
// features must exist. Returns a list of error strings.
export function validateContextEntries(recipe) {
  const errors = [];
  for (const kind of CONTEXT_KINDS) {
    (recipe[kind] || []).forEach((entry, i) => {
      const at = `${kind}[${i}]`;
      for (const k of Object.keys(entry.when || {})) {
        if (!WHEN_KEYS.includes(k)) errors.push(`${at}.when.${k}: unknown condition`);
      }
      for (const v of entry.when?.variants || []) {
        if (!recipe.variants?.[v]) errors.push(`${at}.when.variants: no variant "${v}"`);
      }
      for (const f of entry.when?.features || []) {
        if (!recipe.features?.[f]) errors.push(`${at}.when.features: no feature "${f}"`);
      }
    });
  }
  return errors;
}
