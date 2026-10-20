// `meta.maintained_by`: who maintains a recipe — a `taxonomy.yaml → maintainers`
// id, or a hardware-keyed map of ids in the variant-`tp` keyspace (exact GPU id
// > generation > brand > `default`), so one organization can maintain the
// recipe and another its configurations for some hardware:
//   maintained_by: { default: inferact, amd: amd }

import { hardwareKeyedValue } from "./command-synthesis.js";

// The map form (a bare id is `{ default: id }`), or null.
export function maintainedByMap(recipe) {
  const m = recipe?.meta?.maintained_by;
  if (!m) return null;
  return typeof m === "string" ? { default: m } : m;
}

function describe(id, taxonomy) {
  const m = taxonomy?.maintainers?.[id];
  return { id, name: m?.name || id, logo: m?.logo || null, url: m?.url || null };
}

// The organization maintaining the configuration for this hardware, or null.
export function maintainerFor(recipe, hwProfile, hwId, taxonomy) {
  const id = hardwareKeyedValue(maintainedByMap(recipe), hwProfile, hwId);
  return id ? describe(id, taxonomy) : null;
}

// Every maintainer with the hardware it covers, for the recipe header:
// [{ id, name, logo, url, scopes: [label] }] — the `default` maintainer first,
// with no scopes. Scope labels: a profile's name, a generation, or a brand.
export function recipeMaintainers(recipe, taxonomy) {
  const map = maintainedByMap(recipe);
  if (!map) return [];
  const profiles = taxonomy?.hardware_profiles || {};
  const brandName = Object.fromEntries(Object.values(profiles).filter((p) => p?.brand).map((p) => [p.brand.toLowerCase(), p.brand]));
  const label = (key) => profiles[key]?.display_name
    || (brandName[key] ? `${brandName[key]} hardware` : key.charAt(0).toUpperCase() + key.slice(1));
  const out = new Map();
  for (const [key, id] of Object.entries(map).sort(([a], [b]) => (a === "default" ? -1 : b === "default" ? 1 : 0))) {
    if (!out.has(id)) out.set(id, { ...describe(id, taxonomy), scopes: [] });
    if (key !== "default") out.get(id).scopes.push(label(key));
  }
  return [...out.values()];
}
