import fs from "fs";
import path from "path";
import yaml from "js-yaml";

let cache = null;

const MODELS_DIR = path.join(process.cwd(), "models");
const HF_DATES_PATH = path.join(process.cwd(), "public", "hf-dates.json");

// HF release dates per hf_id (populated at build by scripts/fetch-hf-dates.mjs)
let hfDates = null;
function loadHfDates() {
  if (hfDates !== null) return hfDates;
  try {
    hfDates = JSON.parse(fs.readFileSync(HF_DATES_PATH, "utf8"));
  } catch {
    hfDates = {};
  }
  return hfDates;
}

function parseRecipe(filePath) {
  const raw = yaml.load(fs.readFileSync(filePath, "utf8"));
  // js-yaml auto-parses YYYY-MM-DD into Date objects — normalize to string
  if (raw.meta?.date_updated instanceof Date) {
    raw.meta.date_updated = raw.meta.date_updated.toISOString().split("T")[0];
  }
  // Derive HF identity from file path: models/<org>/<repo>.yaml
  const rel = path.relative(MODELS_DIR, filePath);
  const parts = rel.split(path.sep);
  if (parts.length >= 2) {
    raw.hf_org = parts[0];
    raw.hf_repo = parts[parts.length - 1].replace(/\.(yaml|yml)$/, "");
    raw.hf_id = `${raw.hf_org}/${raw.hf_repo}`;
    // Attach HF release date (ISO string) from build-time manifest
    const dates = loadHfDates();
    raw.hf_released = dates[raw.hf_id] || null;
  }
  return raw;
}

function findYamlFiles(dir) {
  const results = [];
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      results.push(...findYamlFiles(full));
    } else if (entry.name.endsWith(".yaml") || entry.name.endsWith(".yml")) {
      results.push(full);
    }
  }
  return results;
}

export function getAllRecipes() {
  if (cache) return cache;
  const files = findYamlFiles(MODELS_DIR);
  cache = files
    .map(parseRecipe)
    .sort((a, b) => {
      const da = new Date(a.meta.date_updated);
      const db = new Date(b.meta.date_updated);
      return db - da;
    });
  return cache;
}

export function getRecipeByHfId(org, repo) {
  const all = getAllRecipes();
  return all.find((r) => r.hf_org === org && r.hf_repo === repo) || null;
}

/**
 * If `<org>/<repo>` is the `model_id` of some variant of another recipe,
 * return the parent recipe + variant key so the route can redirect to
 * `/<parent.hf_org>/<parent.hf_repo>?variant=<key>`. Otherwise null.
 *
 * Skips the `default` variant and any variant whose model_id equals the
 * parent's own HF id (those are just the base recipe).
 */
export function findVariantRedirect(org, repo) {
  const target = `${org}/${repo}`;
  for (const r of getAllRecipes()) {
    if (r.hf_id === target) return null;
    const variants = r.variants || {};
    for (const [key, v] of Object.entries(variants)) {
      if (key === "default") continue;
      if (v?.model_id && v.model_id === target) {
        return { parent: r, variantKey: key };
      }
    }
  }
  return null;
}

/**
 * All `<org>/<repo>` pairs that should route to a recipe page — base
 * recipes plus the HF ids of each non-default variant. Used by
 * `generateStaticParams` so variant ids are prerendered and emit the
 * redirect response instead of 404.
 */
export function getAllRoutablePairs() {
  const pairs = [];
  const seen = new Set();
  const push = (org, repo) => {
    const k = `${org}/${repo}`;
    if (seen.has(k)) return;
    seen.add(k);
    pairs.push({ org, repo });
  };
  for (const r of getAllRecipes()) {
    push(r.hf_org, r.hf_repo);
    for (const [key, v] of Object.entries(r.variants || {})) {
      if (key === "default") continue;
      if (!v?.model_id || v.model_id === r.hf_id) continue;
      const [vo, ...rest] = v.model_id.split("/");
      if (!vo || rest.length === 0) continue;
      push(vo, rest.join("/"));
    }
  }
  return pairs;
}
