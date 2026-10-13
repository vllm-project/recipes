import fs from "fs";
import path from "path";
import yaml from "js-yaml";
import { contentDir, loadProvenance } from "./content-root.js";

let cache = null;

// Resolved per call, not at import: the content tree may be assembled after
// this module loads (pnpm dev assembles first, but a re-assemble can follow).
const modelsDir = () => path.join(contentDir(), "models");
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

function parseRecipe(filePath, dir, provenance) {
  const raw = yaml.load(fs.readFileSync(filePath, "utf8"));
  // js-yaml auto-parses YYYY-MM-DD into Date objects — normalize to string
  for (const field of ["date_added", "date_updated"]) {
    if (raw.meta?.[field] instanceof Date) {
      raw.meta[field] = raw.meta[field].toISOString().split("T")[0];
    }
  }
  // Derive HF identity from file path: models/<org>/<repo>.yaml
  const rel = path.relative(dir, filePath);
  const parts = rel.split(path.sep);
  if (parts.length >= 2) {
    raw.hf_org = parts[0];
    raw.hf_repo = parts[parts.length - 1].replace(/\.(yaml|yml)$/, "");
    raw.hf_id = `${raw.hf_org}/${raw.hf_repo}`;
    // Attach HF release date (ISO string) from build-time manifest
    const dates = loadHfDates();
    raw.hf_released = dates[raw.hf_id] || null;
    // Owning repo + overlays (from .content/<channel>/provenance.json)
    raw.source = provenance[raw.hf_id] || null;
  }
  return raw;
}

// Skips dangling symlinks (a source checkout renamed or deleted a recipe) and
// treats a missing tree as empty (the brief swap while re-assembling).
function findYamlFiles(dir) {
  const results = [];
  if (!fs.existsSync(dir)) return results;
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      results.push(...findYamlFiles(full));
    } else if ((entry.name.endsWith(".yaml") || entry.name.endsWith(".yml")) && fs.existsSync(full)) {
      results.push(full);
    }
  }
  return results;
}

const mtime = (f) => {
  try { return fs.statSync(f).mtimeMs; } catch { return "missing"; }
};

// In `next dev`, re-parse when any recipe file or provenance.json changed (the
// content tree symlinks into source checkouts such as ~/vllm/recipes), so
// edits show on refresh. Production builds parse once.
let cacheKey = null;

export function getAllRecipes() {
  const dev = process.env.NODE_ENV === "development";
  if (cache && !dev) return cache;
  const dir = modelsDir();
  const files = findYamlFiles(dir);
  const key = dev
    ? [mtime(path.join(contentDir(), "provenance.json")), ...files.map((f) => `${f}:${mtime(f)}`)].join("|")
    : null;
  if (cache && key === cacheKey) return cache;
  const provenance = loadProvenance();
  cacheKey = key;
  cache = files
    .map((f) => parseRecipe(f, dir, provenance))
    .sort((a, b) => {
      const da = new Date(a.meta.date_updated);
      const db = new Date(b.meta.date_updated);
      return db - da;
    });
  return cache;
}

export function getRecipeByHfId(org, repo) {
  const o = org.toLowerCase(), r = repo.toLowerCase();
  return getAllRecipes().find((x) => x.hf_org.toLowerCase() === o && x.hf_repo.toLowerCase() === r) || null;
}

export function getRecipesByOrg(org) {
  const o = org.toLowerCase();
  return getAllRecipes().filter((r) => r.hf_org.toLowerCase() === o);
}

/**
 * If `<org>/<repo>` is the `model_id` of some variant of another recipe,
 * return the parent recipe + variant key so the route can redirect to
 * `/<parent.hf_org>/<parent.hf_repo>?variant=<key>`. Otherwise null.
 *
 * Skips any variant whose model_id equals the parent's own HF id (that's
 * just the base recipe). The `default` variant is NOT skipped: a recipe may
 * point its default at a differently-named checkpoint (DeepSeek-V4-Flash
 * defaults to `-0731`, GLM-5.2 to `-FP8`), and that HF id has to resolve
 * rather than 404.
 */
export function findVariantRedirect(org, repo) {
  const target = `${org}/${repo}`.toLowerCase();
  for (const r of getAllRecipes()) {
    if (r.hf_id.toLowerCase() === target) return null;
    const variants = r.variants || {};
    for (const [key, v] of Object.entries(variants)) {
      if (!v?.model_id || v.model_id === r.hf_id) continue;
      if (v.model_id.toLowerCase() === target) {
        return { parent: r, variantKey: key };
      }
    }
  }
  return null;
}

/**
 * All `<org>/<repo>` pairs that should route to a recipe page — base
 * recipes plus the HF id of every variant that names its own checkpoint
 * (`default` included — see findVariantRedirect). Used by
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
    for (const [, v] of Object.entries(r.variants || {})) {
      if (!v?.model_id || v.model_id === r.hf_id) continue;
      const [vo, ...rest] = v.model_id.split("/");
      if (!vo || rest.length === 0) continue;
      push(vo, rest.join("/"));
    }
  }
  return pairs;
}
