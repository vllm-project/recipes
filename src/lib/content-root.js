import fs from "fs";
import path from "path";
import yaml from "js-yaml";

// Where recipe content is read from. Recipe YAMLs live in their owning repos
// (vllm, vllm-omni, vllm-ascend, …) and are pinned by `sources.yaml` +
// `sources.lock.json`; `scripts/assemble-content.mjs` merges them into
// `.content/<channel>/` with the same layout the repo root used to have
// (models/, strategies/, kv_store/, taxonomy.yaml, platforms.yaml), plus a
// `provenance.json` mapping each hf_id to the repo/path it came from.
//
// Precedence: RECIPES_CONTENT_DIR > .content/<channel> > repo root (the
// pre-assembly layout, so a fresh checkout still renders).

const ROOT = process.cwd();
const SOURCES_PATH = path.resolve(ROOT, process.env.RECIPES_SOURCES || "sources.yaml");

let sourcesCache = null;

export function loadSourcesConfig() {
  if (sourcesCache) return sourcesCache;
  sourcesCache = fs.existsSync(SOURCES_PATH)
    ? yaml.load(fs.readFileSync(SOURCES_PATH, "utf8")) || {}
    : {};
  return sourcesCache;
}

export function defaultChannel() {
  return loadSourcesConfig().default_channel || "latest";
}

export function contentChannel() {
  return process.env.RECIPES_CHANNEL || defaultChannel();
}

export function assembledDir(channel = contentChannel()) {
  return path.join(ROOT, ".content", channel);
}

let warnedFallback = false;

export function contentDir(channel = contentChannel()) {
  if (process.env.RECIPES_CONTENT_DIR) return path.resolve(process.env.RECIPES_CONTENT_DIR);
  const dir = assembledDir(channel);
  if (fs.existsSync(path.join(dir, "models"))) return dir;
  // Un-assembled checkout: only this repo's own models/ is visible, which is a
  // partial catalog once any recipe lives in another source repo.
  const sources = loadSourcesConfig().channels?.[channel] || {};
  if (!warnedFallback && Object.values(sources).some((s) => !s?.local)) {
    warnedFallback = true;
    console.warn(`[recipes] .content/${channel} is not assembled — showing only this repo's models/. Run \`pnpm content\` (pnpm dev and the build do it).`);
  }
  return ROOT;
}

const provenanceCache = new Map();

// hf_id → { source, repo, ref, sha, path, edit_url, overlays[] }. Empty when
// reading the un-assembled repo root.
export function loadProvenance(channel = contentChannel()) {
  const dir = contentDir(channel);
  const file = path.join(dir, "provenance.json");
  // `next dev` re-reads after a re-assemble; production reads once.
  let mtime = 0;
  if (process.env.NODE_ENV === "development") {
    try { mtime = fs.statSync(file).mtimeMs; } catch { mtime = 0; }
  }
  const hit = provenanceCache.get(dir);
  if (hit && hit.mtime === mtime) return hit.data;
  let data = {};
  try {
    data = JSON.parse(fs.readFileSync(file, "utf8")).recipes || {};
  } catch {
    data = {};
  }
  provenanceCache.set(dir, { mtime, data });
  return data;
}
