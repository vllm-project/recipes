import fs from "fs";
import path from "path";
import yaml from "js-yaml";

let cache = null;

const PLATFORMS_PATH = path.join(process.cwd(), "platforms.yaml");

export function loadPlatforms() {
  if (cache) return cache;
  const parsed = yaml.load(fs.readFileSync(PLATFORMS_PATH, "utf8"));
  cache = Array.isArray(parsed?.platforms) ? parsed.platforms : [];
  return cache;
}

// Resolve a recipe's `meta.platforms` list against the catalog. Entries may be
// either a bare id ("modal") to use the catalog default, or an object
// ({ id: "modal", install, url, blurb }) to override individual fields for
// recipes that ship their own deploy script (e.g. Google/gemma4-modal.py).
export function resolveRecipePlatforms(entries) {
  if (!Array.isArray(entries) || entries.length === 0) return [];
  const catalog = loadPlatforms();
  return entries
    .map((entry) => {
      const id = typeof entry === "string" ? entry : entry?.id;
      const base = catalog.find((p) => p.id === id);
      if (!base) return null;
      if (typeof entry === "string") return base;
      return { ...base, ...entry };
    })
    .filter(Boolean);
}
