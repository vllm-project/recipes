import fs from "fs";
import path from "path";
import yaml from "js-yaml";

let cache = null;

const MODELS_DIR = path.join(process.cwd(), "models");

function parseRecipe(filePath) {
  const raw = yaml.load(fs.readFileSync(filePath, "utf8"));
  // js-yaml auto-parses YYYY-MM-DD into Date objects — normalize to string
  if (raw.meta?.date_updated instanceof Date) {
    raw.meta.date_updated = raw.meta.date_updated.toISOString().split("T")[0];
  }
  return raw;
}

export function getAllRecipes() {
  if (cache) return cache;
  const files = fs.readdirSync(MODELS_DIR).filter((f) => f.endsWith(".yaml"));
  cache = files
    .map((f) => parseRecipe(path.join(MODELS_DIR, f)))
    .sort((a, b) => {
      const da = new Date(a.meta.date_updated);
      const db = new Date(b.meta.date_updated);
      return db - da;
    });
  return cache;
}

export function getRecipe(slug) {
  const all = getAllRecipes();
  return all.find((r) => r.meta.slug === slug) || null;
}

export function getRecipeSlugs() {
  return getAllRecipes().map((r) => r.meta.slug);
}
