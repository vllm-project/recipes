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
