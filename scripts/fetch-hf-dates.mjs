/**
 * Build-time: fetch `created_at` from HuggingFace for each recipe's base model_id,
 * write to a manifest at public/hf-dates.json so the UI can sort by release date.
 *
 * Cached between builds (reads existing manifest, only fetches missing entries).
 */

import fs from "fs";
import path from "path";
import yaml from "js-yaml";

const ROOT = process.cwd();
const MANIFEST = path.join(ROOT, "public", "hf-dates.json");

// Load existing manifest (cache)
let cache = {};
if (fs.existsSync(MANIFEST)) {
  try { cache = JSON.parse(fs.readFileSync(MANIFEST, "utf8")); } catch {}
}

function findYamlFiles(dir) {
  const out = [];
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, e.name);
    if (e.isDirectory()) out.push(...findYamlFiles(full));
    else if (e.name.endsWith(".yaml") || e.name.endsWith(".yml")) out.push(full);
  }
  return out;
}

async function fetchCreatedAt(modelId) {
  try {
    const res = await fetch(`https://huggingface.co/api/models/${modelId}`, {
      headers: { "User-Agent": "vllm-recipes-build/1.0" },
    });
    if (!res.ok) return null;
    const data = await res.json();
    return data.createdAt || data.created_at || null;
  } catch {
    return null;
  }
}

const files = findYamlFiles(path.join(ROOT, "models"));
const hfIds = files.map((f) => {
  const rel = path.relative(path.join(ROOT, "models"), f);
  const parts = rel.split(path.sep);
  const repo = parts[parts.length - 1].replace(/\.(yaml|yml)$/, "");
  return `${parts[0]}/${repo}`;
});

let fetched = 0, cached = 0, failed = 0;
for (const id of hfIds) {
  if (cache[id]) { cached++; continue; }
  const ts = await fetchCreatedAt(id);
  if (ts) {
    cache[id] = ts;
    fetched++;
  } else {
    failed++;
  }
  // Gentle rate limit
  await new Promise((r) => setTimeout(r, 100));
}

fs.mkdirSync(path.dirname(MANIFEST), { recursive: true });
fs.writeFileSync(MANIFEST, JSON.stringify(cache, null, 2));

console.log(`✓ HF dates: ${fetched} fetched, ${cached} cached, ${failed} failed (${hfIds.length} total)`);
