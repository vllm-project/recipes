import fs from "fs";
import path from "path";
import yaml from "js-yaml";

let cache = null;

const STRATEGIES_DIR = path.join(process.cwd(), "strategies");

export function loadStrategies() {
  if (cache) return cache;
  cache = {};
  const files = fs.readdirSync(STRATEGIES_DIR).filter((f) => f.endsWith(".yaml"));
  for (const file of files) {
    const s = yaml.load(fs.readFileSync(path.join(STRATEGIES_DIR, file), "utf8"));
    cache[s.name] = s;
  }
  return cache;
}

export function loadStrategy(name) {
  const all = loadStrategies();
  return all[name] || null;
}
