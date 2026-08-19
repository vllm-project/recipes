import fs from "fs";
import path from "path";
import yaml from "js-yaml";

let cache = null;

const STRATEGIES_DIR = path.join(process.cwd(), "strategies");
// KV-store deployments (deploy_type: kv_store_lb) live in their own dir —
// they power the KV Offload row, not the Strategy row — but merge into the
// same map: resolveCommand and the command builder look every deployment
// spec up by id.
const KV_STORE_DIR = path.join(process.cwd(), "kv_store");

export function loadStrategies() {
  if (cache) return cache;
  cache = {};
  for (const dir of [STRATEGIES_DIR, KV_STORE_DIR]) {
    if (!fs.existsSync(dir)) continue;
    const files = fs.readdirSync(dir).filter((f) => f.endsWith(".yaml"));
    for (const file of files) {
      const s = yaml.load(fs.readFileSync(path.join(dir, file), "utf8"));
      cache[s.name] = s;
    }
  }
  return cache;
}

export function loadStrategy(name) {
  const all = loadStrategies();
  return all[name] || null;
}
