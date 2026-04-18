import fs from "fs";
import path from "path";
import yaml from "js-yaml";

let cache = null;

const TAXONOMY_PATH = path.join(process.cwd(), "taxonomy.yaml");

export function loadTaxonomy() {
  if (cache) return cache;
  cache = yaml.load(fs.readFileSync(TAXONOMY_PATH, "utf8"));
  return cache;
}
