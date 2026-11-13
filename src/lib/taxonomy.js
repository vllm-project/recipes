import fs from "fs";
import path from "path";
import yaml from "js-yaml";
import { contentDir } from "./content-root.js";

let cache = null;

const TAXONOMY_PATH = path.join(contentDir(), "taxonomy.yaml");

export function loadTaxonomy() {
  if (cache) return cache;
  cache = yaml.load(fs.readFileSync(TAXONOMY_PATH, "utf8"));
  return cache;
}
