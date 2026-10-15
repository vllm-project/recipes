/**
 * Migration helper: split every recipe under models/ into a core recipe and
 * an overlay holding the keys a brand scope owns (the inverse of the overlay
 * merge in assemble-content.mjs).
 *
 *   node scripts/split-overlay.mjs --brands Huawei --source vllm-ascend --out <dir> [--guide-heading Ascend]
 *
 * Writes <dir>/core/models/<org>/<repo>.yaml and
 * <dir>/<source>/overlays/<org>/<repo>.yaml. Output is re-serialized YAML
 * (comments dropped): use it to stage overlay files and to verify a split
 * round-trips; edit the commented core YAMLs by hand.
 */

import fs from "fs";
import path from "path";
import yaml from "js-yaml";
import { resolveScope, splitOverlay } from "../src/lib/overlay-merge.js";
import { parseArgs } from "./lib/content-sources.mjs";

const args = parseArgs(process.argv.slice(2));
if (!args.brands || !args.source || !args.out) {
  console.error("usage: split-overlay.mjs --brands <B1,B2> --source <name> --out <dir> [--models <dir>] [--guide-heading <regex>]");
  process.exit(1);
}
const ROOT = process.cwd();
const modelsDir = path.resolve(args.models || path.join(ROOT, "models"));
const out = path.resolve(args.out);
const taxonomy = yaml.load(fs.readFileSync(path.join(ROOT, "taxonomy.yaml"), "utf8"));
const ctx = resolveScope({ brands: String(args.brands).split(",") }, taxonomy);
const guideHeading = args["guide-heading"] ? new RegExp(args["guide-heading"], "i") : null;
const dump = (o) => yaml.dump(o, { lineWidth: -1, noRefs: true });

function yamlFiles(dir) {
  return fs.readdirSync(dir, { withFileTypes: true }).flatMap((e) => {
    const full = path.join(dir, e.name);
    return e.isDirectory() ? yamlFiles(full) : /\.ya?ml$/.test(e.name) ? [full] : [];
  });
}

let split = 0;
for (const file of yamlFiles(modelsDir).sort()) {
  const rel = path.relative(modelsDir, file);
  const { core, overlay, movedVariants } = splitOverlay(yaml.load(fs.readFileSync(file, "utf8")), {
    ctx, source: args.source, guideHeading,
  });
  const coreFile = path.join(out, "core", "models", rel);
  fs.mkdirSync(path.dirname(coreFile), { recursive: true });
  if (Object.keys(overlay).length === 0) {
    fs.copyFileSync(file, coreFile);
    continue;
  }
  fs.writeFileSync(coreFile, dump(core));
  const ovFile = path.join(out, args.source, "overlays", rel);
  fs.mkdirSync(path.dirname(ovFile), { recursive: true });
  fs.writeFileSync(ovFile, dump(overlay));
  split++;
  console.log(`  ${rel}: ${Object.keys(overlay).join(", ")}${movedVariants.length ? ` (variants: ${movedVariants.join(", ")})` : ""}`);
}
console.log(`✓ ${split} recipes split → ${out}`);
