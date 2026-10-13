import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { test } from "node:test";
import yaml from "js-yaml";
import { applyOverlay, resolveScope, splitOverlay } from "../src/lib/overlay-merge.js";
import { contentDir } from "../src/lib/content-root.js";

const ROOT = new URL("..", import.meta.url).pathname;
const taxonomy = yaml.load(fs.readFileSync(path.join(ROOT, "taxonomy.yaml"), "utf8"));
const ctx = resolveScope({ brands: ["Huawei"] }, taxonomy);
const source = "vllm-ascend";

function yamlFiles(dir) {
  return fs.readdirSync(dir, { withFileTypes: true }).flatMap((e) => {
    const full = path.join(dir, e.name);
    return e.isDirectory() ? yamlFiles(full) : /\.ya?ml$/.test(e.name) ? [full] : [];
  });
}

// Deep equality that ignores map key order (merge appends overlay keys) but
// keeps array order.
function canonical(v) {
  if (v instanceof Date) return v.toISOString();
  if (Array.isArray(v)) return v.map(canonical);
  if (v && typeof v === "object") {
    return Object.fromEntries(Object.keys(v).sort().map((k) => [k, canonical(v[k])]));
  }
  return v;
}

// The assembled tree (every source), so the round-trip covers recipes owned by
// other repos too. Run build-recipes-api / assemble-content first.
const recipes = yamlFiles(path.join(contentDir(), "models")).map((file) => ({
  file: path.relative(ROOT, file),
  recipe: yaml.load(fs.readFileSync(file, "utf8")),
}));

// Holds for any catalog — including one whose Ascend content already lives in
// vllm-ascend overlays, where every split is empty.
test("split → merge reproduces every recipe (Huawei scope)", () => {
  for (const { file, recipe } of recipes) {
    const { core, overlay } = splitOverlay(recipe, { ctx, source, guideHeading: /ascend/i });
    // The overlay must survive a YAML round-trip, as it will on disk.
    const ov = yaml.load(yaml.dump(overlay, { lineWidth: -1 })) || {};
    const { recipe: merged, errors } = applyOverlay(core, ov, { ctx, source });
    assert.deepEqual(errors, [], `${file}: ${errors.join("; ")}`);
    // Overlay deps land before core deps (or after, with `position: end`), so
    // an in-scope dep sitting between core deps moves to the front; compare
    // the in-scope and out-of-scope sequences separately.
    const byScope = (deps = []) => [deps.filter((d) => !ctx.brands.has(String(d.brand).toLowerCase())), deps.filter((d) => ctx.brands.has(String(d.brand).toLowerCase()))];
    const norm = (r) => canonical({ ...r, dependencies: byScope(r.dependencies) });
    assert.deepEqual(norm(merged), norm(recipe), `${file}: round-trip mismatch`);
    // Core keeps nothing in scope except prose and per-variant references.
    assert.equal(splitOverlay(core, { ctx, source, guideHeading: /ascend/i }).overlay.meta, undefined, file);
  }
});

// A fixed fixture (the starter template), so a catalog change can't break the
// scope tests — bump-sources.yml runs this file to validate new content.
const core0 = yaml.load(fs.readFileSync(path.join(ROOT, "templates/recipe.yaml"), "utf8"));
const errs = (overlay) => applyOverlay(core0, overlay, { ctx, source }).errors;

test("overlay can't touch out-of-scope hardware or model config", () => {
  assert.equal(errs({ hardware_overrides: { blackwell: { extra_args: ["--x"] } } }).length, 1);
  assert.equal(errs({ meta: { hardware: { b200: "unsupported" } } }).length, 1);
  assert.equal(errs({ model: { base_args: ["--x"] } }).length, 1);
  assert.equal(errs({ compatible_strategies: ["pd_cluster"] }).length, 1);
  assert.equal(errs({ strategy_hardware: { pd_cluster: { h200: "unsupported" } } }).length, 1);
  assert.equal(errs({ meta: { default_hardware: "ascend_910c" } }).length, 1);
});

test("overlay is add-only", () => {
  const once = applyOverlay(core0, { meta: { hardware: { ascend_910c: "verified" } } }, { ctx, source }).recipe;
  assert.equal(applyOverlay(once, { meta: { hardware: { ascend_910c: "unsupported" } } }, { ctx, source }).errors.length, 1);
  // An existing variant can't be replaced wholesale, only extended per hardware.
  assert.equal(errs({ variants: { default: { precision: "int4" } } }).length, 1);
  assert.deepEqual(errs({ variants: { default: { hardware_overrides: { ascend_910c: { docker_image: "x" } } } } }), []);
});

test("new variants and dependencies must be in scope", () => {
  assert.equal(errs({ variants: { foo: { precision: "fp8", supported_hardware: ["h200"] } } }).length, 1);
  assert.equal(errs({ variants: { foo: { precision: "fp8" } } }).length, 1);
  assert.deepEqual(errs({ variants: { foo: { precision: "fp8", supported_hardware: ["ascend_910c"] } } }), []);
  assert.equal(errs({ dependencies: [{ command: "pip install x" }] }).length, 1);
  assert.deepEqual(errs({ dependencies: [{ command: "pip install x", brand: "Huawei" }] }), []);
});

test("an overlay source needs a granted scope", () => {
  const none = resolveScope({}, taxonomy);
  assert.equal(applyOverlay(core0, { meta: { hardware: { ascend_910c: "verified" } } }, { ctx: none, source }).errors.length, 1);
});

test("guide sections fill their marker, else append", () => {
  const core = { meta: {}, guide: "## A\n\n<!-- overlay:vllm-ascend:npu -->\n## B\n" };
  const { recipe } = applyOverlay(core, { guide_sections: { npu: "## NPU\nx\n\n", tail: "## Tail\n" } }, { ctx, source });
  assert.equal(recipe.guide, "## A\n\n## NPU\nx\n\n## B\n\n## Tail\n");
});
