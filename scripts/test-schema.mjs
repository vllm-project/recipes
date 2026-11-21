import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { test } from "node:test";
import yaml from "js-yaml";
import Ajv from "ajv";
import { buildRecipeSchema } from "../src/lib/recipe-schema.js";
import { contentDir } from "../src/lib/content-root.js";
import { resolveRelatedRecipes } from "../src/lib/recipe-utils.js";

// Reads the assembled tree (every source): run build-recipes-api first.
const dir = contentDir();
const load = (f) => yaml.load(fs.readFileSync(f, "utf8"));
const taxonomy = load(path.join(dir, "taxonomy.yaml"));
const strategies = Object.fromEntries(
  ["strategies", "kv_store"].flatMap((d) => fs.readdirSync(path.join(dir, d))
    .filter((f) => f.endsWith(".yaml"))
    .map((f) => { const s = load(path.join(dir, d, f)); return [s.name, s]; })),
);
const validate = new Ajv({ allErrors: true, strict: false }).compile(buildRecipeSchema({ taxonomy, strategies }));

// js-yaml parses YYYY-MM-DD as Date; the build normalizes the same way.
const normalize = (v) => (v instanceof Date ? v.toISOString().slice(0, 10)
  : Array.isArray(v) ? v.map(normalize)
  : v && typeof v === "object" ? Object.fromEntries(Object.entries(v).map(([k, x]) => [k, normalize(x)])) : v);

function yamlFiles(d) {
  return fs.readdirSync(d, { withFileTypes: true }).flatMap((e) => {
    const full = path.join(d, e.name);
    return e.isDirectory() ? yamlFiles(full) : /\.ya?ml$/.test(e.name) ? [full] : [];
  });
}

const recipes = yamlFiles(path.join(dir, "models")).map((f) => ({ f, r: normalize(load(f)) }));
// Mutations start from a fixed fixture (the starter template), not a catalog
// recipe that a source update could rename or remove.
const fixture = normalize(load(new URL("../templates/recipe.yaml", import.meta.url).pathname));
const errorsFor = (mutate) => {
  const r = structuredClone(fixture);
  mutate(r);
  return validate(r) ? [] : validate.errors;
};

test("every recipe matches the schema", () => {
  for (const { f, r } of recipes) assert.ok(validate(r), `${f}: ${JSON.stringify(validate.errors?.slice(0, 3))}`);
});

test("templates match the schema", () => {
  const tdir = new URL("../templates", import.meta.url).pathname;
  for (const f of yamlFiles(tdir)) assert.ok(validate(normalize(load(f))), `${f}: ${JSON.stringify(validate.errors?.slice(0, 3))}`);
});

test("misspelled keys are rejected", () => {
  assert.ok(errorsFor((r) => { r.compatible_stratgies = r.compatible_strategies; delete r.compatible_strategies; }).length);
  assert.ok(errorsFor((r) => { r.variants.default.extra_arg = ["--x"]; }).length);
  assert.ok(errorsFor((r) => { r.features.tool_calling.arg = ["--x"]; }).length);
});

test("vocabulary and status values are checked", () => {
  assert.ok(errorsFor((r) => { r.variants.default.supported_frontends = ["go"]; }).length);
  assert.ok(errorsFor((r) => { r.meta.hardware.h999 = "verified"; }).length);
  assert.ok(errorsFor((r) => { r.meta.hardware.h200 = "unverified"; }).length);
  assert.ok(errorsFor((r) => { r.compatible_strategies = ["single_node_tpp"]; }).length);
  assert.ok(errorsFor((r) => { r.hardware_overrides.dgx_spark_gb10 = "verified"; }).length);
  assert.deepEqual(errorsFor((r) => { r.meta.hardware.dgx_spark_gb10 = "supported"; }), []);
});

test("build rejects inconsistent variant feature and frontend references", () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "recipes-variant-validation-"));
  try {
    const input = path.join(tmp, "content");
    fs.mkdirSync(path.join(input, "models/TODO-org"), { recursive: true });
    for (const name of ["taxonomy.yaml", "platforms.yaml", "strategies", "kv_store"]) {
      fs.cpSync(path.join(dir, name), path.join(input, name), { recursive: true });
    }
    const build = (extra, modeVariants) => {
      const r = structuredClone(fixture);
      r.model.default_frontend = "python";
      r.model.rust_frontend = "unsupported";
      r.features = { reasoning: { description: "Reasoning", modes: { on: { args: ["--reasoning-parser", "example"], variants: modeVariants } } } };
      r.variants.local = { ...r.variants.default, supported_features: [], supported_frontends: ["python"], ...extra };
      fs.writeFileSync(path.join(input, "models/TODO-org/TODO-repo.yaml"), yaml.dump(r));
      return spawnSync(process.execPath, ["scripts/build-recipes-api.mjs"], {
        env: { ...process.env, RECIPES_CONTENT_DIR: input, RECIPES_PUBLIC_DIR: path.join(tmp, "public") },
        encoding: "utf8",
      });
    };
    for (const [extra, field] of [
      [{ supported_features: ["reasonin"] }, "supported_features"],
      [{ supported_frontends: ["rust"] }, "supported_frontends"],
      [{ default_modes: { reasoning: "on" } }, "default_modes.reasoning"],
    ]) {
      const result = build(extra);
      assert.notEqual(result.status, 0);
      assert.ok(result.stderr.includes(`variants.local.${field}`), result.stderr);
    }
    for (const [extra, modes] of [[{}, ["default"]], [{ supported_features: undefined }, undefined]]) {
      const result = build(extra, modes);
      assert.equal(result.status, 0, result.stderr);
    }
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true });
  }
});

test("related_recipes resolves every entry case-insensitively", () => {
  // The page matches `related_recipes` against `hf_id` and `meta.slug`, and
  // silently drops what it can't find. The org half of an HF id is lowercase
  // while the directory that derives `hf_id` keeps the author's casing
  // (`models/Google/…`), so a case-sensitive match lost 14 of 289 entries in
  // the catalog with nothing rendered to show for it.
  const recipes = [
    { hf_id: "Google/gemma-4-31B-it", meta: { slug: "gemma-4-31b-it" } },
    { hf_id: "google/gemma-4-12B-it", meta: { slug: "gemma-4-12b-it" } },
  ];
  const one = (related) => resolveRelatedRecipes({ hf_id: "other/model", meta: { related_recipes: related } }, recipes);

  assert.deepEqual(one(["google/gemma-4-31B-it"]).map((r) => r.hf_id), ["Google/gemma-4-31B-it"]);
  assert.deepEqual(one(["Google/gemma-4-31B-it"]).map((r) => r.hf_id), ["Google/gemma-4-31B-it"]);
  assert.deepEqual(one(["gemma-4-12b-it"]).map((r) => r.hf_id), ["google/gemma-4-12B-it"]);
  assert.deepEqual(one(["google/nope"]), []);
  assert.deepEqual(one(undefined), []);
  // An entry naming the page itself, or repeated, would render a link back to
  // itself or the same link twice.
  const self = { hf_id: "Google/gemma-4-31B-it", meta: { related_recipes: ["google/gemma-4-31B-it", "gemma-4-31b-it"] } };
  assert.deepEqual(resolveRelatedRecipes(self, recipes), []);
  assert.deepEqual(one(["gemma-4-12b-it", "google/gemma-4-12B-it"]).map((r) => r.hf_id), ["google/gemma-4-12B-it"]);

  // A non-array value is a schema error, but this also runs in the page against
  // whatever the loader returned — it must render no row, not throw in render.
  assert.deepEqual(one(42), []);
  assert.deepEqual(one("google/gemma-4-31B-it"), []);

  // `hf_id` is derived from the file path, so a recipe outside
  // `models/<org>/<repo>.yaml` has none. Indexing must skip it rather than
  // throw, and an entry matching only slugs must still resolve.
  const withHfIdLess = [...recipes, { meta: { slug: "no-id" } }];
  assert.deepEqual(
    resolveRelatedRecipes({ hf_id: "other/model", meta: { related_recipes: ["gemma-4-12b-it", "no-id"] } }, withHfIdLess)
      .map((r) => r.meta.slug),
    ["gemma-4-12b-it", "no-id"],
  );
  // A page with no `hf_id` of its own must not treat every id-less match as a
  // self-reference and drop it.
  assert.deepEqual(
    resolveRelatedRecipes({ meta: { related_recipes: ["no-id"] } }, withHfIdLess).map((r) => r.meta.slug),
    ["no-id"],
  );
  // ... while a repeated id-less entry is still deduped.
  assert.deepEqual(
    resolveRelatedRecipes({ meta: { related_recipes: ["no-id", "no-id"] } }, withHfIdLess).map((r) => r.meta.slug),
    ["no-id"],
  );
});

// The build rejects an unresolvable entry so it can't vanish silently. Run the
// real script: a validator that only exists in the test proves nothing.
test("the build rejects a related_recipes entry that resolves to nothing", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "recipes-related-"));
  const content = path.join(dir, "content");
  fs.mkdirSync(path.join(content, "models/example"), { recursive: true });
  for (const name of ["taxonomy.yaml", "platforms.yaml", "strategies", "kv_store"]) {
    fs.cpSync(path.join(contentDir(), name), path.join(content, name), { recursive: true });
  }
  const recipe = (title, slug, related) => `meta:
  title: "${title}"
  slug: "${slug}"
  provider: "Example"
  description: "Fixture."
  date_added: 2026-01-01
  date_updated: 2026-01-01
  tasks: [text]
  related_recipes: [${related}]
model:
  model_id: "example/${slug}"
  min_vllm_version: "0.30.0"
  architecture: dense
  parameter_count: "8B"
  active_parameters: "8B"
  context_length: 8192
variants:
  default: { precision: bf16, vram_minimum_gb: 20 }
compatible_strategies: [single_node_tp]
guide: |
  ## Overview
  Fixture.
`;
  const alpha = path.join(content, "models/example/Alpha.yaml");
  fs.writeFileSync(path.join(content, "models/example/Beta.yaml"), recipe("Beta", "beta", ""));
  const build = () => spawnSync(process.execPath, ["scripts/build-recipes-api.mjs"], {
    cwd: path.resolve(import.meta.dirname, ".."),
    env: { ...process.env, RECIPES_CONTENT_DIR: content, RECIPES_PUBLIC_DIR: path.join(dir, "public") },
    encoding: "utf8",
  });
  try {
    // Lowercase id (the HF org form), a legacy slug, and the directory casing
    // all resolve; a typo does not.
    for (const ok of ['"example/beta"', '"beta"', '"Example/Beta"']) {
      fs.writeFileSync(alpha, recipe("Alpha", "alpha", ok));
      const run = build();
      assert.equal(run.status, 0, `${ok}: ${run.stderr}`);
    }
    fs.writeFileSync(alpha, recipe("Alpha", "alpha", '"example/nope"'));
    const run = build();
    assert.equal(run.status, 1);
    assert.match(run.stderr, /Unresolvable related_recipes/);
    assert.match(run.stderr, /names "example\/nope"/);

    // The collection loop runs before schema failures are thrown, so a
    // non-array `related_recipes` must reach the schema report rather than
    // crash the pre-scan with "not iterable" — that error names neither the
    // file nor the field the author has to fix.
    fs.writeFileSync(alpha, recipe("Alpha", "alpha", "").replace("related_recipes: []", "related_recipes: 42"));
    const malformed = build();
    assert.equal(malformed.status, 1);
    assert.doesNotMatch(malformed.stderr, /is not iterable/);
    assert.match(malformed.stderr, /meta\/related_recipes must be array/);
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});
