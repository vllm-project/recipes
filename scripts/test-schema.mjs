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

// Reference fields name another object in the recipe (a feature, a variant, a
// platform, a strategy) and every one of them is a lookup that misses
// silently: no schema violation, no renderer error, just a page that quietly
// differs from what the author wrote. Run the real build — a validator that
// only exists in the test proves nothing.
test("the build rejects a reference that names nothing", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "recipes-refs-"));
  const content = path.join(dir, "content");
  fs.mkdirSync(path.join(content, "models/example"), { recursive: true });
  for (const name of ["taxonomy.yaml", "platforms.yaml", "strategies", "kv_store"]) {
    fs.cpSync(path.join(contentDir(), name), path.join(content, name), { recursive: true });
  }
  // `meta` holds the meta-scoped references, `top` the recipe-level ones.
  const recipe = (meta = "", top = "") => `meta:
  title: "Probe"
  slug: "probe"
  provider: "Example"
  description: "Fixture."
  date_added: 2026-01-01
  date_updated: 2026-01-01
  tasks: [text]
${meta}
model:
  model_id: "example/probe"
  min_vllm_version: "0.30.0"
  architecture: dense
  parameter_count: "8B"
  active_parameters: "8B"
  context_length: 8192
${top}
features:
  tool_calling:
    description: "Tool calling."
    args: ["--enable-auto-tool-choice"]
variants:
  default: { precision: bf16, vram_minimum_gb: 20 }
compatible_strategies: [single_node_tp]
guide: |
  ## Overview
  Fixture.
`;
  const file = path.join(content, "models/example/Probe.yaml");
  const build = (meta, top) => {
    fs.writeFileSync(file, recipe(meta, top));
    return spawnSync(process.execPath, ["scripts/build-recipes-api.mjs"], {
      cwd: path.resolve(import.meta.dirname, ".."),
      env: { ...process.env, RECIPES_CONTENT_DIR: content, RECIPES_PUBLIC_DIR: path.join(dir, "public") },
      encoding: "utf8",
    });
  };
  try {
    // The bare fixture is valid, so every failure below is the mutation's.
    const clean = build();
    assert.equal(clean.status, 0, clean.stderr);
    assert.match(clean.stdout, /✓ JSON API: 1 models/);

    // [field, meta-level addition, recipe-level addition, the message to expect]
    const cases = [
      ["meta.platforms", "  platforms: [modall]", "", /meta\.platforms: unknown platform "modall"/],
      // An object entry names its platform through `id`; a typo there is the
      // same silent drop as one in a bare string.
      ["meta.platforms.id", '  platforms: [{ id: sagemakerr, blurb: "x" }]', "", /meta\.platforms: unknown platform "sagemakerr"/],
      ["opt_in_features", "", "opt_in_features: [tool_callingg]", /opt_in_features: no feature "tool_callingg"/],
      ["hardware_opt_in_features", "", "hardware_opt_in_features: { h200: [no_such_feature] }", /hardware_opt_in_features\.h200: no feature "no_such_feature"/],
      ["meta.default_variant", "  default_variant: { h200: typo_variant }", "", /meta\.default_variant\.h200: no variant "typo_variant"/],
      ["default_strategy", "", "default_strategy: multi_node_dep", /default_strategy: "multi_node_dep" is not offered/],
    ];
    for (const [field, meta, top, message] of cases) {
      const run = build(meta, top);
      assert.equal(run.status, 1, `${field}: expected a build failure`);
      assert.match(run.stderr, /Unresolvable references/, `${field}: wrong error class`);
      assert.match(run.stderr, message, `${field}: wrong message`);
    }

    // A reference that does resolve stays green: a catalog platform id, a
    // feature and variant that exist, and a strategy the recipe offers.
    const ok = build(
      '  platforms: [modal]\n  default_variant: { h200: default }',
      "opt_in_features: [tool_calling]\ndefault_strategy: single_node_tp",
    );
    assert.equal(ok.status, 0, ok.stderr);
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});
