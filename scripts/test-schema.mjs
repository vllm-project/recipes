import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";
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
