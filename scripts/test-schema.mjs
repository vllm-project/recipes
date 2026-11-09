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

// `meta.hardware.<gpu>` is a claim about a GPU; whether a rendering exists for
// that GPU is computed from precision + the variant allowlist + single-node
// fit. Nothing compares the two, so a claim no variant backs keeps its ✓ badge,
// stays in the picker and on the browse facets, and then disables every
// variant — the page says the model runs there and gives no way to run it.
// Same for a variant override keyed on a GPU its own allowlist excludes: dead
// config, never read. Warnings, not failures — the YAML belongs upstream.
test("the build reports hardware declarations no variant can honour", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "recipes-claims-"));
  const content = path.join(dir, "content");
  fs.mkdirSync(path.join(content, "models/example"), { recursive: true });
  for (const name of ["taxonomy.yaml", "platforms.yaml", "strategies", "kv_store"]) {
    fs.cpSync(path.join(contentDir(), name), path.join(content, name), { recursive: true });
  }
  // `hardware` is the meta-level map, `variant` the variant body.
  const recipe = (hardware, variant = "    precision: bf16\n    vram_minimum_gb: 20") => `meta:
  title: "Probe"
  slug: "probe"
  provider: "Example"
  description: "Fixture."
  date_added: 2026-01-01
  date_updated: 2026-01-01
  tasks: [text]
  hardware: ${hardware}
model:
  model_id: "example/probe"
  min_vllm_version: "0.30.0"
  architecture: dense
  parameter_count: "8B"
  active_parameters: "8B"
  context_length: 8192
variants:
  default:
${variant}
compatible_strategies: [single_node_tp]
guide: |
  ## Overview
  Fixture.
`;
  const file = path.join(content, "models/example/Probe.yaml");
  const build = (hardware, variant) => {
    fs.writeFileSync(file, recipe(hardware, variant));
    return spawnSync(process.execPath, ["scripts/build-recipes-api.mjs"], {
      cwd: path.resolve(import.meta.dirname, ".."),
      env: { ...process.env, RECIPES_CONTENT_DIR: content, RECIPES_PUBLIC_DIR: path.join(dir, "public") },
      encoding: "utf8",
    });
  };
  try {
    // A claim the only variant backs: no warning, and the build is green.
    const clean = build("{ h200: verified }");
    assert.equal(clean.status, 0, clean.stderr);
    assert.match(clean.stdout, /✓ JSON API: 1 models/);
    assert.doesNotMatch(clean.stderr, /no variant runs there|never read/);

    // `gb200` claimed verified while the only variant's allowlist excludes it.
    // The build still succeeds — the stale claim is upstream content.
    const orphan = build("{ h200: verified, gb200: verified }", "    precision: bf16\n    vram_minimum_gb: 20\n    supported_hardware: [h200]");
    assert.equal(orphan.status, 0, orphan.stderr);
    assert.match(orphan.stderr, /meta\.hardware\.gb200: declared verified, but no variant runs there/);
    // The message names the rule, so the reader knows whether to widen the
    // allowlist or drop the claim.
    assert.match(orphan.stderr, /supported_hardware is h200/);
    // h200 itself is backed, so it stays quiet: the check is per pair.
    assert.doesNotMatch(orphan.stderr, /meta\.hardware\.h200/);

    // A variant override keyed on a GPU the variant excludes is never read.
    const dead = build(
      "{ h200: verified }",
      "    precision: bf16\n    vram_minimum_gb: 20\n    supported_hardware: [h200]\n    hardware_overrides:\n      gb200:\n        extra_args: [\"--max-num-seqs\", \"1\"]",
    );
    assert.equal(dead.status, 0, dead.stderr);
    assert.match(dead.stderr, /variants\.default\.hardware_overrides\.gb200: never read/);

    // Vram that can't fit a non-scalable profile is the third way a claim goes
    // unbacked; say so by name rather than repeating the allowlist message.
    const big = build("{ rtx_pro_6000: supported }", "    precision: bf16\n    vram_minimum_gb: 500");
    assert.equal(big.status, 0, big.stderr);
    assert.match(big.stderr, /meta\.hardware\.rtx_pro_6000: declared supported, but no variant runs there/);
    assert.match(big.stderr, /do not fit single-node/);

    // The budget the message quotes must be the one the fit check used. An
    // exact-GPU `hardware_overrides.<gpu>.vram_minimum_gb` replaces the
    // variant's top-level value, so quoting the latter names a number that was
    // never compared against the node.
    const overridden = build(
      "{ rtx_pro_6000: supported }",
      "    precision: bf16\n    vram_minimum_gb: 20\n    hardware_overrides:\n      rtx_pro_6000:\n        vram_minimum_gb: 500",
    );
    assert.equal(overridden.status, 0, overridden.stderr);
    assert.match(overridden.stderr, /meta\.hardware\.rtx_pro_6000: declared supported, but no variant runs there/);
    assert.match(overridden.stderr, /do not fit single-node on rtx_pro_6000 \(500 GB\)/);
    // …and the top-level 20 GB, which does fit, must not be what is reported.
    assert.doesNotMatch(overridden.stderr, /\(20 GB\)/);
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});
