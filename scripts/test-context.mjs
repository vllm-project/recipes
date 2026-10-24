// Configuration-scoped guide entries (`notes` / `troubleshooting`): `when`
// matching, labels, cross-field validation and the schema shape.
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { test } from "node:test";
import yaml from "js-yaml";
import Ajv from "ajv";
import { buildRecipeSchema } from "../src/lib/recipe-schema.js";
import { contentDir } from "../src/lib/content-root.js";
import { validateContextEntries, whenLabels, whenMatches } from "../src/lib/recipe-context.js";

const dir = contentDir();
const load = (f) => yaml.load(fs.readFileSync(f, "utf8"));
const taxonomy = load(path.join(dir, "taxonomy.yaml"));
const strategies = Object.fromEntries(
  ["strategies", "kv_store"].flatMap((d) => fs.readdirSync(path.join(dir, d))
    .filter((f) => f.endsWith(".yaml"))
    .map((f) => { const s = load(path.join(dir, d, f)); return [s.name, s]; })),
);
const hw = taxonomy.hardware_profiles;
const sel = (o) => ({ hwId: "h200", strategy: "single_node_tp", variant: "default", features: [], ...o });

test("when: absent matches anything, a null selection matches nothing conditional", () => {
  assert.equal(whenMatches(undefined, sel(), hw), true);
  assert.equal(whenMatches({ hardware: ["nvidia"] }, null, hw), false);
});

test("when.hardware: exact id, generation, lowercase brand and arch selector", () => {
  assert.equal(whenMatches({ hardware: ["h200"] }, sel(), hw), true);
  assert.equal(whenMatches({ hardware: ["hopper"] }, sel(), hw), true);
  assert.equal(whenMatches({ hardware: ["nvidia"] }, sel(), hw), true);
  assert.equal(whenMatches({ hardware: ["blackwell", "amd"] }, sel(), hw), false);
  assert.equal(whenMatches({ hardware: ["arch:gfx950"] }, sel({ hwId: "mi355x" }), hw), true);
  assert.equal(whenMatches({ hardware: ["arch:gfx950"] }, sel(), hw), false);
});

test("when: keys AND together, values OR together; features match while on", () => {
  const when = { hardware: ["gb200", "gb300"], strategies: ["pd_cluster", "multi_node_dep"] };
  assert.equal(whenMatches(when, sel({ hwId: "gb300", strategy: "pd_cluster" }), hw), true);
  assert.equal(whenMatches(when, sel({ hwId: "gb300", strategy: "single_node_tp" }), hw), false);
  assert.equal(whenMatches(when, sel({ hwId: "h200", strategy: "pd_cluster" }), hw), false);
  assert.equal(whenMatches({ features: ["tool_calling"] }, sel({ features: ["reasoning", "tool_calling"] }), hw), true);
  assert.equal(whenMatches({ features: ["tool_calling"] }, sel({ features: ["reasoning"] }), hw), false);
  assert.equal(whenMatches({ variants: ["nvfp4"] }, sel({ variant: "nvfp4" }), hw), true);
});

test("whenLabels uses display names", () => {
  const recipe = { variants: { nvfp4: { precision: "nvfp4" } }, features: { tool_calling: {} } };
  const labels = whenLabels(
    { hardware: ["gb200", "nvidia", "hopper"], strategies: ["pd_cluster"], variants: ["nvfp4"], features: ["tool_calling"] },
    { hardwareProfiles: hw, strategies, recipe },
  );
  assert.deepEqual(labels, [hw.gb200.display_name, "NVIDIA", "Hopper", strategies.pd_cluster.display_name, "NVFP4", "Tool calling"]);
});

test("validateContextEntries rejects unknown variants and features", () => {
  const recipe = {
    variants: { default: {} },
    features: { reasoning: {} },
    notes: [{ when: { variants: ["fp8"] }, body: "x" }],
    troubleshooting: [{ title: "t", when: { features: ["tool_calling"] }, body: "x" }],
  };
  assert.deepEqual(validateContextEntries(recipe), [
    'notes[0].when.variants: no variant "fp8"',
    'troubleshooting[0].when.features: no feature "tool_calling"',
  ]);
});

test("schema: notes need a when, troubleshooting a title; ids are checked", () => {
  const validate = new Ajv({ allErrors: true, strict: false }).compile(buildRecipeSchema({ taxonomy, strategies }));
  const base = load(new URL("../templates/recipe.yaml", import.meta.url).pathname);
  base.meta.date_added = "2026-01-01";
  base.meta.date_updated = "2026-01-01";
  const ok = (patch) => validate({ ...structuredClone(base), ...patch });
  assert.equal(ok({}), true, JSON.stringify(validate.errors));
  assert.equal(ok({ notes: [{ when: { hardware: ["blackwell"] }, body: "x" }] }), true, JSON.stringify(validate.errors));
  assert.equal(ok({ notes: [{ body: "x" }] }), false);
  assert.equal(ok({ troubleshooting: [{ body: "x" }] }), false);
  assert.equal(ok({ troubleshooting: [{ title: "t", body: "x" }] }), true);
  assert.equal(ok({ notes: [{ when: { hardware: ["h900"] }, body: "x" }] }), false);
  assert.equal(ok({ notes: [{ when: { strategies: ["warp_drive"] }, body: "x" }] }), false);
  assert.equal(ok({ notes: [{ when: { gpu: ["h200"] }, body: "x" }] }), false);
});
