import assert from "node:assert/strict";
import { execFileSync, spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { test } from "node:test";
import yaml from "js-yaml";
import Ajv from "ajv";
import { createAscendPdRecipe } from "./fixtures/ascend-deployment.mjs";
import { resolvePdRoleLayout, initialVariantKey, isVariantHardwareSupported } from "../src/lib/command-synthesis.js";
import { bindRouterPort } from "../src/lib/cluster-endpoints.js";
import { resolveModelReference, validateModelReferences } from "../src/lib/model-reference.js";
import { buildRecipeSchema } from "../src/lib/recipe-schema.js";
import { resolveScope, applyOverlay, splitOverlay } from "../src/lib/overlay-merge.js";
import { buildVariantRendering, renderCommand, resolveSelection } from "../src/lib/recipe-render.js";
import { selectionMarkdown } from "../src/lib/recipe-markdown.js";

const root = fileURLToPath(new URL("../", import.meta.url));
const read = (file) => yaml.load(fs.readFileSync(path.join(root, file), "utf8"));
const taxonomy = read("taxonomy.yaml");
const strategies = Object.fromEntries(["strategies", "kv_store"].flatMap((dir) => fs.readdirSync(path.join(root, dir))
  .filter((file) => file.endsWith(".yaml")).map((file) => { const value = read(`${dir}/${file}`); return [value.name, value]; })));
const flag = (argv, key) => argv[argv.lastIndexOf(key) + 1];
const select = (recipe, params) => resolveSelection(recipe, new URLSearchParams(params), strategies, taxonomy);
const pageQuery = { hardware: "ascend_950dt", strategy: "pd_cluster", prefill_nodes: "5", decode_nodes: "5", prefill_rank: "7", decode_rank: "999", prefill_mode: "dep", decode_mode: "tp" };
const urls = { page: "https://example.test/test/Ascend-PD", md: "https://example.test/test/Ascend-PD.md", json: "https://example.test/test/Ascend-PD.json" };

test("shared layouts and selection round five nodes to eight, with group-local rank and master", () => {
  const recipe = createAscendPdRecipe();
  recipe.hf_id = "test/Ascend-PD";
  const resolved = select(recipe, pageQuery);
  assert.equal(resolved.selection.variant, "ascend_mxfp4");
  for (const role of ["prefill", "decode"]) {
    const layout = resolvePdRoleLayout({ recipe, strategy: strategies.pd_cluster, role,
      hwProfile: taxonomy.hardware_profiles.ascend_950dt, hwProfileId: "ascend_950dt",
      selection: { nodes: 5, rank: role === "prefill" ? 7 : 999, parallelism: role === "prefill" ? "dep" : "tp" } });
    assert.equal(layout.nodes, 8);
    assert.equal(layout.nodeStep, 4);
    assert.equal(layout.currentNode, 7);
    assert.equal(layout.instanceNodeRank, 3);
    assert.equal(layout.instanceStart, 5);
    assert.deepEqual(resolved.selection.pd[role], { nodes: layout.nodes, rank: layout.currentNode, parallelism: layout.parallelism });
    assert.equal(resolved.rendering[role].nodes, 8);
    const argv = resolved.rendering[role].argv;
    assert.equal(flag(argv, role === "prefill" ? "--node-rank" : "--data-parallel-rank"), "3");
    assert.equal(flag(argv, role === "prefill" ? "--master-addr" : "--data-parallel-address"), `$${role.toUpperCase()}_NODE_5`);
  }
  for (const param of ["prefill_nodes", "decode_nodes", "prefill_mode", "decode_mode", "decode_rank"]) {
    assert.ok(resolved.selection.adjusted.some((a) => a.param === param), param);
  }
  const markdown = selectionMarkdown(recipe, resolved, { strategies, taxonomy, urls });
  assert.match(markdown, /--dp-rank-start 3/);
  assert.match(markdown, /\$PREFILL_NODE_5/);
  assert.match(markdown, /Model weights:.*modelscope\.cn/);
});

test("ordinary GPU colocated pools retain zero-node half-GPU semantics", () => {
  const recipe = createAscendPdRecipe();
  const layout = resolvePdRoleLayout({ recipe, strategy: strategies.pd_cluster, role: "prefill",
    hwProfile: taxonomy.hardware_profiles.h200, hwProfileId: "h200", selection: { nodes: 0, rank: 7, parallelism: "tp" } });
  assert.equal(layout.nodes, 0);
  assert.equal(layout.poolGpus, 4);
  assert.equal(layout.tp, 4);
  assert.equal(layout.currentNode, 0);
  assert.equal(layout.nodeStep, 1);
});

test("Ascend proxy port binding agrees in command, argv, Docker and public renderings", () => {
  const recipe = createAscendPdRecipe();
  const resolved = select(recipe, pageQuery);
  const bound = bindRouterPort({ ...resolved.rendering.router,
    command: "python proxy --port $ROUTER_PORT", argv: ["python", "proxy", "--port", "$ROUTER_PORT"], dockerCommand: "docker run image --port $ROUTER_PORT" }, "18000");
  assert.equal(bound.port, 18000);
  assert.match(bound.command, /--port 18000/);
  assert.equal(flag(bound.argv, "--port"), "18000");
  assert.match(bound.dockerCommand, /--port 18000/);
  const render = (routerPort) => renderCommand(recipe, "ascend_mxfp4", "pd_cluster", "ascend_950dt", 1, [], strategies, taxonomy,
    null, null, undefined, undefined, true, { routerPort });
  for (const [requested, expected] of [[undefined, 8000], [18000, 18000]]) {
    const router = render(requested).router;
    assert.equal(router.port, expected);
    assert.equal(flag(router.argv, "--port"), String(expected));
    for (const text of [router.command, router.dockerCommand]) assert.match(text, new RegExp(`--port ${expected}\\b`));
  }
  for (const port of [0, -1, 65536, 1.2, "invalid", Infinity]) assert.throws(() => bindRouterPort(bound, port), /1 to 65535/);
});

test("hardware defaults apply before compatibility, while explicit variants report a hardware fallback", () => {
  const recipe = createAscendPdRecipe();
  assert.equal(initialVariantKey(recipe, taxonomy.hardware_profiles, "ascend_950dt"), "ascend_mxfp4");
  assert.equal(select(recipe, { hardware: "ascend_950dt" }).selection.variant, "ascend_mxfp4");
  assert.equal(initialVariantKey(recipe, taxonomy.hardware_profiles, "ascend_950dt", "fp8"), "fp8");
  assert.equal(isVariantHardwareSupported(recipe.variants.fp8, "ascend_950dt"), false);
  assert.equal(buildVariantRendering(recipe, "fp8", "ascend_950dt", strategies, taxonomy), null);
  const explicit = select(recipe, { hardware: "ascend_950dt", variant: "fp8" });
  assert.equal(explicit.selection.variant, "fp8");
  assert.notEqual(explicit.selection.hardware, "ascend_950dt");
  assert.ok(explicit.selection.adjusted.some((a) => a.param === "hardware"));
});

test("variant exclusions and defaults stay hardware-scoped and survive overlay round trips", () => {
  const ctx = resolveScope({ brands: ["Huawei"] }, taxonomy);
  const core = createAscendPdRecipe();
  delete core.meta.default_variant;
  delete core.variants.default.hardware;
  const overlay = { meta: { default_variant: { ascend_950dt: "ascend_mxfp4" } }, variants: { default: { hardware: { ascend_950dt: "unsupported" } } } };
  const merged = applyOverlay(core, overlay, { ctx, source: "ascend" });
  assert.deepEqual(merged.errors, []);
  for (const malicious of [{ meta: { default_variant: { h200: "default" } } }, { variants: { default: { hardware: { h200: "unsupported" } } } },
    { variants: { default: { model_url: "https://example.test/foreign" } } }]) {
    assert.ok(applyOverlay(core, malicious, { ctx, source: "ascend" }).errors.length);
  }
  const split = splitOverlay(merged.recipe, { ctx, source: "ascend" });
  assert.deepEqual(applyOverlay(split.core, split.overlay, { ctx, source: "ascend" }).recipe, merged.recipe);
  const validate = new Ajv({ allErrors: true, strict: false }).compile(buildRecipeSchema({ taxonomy, strategies }));
  assert.ok(validate(merged.recipe), JSON.stringify(validate.errors));
  merged.recipe.variants.default.hardware = { nvidia: "unsupported" };
  assert.equal(validate(merged.recipe), false, "variant exclusions only accept exact hardware ids");
});

test("model references inherit only for the same checkpoint, and reject unsafe or malformed URLs", () => {
  const recipe = createAscendPdRecipe();
  recipe.model.model_url = "https://modelscope.cn/models/test/Base";
  assert.equal(resolveModelReference(recipe, "default").model_url, recipe.model.model_url);
  assert.equal(resolveModelReference(recipe, "fp8").model_url, "https://huggingface.co/test/Ascend-PD-FP8");
  assert.equal(resolveModelReference(recipe, "ascend_mxfp4").provider, "ModelScope");
  recipe.variants.fp8.model_url = "https://weights.example.test/checkpoint";
  assert.equal(resolveModelReference(recipe, "fp8").provider, "weights.example.test");
  for (const value of ["javascript:alert(1)", "/relative", "https://", "ftp://example.test/model"]) {
    recipe.variants.fp8.model_url = value;
    assert.throws(() => resolveModelReference(recipe, "fp8"));
    assert.ok(validateModelReferences(recipe).some((e) => e.includes("variants.fp8.model_url")));
  }
});

test("the real static API builder uses hardware defaults and preserves promoted checkpoint URLs", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "recipes-ascend-api-"));
  try {
    for (const file of ["taxonomy.yaml", "platforms.yaml", "strategies", "kv_store"]) fs.cpSync(path.join(root, file), path.join(dir, file), { recursive: true });
    fs.mkdirSync(path.join(dir, "models", "test"), { recursive: true });
    const recipe = createAscendPdRecipe();
    recipe.model.model_url = "https://modelscope.cn/models/test/Base";
    recipe.meta.default_variant.nvidia = "fp8";
    recipe.variants.fp8.supported_hardware = ["h200"];
    const recipeFile = path.join(dir, "models/test/Ascend-PD.yaml");
    fs.writeFileSync(recipeFile, yaml.dump(recipe));
    const out = path.join(dir, "api");
    const env = { ...process.env, RECIPES_CONTENT_DIR: dir, RECIPES_PUBLIC_DIR: out };
    execFileSync(process.execPath, ["scripts/build-recipes-api.mjs"], { cwd: root, env, stdio: "pipe", timeout: 60000 });
    const api = (file) => JSON.parse(fs.readFileSync(path.join(out, file), "utf8"));
    assert.equal(api("test/Ascend-PD/hw/ascend_950dt.json").variant, "ascend_mxfp4");
    assert.equal(api("test/Ascend-PD/hw/h200.json").variant, "fp8");
    assert.equal(api("test/Ascend-PD/hw/b200.json").variant, "default", "a broad preference is ignored on incompatible profiles");
    const promoted = api("test/Ascend-PD-MXFP4.json");
    assert.equal(promoted.model.model_id, recipe.variants.ascend_mxfp4.model_id);
    assert.equal(promoted.model.model_url, recipe.variants.ascend_mxfp4.model_url);
    assert.equal(api("test/Ascend-PD-FP8.json").model.model_url, "https://huggingface.co/test/Ascend-PD-FP8");
    assert.equal(fs.existsSync(path.join(out, "test/Ascend-PD-FP8/hw/ascend_950dt.json")), false);
    recipe.meta.default_variant.ascend_950dt = "fp8";
    fs.writeFileSync(recipeFile, yaml.dump(recipe));
    const invalidDefault = spawnSync(process.execPath, ["scripts/build-recipes-api.mjs"], { cwd: root, env, encoding: "utf8", timeout: 60000 });
    assert.notEqual(invalidDefault.status, 0);
    assert.match(invalidDefault.stdout + invalidDefault.stderr, /meta\.default_variant\.ascend_950dt/);
    recipe.meta.default_variant.ascend_950dt = "ascend_mxfp4";
    recipe.variants.fp8.model_url = "https://";
    fs.writeFileSync(recipeFile, yaml.dump(recipe));
    const invalid = spawnSync(process.execPath, ["scripts/build-recipes-api.mjs"], { cwd: root, env, encoding: "utf8", timeout: 60000 });
    assert.notEqual(invalid.status, 0);
    assert.match(invalid.stdout + invalid.stderr, /variants\.fp8\.model_url/);
  } finally {
    assert.equal(path.dirname(dir), path.resolve(os.tmpdir()));
    assert.ok(path.basename(dir).startsWith("recipes-ascend-api-"));
    fs.rmSync(dir, { recursive: true, force: true });
  }
});
