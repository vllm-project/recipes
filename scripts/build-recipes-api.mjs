/**
 * Build-time script: reads YAML recipe/strategy/taxonomy files,
 * writes static JSON files for the API.
 *
 * API URLs (friendly, no /api/ prefix):
 *   /models.json                                — all recipes index
 *   /{hf_id}.json                               — single recipe (default-hw rendering)
 *   /{hf_id}/strategies/{strategy}.json         — default-hw alternative (per strategy)
 *   /{hf_id}/hw/{hw}.json                       — per-hardware rendering
 *   /{hf_id}/hw/{hw}/strategies/{strategy}.json — per-(hw, strategy) alternative
 *   /{hf_id}/presets/{preset}.json              — a tuned preset, one rendering per point
 *   /strategies.json                            — all strategies
 *   /strategies/{strategy}.json                 — single strategy spec
 *   /taxonomy.json                              — controlled vocabulary
 *   /quickstart/8x-h100.json                    — hardware quickstart
 *
 * Rendered on demand, not written here (src/app/api/render/):
 *   /{hf_id}.md?{page query}                    — a page URL's configuration as Markdown
 *   /{hf_id}.json?{page query}                  — the same as one JSON rendering
 *
 * Run: node scripts/build-recipes-api.mjs
 */

import fs from "fs";
import path from "path";
import yaml from "js-yaml";
import Ajv from "ajv";
import { buildRecipeSchema } from "../src/lib/recipe-schema.js";
import { BASELINE, effectivePreset, presetImage, presetPoints, presetSelect, resolvePresetKey, validatePresets, withPreset } from "../src/lib/recipe-presets.js";
import { validateContextEntries } from "../src/lib/recipe-context.js";
import {
  pickDefaultHardware,
  listCompatibleHardware,
  recommendStrategy,
  fitsSingleNode,
  variantVramMinimumGb,
  isHardwareScalable,
  isKvStoreBrandSupported,
  strategyAllowsKvOffload,
  pickFittingVariant,
  pdFitsSingleNode,
  pdRoleNodes,
  pdRoleParallelism,
  resolveCommand,
  computeDockerMeta,
  buildDockerRun,
  buildDockerArgv,
  nodesForStrategy,
  isStrategyReachable,
  isStrategySupportedOnHardware,
  effectiveCompatibleStrategies,
  FALLBACK_FRONTEND,
} from "../src/lib/command-synthesis.js";
import { contentChannel, contentDir, loadProvenance } from "../src/lib/content-root.js";
import { buildVariantRendering, defaultFeaturesFor, pickPdNodes, renderCommand, synthesizeInstall } from "../src/lib/recipe-render.js";
import { assembleChannel } from "./assemble-content.mjs";
import { loadOverrides } from "./lib/content-sources.mjs";

const ROOT = process.cwd();
// RECIPES_PUBLIC_DIR redirects the output (validate-source.mjs builds into a
// temp dir so a check never clobbers the dev site's public/).
const PUBLIC = process.env.RECIPES_PUBLIC_DIR ? path.resolve(process.env.RECIPES_PUBLIC_DIR) : path.join(ROOT, "public");

// Assemble the channel's sources into .content/<channel>/ first (cheap and
// local — remote sources must already be fetched by sync-sources.mjs), so a
// new or edited YAML in any source is picked up. RECIPES_CONTENT_DIR points at
// a prebuilt tree instead.
if (!process.env.RECIPES_CONTENT_DIR) {
  assembleChannel(contentChannel(), { overrides: loadOverrides() });
}
const CONTENT = contentDir();
const provenance = loadProvenance();

// Report a recipe by the path in its source repo, not its .content/ alias.
function sourceRel(file) {
  const hfId = path.relative(path.join(CONTENT, "models"), file).replace(/\.(yaml|yml)$/, "").split(path.sep).join("/");
  const p = provenance[hfId];
  return p ? (p.repo ? `${p.repo}:${p.path}` : p.path) : path.relative(ROOT, file);
}

function readYaml(filePath) {
  return yaml.load(fs.readFileSync(filePath, "utf8"), { filename: path.relative(ROOT, filePath) });
}

function writeJson(relPath, data) {
  const fullPath = path.join(PUBLIC, relPath);
  fs.mkdirSync(path.dirname(fullPath), { recursive: true });
  fs.writeFileSync(fullPath, JSON.stringify(data, null, 2));
}

// Normalize dates (js-yaml parses YYYY-MM-DD as Date objects)
function normalizeDates(obj) {
  if (obj instanceof Date) return obj.toISOString().split("T")[0];
  if (Array.isArray(obj)) return obj.map(normalizeDates);
  if (obj && typeof obj === "object") {
    return Object.fromEntries(Object.entries(obj).map(([k, v]) => [k, normalizeDates(v)]));
  }
  return obj;
}


// Validate the feature/mode cross-references before synthesis. Modes are
// intentionally a strict alternative to a feature's flat `args`: accepting
// both makes one silently shadow the other. Every checkpoint must retain at
// least one usable mode so enabling the parent feature can never be a no-op.
function validateFeatureModes(recipe, sourceFile) {
  const errors = [];
  const features = recipe.features || {};
  const variants = recipe.variants || {};

  for (const [featureKey, feature] of Object.entries(features)) {
    if (feature?.modes === undefined) continue;
    const modes = feature.modes;
    const modeKeys = modes && typeof modes === "object" && !Array.isArray(modes)
      ? Object.keys(modes)
      : [];

    if (modeKeys.length === 0) {
      errors.push(`features.${featureKey}.modes must be a non-empty map`);
      continue;
    }
    if (feature.args !== undefined) {
      errors.push(`features.${featureKey} cannot declare both args and modes`);
    }
    if (feature.default_mode && !modeKeys.includes(feature.default_mode)) {
      errors.push(`features.${featureKey}.default_mode references unknown mode ${feature.default_mode}`);
    }

    for (const [modeKey, mode] of Object.entries(modes)) {
      if (!Array.isArray(mode?.args)) {
        errors.push(`features.${featureKey}.modes.${modeKey}.args must be an array`);
      }
      if (mode?.variants !== undefined && !Array.isArray(mode.variants)) {
        errors.push(`features.${featureKey}.modes.${modeKey}.variants must be an array`);
      }
      const modeVariants = Array.isArray(mode?.variants) ? mode.variants : [];
      for (const variantKey of modeVariants) {
        if (!Object.hasOwn(variants, variantKey)) {
          errors.push(`features.${featureKey}.modes.${modeKey}.variants references unknown variant ${variantKey}`);
        }
      }
    }

    for (const variantKey of Object.keys(variants)) {
      const available = modeKeys.some((modeKey) => {
        const allow = modes[modeKey]?.variants;
        return !Array.isArray(allow) || allow.length === 0 || allow.includes(variantKey);
      });
      if (!available) {
        errors.push(`features.${featureKey} has no mode available for variant ${variantKey}`);
      }
    }
  }

  for (const [variantKey, variant] of Object.entries(variants)) {
    for (const [featureKey, modeKey] of Object.entries(variant?.default_modes || {})) {
      const feature = features[featureKey];
      const mode = feature?.modes?.[modeKey];
      if (!feature) {
        errors.push(`variants.${variantKey}.default_modes references unknown feature ${featureKey}`);
      } else if (!feature.modes) {
        errors.push(`variants.${variantKey}.default_modes references non-modal feature ${featureKey}`);
      } else if (!mode) {
        errors.push(`variants.${variantKey}.default_modes.${featureKey} references unknown mode ${modeKey}`);
      } else if (Array.isArray(mode.variants) && mode.variants.length > 0 && !mode.variants.includes(variantKey)) {
        errors.push(`variants.${variantKey}.default_modes.${featureKey} selects mode ${modeKey}, which is unavailable for that variant`);
      }
    }
  }

  if (errors.length) {
    const rel = sourceRel(sourceFile);
    throw new Error(`Invalid feature modes in ${rel}:\n  - ${errors.join("\n  - ")}`);
  }
}

// Frontend keys under `model` are matched by exact id against
// `taxonomy.frontends`, so a typo would silently keep a broken frontend
// selectable (or default to one that doesn't exist). `default_frontend` must
// name a known frontend; `<id>_frontend` may only opt a non-baseline frontend
// out (`unsupported`), and never the one the recipe defaults to.
function validateFrontends(recipe, sourceFile, taxonomy) {
  const errors = [];
  const known = taxonomy.frontends || {};
  const model = recipe.model || {};
  const def = model.default_frontend;
  if (def !== undefined && !known[def]) {
    errors.push(`model.default_frontend: unknown frontend "${def}" (known: ${Object.keys(known).join(", ")})`);
  }
  for (const [key, value] of Object.entries(model)) {
    const m = key.match(/^(.+)_frontend$/);
    if (!m || key === "default_frontend") continue;
    const id = m[1];
    if (!known[id]) {
      errors.push(`model.${key}: unknown frontend "${id}" (known: ${Object.keys(known).join(", ")})`);
    } else if (id === FALLBACK_FRONTEND) {
      errors.push(`model.${key}: the ${id} frontend is the fallback and can't be marked unsupported`);
    } else if (value !== "unsupported") {
      errors.push(`model.${key} must be "unsupported" (omit the key when the frontend works)`);
    } else if (def === id) {
      errors.push(`model.default_frontend is "${id}", but model.${key} marks it unsupported`);
    }
  }
  if (errors.length) {
    const rel = sourceRel(sourceFile);
    throw new Error(`Invalid frontend config in ${rel}:\n  - ${errors.join("\n  - ")}`);
  }
}

// Validate every hardware-keyed map against the taxonomy vocabulary. Hardware
// lookups are exact string matches (`isHardwareSupported`, `hardwareKeyedValue`),
// so a key naming no known GPU is silently inert: an author's `verified` badge
// never renders, and — worse — an intended `unsupported` opt-out never disables
// anything. Nothing downstream can detect that, hence the build-time check.
//
// Three keyspaces, matching the schema in CLAUDE.md:
//   - exact:   only a taxonomy profile id (`meta.hardware`)
//   - variant selectors: profile id or `arch:<compute_arch>`
//   - layered: profile id > generation > brand > `default` (`strategy_hardware`,
//              variant `tp`, `hardware_overrides`)
function validateHardwareKeys(recipe, sourceFile, taxonomy, strategies) {
  const errors = [];
  const profiles = taxonomy?.hardware_profiles || {};
  const ids = new Set(Object.keys(profiles));
  const generations = new Set();
  const brands = new Set();
  const computeArchs = new Set();
  for (const p of Object.values(profiles)) {
    if (p?.generation) generations.add(p.generation);
    if (p?.brand) brands.add(String(p.brand).toLowerCase());
    if (p?.compute_arch) computeArchs.add(p.compute_arch);
  }
  const layered = new Set([...ids, ...generations, ...brands, "default"]);

  const exact = (loc, key) => {
    if (!ids.has(key)) errors.push(`${loc} references unknown hardware profile ${key}`);
  };
  const keyed = (loc, key) => {
    if (!layered.has(key)) errors.push(`${loc} references unknown hardware key ${key}`);
  };
  const keysOf = (o) => (o && typeof o === "object" && !Array.isArray(o) ? Object.keys(o) : []);

  if (recipe?.meta?.default_hardware) exact("meta.default_hardware", recipe.meta.default_hardware);
  for (const k of keysOf(recipe?.meta?.hardware)) exact("meta.hardware", k);

  for (const [kvId, map] of Object.entries(recipe?.kv_cache_strategy_hardware || {})) {
    for (const k of keysOf(map)) exact(`kv_cache_strategy_hardware.${kvId}`, k);
  }

  for (const [sId, map] of Object.entries(recipe?.strategy_hardware || {})) {
    for (const k of keysOf(map)) keyed(`strategy_hardware.${sId}`, k);
  }

  for (const k of keysOf(recipe?.hardware_overrides)) keyed("hardware_overrides", k);

  // `strategy_min_gpus` mixes strategy ids with an exact-GPU-id keyspace; only
  // the latter is ours to check.
  for (const k of keysOf(recipe?.strategy_min_gpus)) {
    if (!strategies?.[k]) exact("strategy_min_gpus", k);
  }

  for (const [variantKey, variant] of Object.entries(recipe?.variants || {})) {
    const at = `variants.${variantKey}`;
    for (const k of variant?.supported_hardware || []) {
      if (k.startsWith("arch:")) {
        if (!computeArchs.has(k.slice(5))) errors.push(`${at}.supported_hardware references unknown compute architecture ${k}`);
      } else {
        exact(`${at}.supported_hardware`, k);
      }
    }
    for (const k of keysOf(variant?.hardware_overrides)) keyed(`${at}.hardware_overrides`, k);
    if (variant?.tp && typeof variant.tp === "object") {
      for (const k of keysOf(variant.tp)) keyed(`${at}.tp`, k);
    }
    // `precision_hardware` widens a precision's hardware constraint by value,
    // not by key — check the vocabulary it names.
    const ph = variant?.precision_hardware;
    if (ph?.generation && !generations.has(ph.generation)) {
      errors.push(`${at}.precision_hardware.generation references unknown generation ${ph.generation}`);
    }
    if (ph?.brand && !brands.has(String(ph.brand).toLowerCase())) {
      errors.push(`${at}.precision_hardware.brand references unknown brand ${ph.brand}`);
    }
  }

  for (const [featureKey, feature] of Object.entries(recipe?.features || {})) {
    for (const k of keysOf(feature?.hardware_overrides)) keyed(`features.${featureKey}.hardware_overrides`, k);
    for (const [modeKey, mode] of Object.entries(feature?.modes || {})) {
      const at = `features.${featureKey}.modes.${modeKey}`;
      for (const k of keysOf(mode?.hardware)) keyed(`${at}.hardware`, k);
      for (const k of keysOf(mode?.hardware_overrides)) keyed(`${at}.hardware_overrides`, k);
    }
  }

  if (errors.length) {
    const rel = sourceRel(sourceFile);
    throw new Error(`Invalid hardware keys in ${rel}:\n  - ${errors.join("\n  - ")}`);
  }
}


// Recursively find all .yaml files under a directory
function findYamlFiles(dir) {
  const results = [];
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      results.push(...findYamlFiles(full));
    } else if (entry.name.endsWith(".yaml") || entry.name.endsWith(".yml")) {
      results.push(full);
    }
  }
  return results;
}

// ── Taxonomy ──
const taxonomy = normalizeDates(readYaml(path.join(CONTENT, "taxonomy.yaml")));
writeJson("taxonomy.json", taxonomy);

// ── Platforms ── (third-party self-host targets; flat catalog)
const platformsFile = path.join(CONTENT, "platforms.yaml");
let platformsCount = 0;
if (fs.existsSync(platformsFile)) {
  const platforms = normalizeDates(readYaml(platformsFile));
  writeJson("platforms.json", platforms);
  platformsCount = Array.isArray(platforms?.platforms) ? platforms.platforms.length : 0;
}

// ── Strategies ──
const strategiesDir = path.join(CONTENT, "strategies");
const strategies = {};
for (const file of fs.readdirSync(strategiesDir).filter((f) => f.endsWith(".yaml"))) {
  const s = normalizeDates(readYaml(path.join(strategiesDir, file)));
  strategies[s.name] = s;
  writeJson(`strategies/${s.name}.json`, s);
}
writeJson("strategies.json", strategies);
const servingStrategyCount = Object.keys(strategies).length;

// ── KV-store deployments ── (kv_store/*.yaml, deploy_type: kv_store_lb)
// They power the KV Offload row, not the Strategy row, so they publish under
// /kv_store/ and stay out of /strategies.json — but merge into the same
// in-memory map since renderCommand looks every deployment spec up by id.
const kvStoreDir = path.join(CONTENT, "kv_store");
const kvStoreDeployments = {};
for (const file of fs.readdirSync(kvStoreDir).filter((f) => f.endsWith(".yaml"))) {
  const s = normalizeDates(readYaml(path.join(kvStoreDir, file)));
  kvStoreDeployments[s.name] = s;
  strategies[s.name] = s;
  writeJson(`kv_store/${s.name}.json`, s);
}
writeJson("kv_store.json", kvStoreDeployments);

// ── Recipe schema ── generated from the taxonomy + strategies, published at
// /recipe.schema.json for editors, and enforced on every recipe below.
const recipeSchema = buildRecipeSchema({ taxonomy, strategies });
writeJson("recipe.schema.json", recipeSchema);
const validateRecipe = new Ajv({ allErrors: true, strict: false }).compile(recipeSchema);

// One line per problem. anyOf branches report every alternative they tried;
// keep the deepest, most specific message for each location.
function schemaErrors(errors) {
  // A bad map key fails `propertyNames` and also each alternative inside it
  // (enum / pattern, reported with propertyName set); keep only the summary.
  const badKeys = new Set(errors.filter((e) => e.keyword === "propertyNames")
    .map((e) => `${e.instancePath}|${e.params.propertyName}`));
  const out = new Set();
  for (const e of errors) {
    if (e.keyword === "anyOf" || e.keyword === "if") continue;
    if (e.keyword !== "propertyNames" && e.propertyName !== undefined
      && badKeys.has(`${e.instancePath}|${e.propertyName}`)) continue;
    const at = e.instancePath || "/";
    if (e.keyword === "additionalProperties") out.add(`unknown key "${e.params.additionalProperty}" at ${at}`);
    else if (e.keyword === "propertyNames") out.add(`key "${e.params.propertyName}" is not valid at ${at}`);
    else if (e.keyword === "enum") out.add(`${at} must be one of: ${e.params.allowedValues.slice(0, 12).join(", ")}${e.params.allowedValues.length > 12 ? ", …" : ""}`);
    else out.add(`${at} ${e.message}`);
  }
  return [...out];
}

// ── Recipes ── (walks models/<hf_org>/<hf_repo>.yaml)
const modelsDir = path.join(CONTENT, "models");

// Pre-scan: collect every recipe's hf_id so variant-promotion can detect
// collisions with standalone YAMLs (e.g. inclusionAI/Ring-1T-FP8.yaml exists
// as its own recipe, so a sibling recipe must not also promote a "Ring-1T-FP8"
// variant).
const allRecipeHfIds = new Set();
const schemaFailures = [];
for (const file of findYamlFiles(modelsDir)) {
  if (!validateRecipe(normalizeDates(readYaml(file)))) {
    schemaFailures.push(`${sourceRel(file)}:\n    - ${schemaErrors(validateRecipe.errors).join("\n    - ")}`);
  }
  const rel = path.relative(modelsDir, file);
  const parts = rel.split(path.sep);
  if (parts.length >= 2) {
    const org = parts[0];
    const repo = parts[parts.length - 1].replace(/\.(yaml|yml)$/, "");
    allRecipeHfIds.add(`${org}/${repo}`);
  }
}
if (schemaFailures.length) {
  throw new Error(`Recipes don't match the schema (SCHEMA.md, /recipe.schema.json):\n  ${schemaFailures.join("\n  ")}`);
}

// Write the public recipe JSON (parent or promoted) with HEAD_KEYS ordering.
function writeRecipeJson(hfId, payload) {
  const HEAD_KEYS = ["hf_id", "meta", "recommended_command"];
  const ordered = {};
  for (const k of HEAD_KEYS) if (k in payload) ordered[k] = payload[k];
  for (const [k, v] of Object.entries(payload)) if (!HEAD_KEYS.includes(k)) ordered[k] = v;
  writeJson(`${hfId}.json`, ordered);
  return ordered;
}

// Render one variant's full hardware × strategy matrix and write all the
// per-resource JSON files. Returns the parent's `recommended_command` object
// (the default-hardware rendering with `alternatives` path map + `by_hardware`
// path map) or null if the variant can't be rendered on its default hardware.
//
// `altBaseHfId` is the base directory for written files — the parent recipe's
// hf_id, or a promoted variant's hf_id (e.g. zai-org/GLM-5.1-FP8). Output:
//
//   /<base>.json                          (parent — written by caller)
//   /<base>/strategies/<s>.json           (default-hw alternatives — legacy path)
//   /<base>/hw/<hw>.json                  (per-hw recommended + alternatives index)
//   /<base>/hw/<hw>/strategies/<s>.json   (per-(hw, strategy) — non-default hw only)
//
// The default-hw entry in `by_hardware` re-uses the legacy `/strategies/`
// paths to avoid duplicating identical files; non-default hw entries each get
// their own `/hw/<hw>/strategies/` subtree.
function renderAndWriteVariant(recipe, variantKey, altBaseHfId, strategies, taxonomy) {
  const variant = recipe.variants?.[variantKey];
  if (!variant) return null;
  if ((recipe.meta?.tasks || []).includes("omni")) return null;

  // Hardware/strategy compatibility can shrink between recipe revisions.
  // Clear this variant's generated subtree so removed targets do not survive
  // as stale, directly-addressable JSON files from an earlier build.
  fs.rmSync(path.join(PUBLIC, altBaseHfId), { recursive: true, force: true });

  const hwProfiles = taxonomy.hardware_profiles || {};
  const defaultHw = pickDefaultHardware(hwProfiles, variant, recipe);
  // Mirror the UI rule: `restricted` profiles (TPU, etc.) only surface for
  // recipes that explicitly opt in via `meta.hardware.<id>`. Otherwise we'd
  // emit speculative TPU commands for every NVIDIA recipe.
  const declaredHw = recipe.meta?.hardware || {};
  const compatibleHw = listCompatibleHardware(hwProfiles, variant, recipe)
    .filter((id) => !hwProfiles[id]?.restricted || id in declaredHw);
  // Ensure default is first; de-dupe.
  const hwIds = Array.from(new Set([defaultHw, ...compatibleHw]));

  // Render every (variant × hardware) combo up front.
  const renderedByHw = {};
  for (const hwId of hwIds) {
    const built = buildVariantRendering(recipe, variantKey, hwId, strategies, taxonomy);
    if (built) renderedByHw[hwId] = built;
  }
  if (!renderedByHw[defaultHw]) return null;

  const byHardware = {};
  for (const [hwId, { recommended, alternatives }] of Object.entries(renderedByHw)) {
    const isDefault = hwId === defaultHw;
    const stratDir = isDefault
      ? `${altBaseHfId}/strategies`
      : `${altBaseHfId}/hw/${hwId}/strategies`;
    const altLinks = {};
    for (const [s, rendered] of Object.entries(alternatives)) {
      const altPath = `${stratDir}/${s}.json`;
      writeJson(altPath, rendered);
      altLinks[s] = `/${altPath}`;
    }
    if (Object.keys(altLinks).length) recommended.alternatives = altLinks;

    const hwPath = `${altBaseHfId}/hw/${hwId}.json`;
    writeJson(hwPath, recommended);
    byHardware[hwId] = `/${hwPath}`;
  }

  // Parent's recommended_command = default-hw rendering + by_hardware index.
  // Shallow clone so the on-disk per-hw default file stays clean of the index.
  const defaultRecommended = { ...renderedByHw[defaultHw].recommended, by_hardware: byHardware };
  return defaultRecommended;
}

// Every tuned preset rendered at each of its points, so agents can read the
// exact commands of a non-default preset or point without the builder:
//   /<hf_id>/presets/<key>.json = { preset meta, points: { <point|default>: rendering } }
// Variant: the preset's first listed one, else `default`. Strategy: the
// preset's, else the recommendation for that GPU.
function writePresetRenderings(recipe, hfId, strategies, taxonomy) {
  const written = {};
  for (const [key, p] of Object.entries(recipe.presets || {})) {
    const hwId = p.hardware;
    const hwProfile = taxonomy.hardware_profiles?.[hwId] || {};
    const variantKey = p.variants?.[0] || "default";
    const variant = recipe.variants?.[variantKey];
    if (!variant) continue;
    const strategy = p.strategy || recommendStrategy(recipe, hwProfile, fitsSingleNode(hwProfile, variant, hwId) ? 1 : 2);
    const deploy = strategies[strategy]?.deploy_type;
    const feats = defaultFeaturesFor(recipe, hwId, variantKey);
    const points = presetPoints(p);
    const out = {};
    for (const pt of points.length ? points : [null]) {
      let nc = 1;
      let pdNodes = null;
      if (deploy === "pd_cluster") {
        pdNodes = pickPdNodes(hwProfile, variant, recipe, strategies, hwId, key, pt?.key, variantKey);
        if (pdNodes === "skip") continue;
      } else if (deploy === "multi_node") {
        nc = Math.max(2, nodesForStrategy(recipe, strategy, hwProfile.gpu_count, hwId));
      }
      const rendered = renderCommand(recipe, variantKey, strategy, hwId, nc, feats, strategies, taxonomy, pdNodes, null, key, pt?.key, true);
      if (!rendered) continue;
      out[pt?.key || "default"] = { ...(pt ? { label: pt.label, ...(pt.concurrency ? { concurrency: pt.concurrency } : {}) } : {}), ...rendered };
    }
    if (!Object.keys(out).length) continue;
    const rel = `${hfId}/presets/${key}.json`;
    writeJson(rel, {
      preset: key,
      label: p.label,
      hardware: hwId,
      strategy,
      variant: variantKey,
      ...(p.workload ? { workload: p.workload } : {}),
      default: !!p.default,
      ...(p.source ? { source: p.source } : {}),
      ...(p.image ? { image: presetImage(p.image) } : {}),
      ...(points.length ? { default_point: (points.find((x) => x.default) || points[0]).key } : {}),
      points: out,
    });
    written[key] = `/${rel}`;
  }
  return written;
}

// KV-store deployment ids a preset's `select.kv_offload` may name.
const kvStoreIds = Object.keys(strategies).filter((id) => strategies[id]?.deploy_type === "kv_store_lb");

const recipes = [];
const promotedIndexEntries = [];
let promotedCount = 0;
let collisionCount = 0;

for (const file of findYamlFiles(modelsDir)) {
  const r = normalizeDates(readYaml(file));
  validateFeatureModes(r, file);
  validateFrontends(r, file, taxonomy);
  validateHardwareKeys(r, file, taxonomy, strategies);
  const presetErrors = validatePresets(r, taxonomy, effectiveCompatibleStrategies(r), kvStoreIds, strategies.pd_cluster?.connectors);
  if (presetErrors.length) throw new Error(`Invalid presets in ${sourceRel(file)}:\n  - ${presetErrors.join("\n  - ")}`);
  const contextErrors = validateContextEntries(r);
  if (contextErrors.length) throw new Error(`Invalid notes/troubleshooting in ${sourceRel(file)}:\n  - ${contextErrors.join("\n  - ")}`);
  // Derive HF identity from path. Only `hf_id` is exposed in the public JSON;
  // `org` and `repo` are trivially `hf_id.split("/")` for consumers.
  const rel = path.relative(modelsDir, file);
  const parts = rel.split(path.sep);
  let hfOrg = "";
  let hfRepo = "";
  if (parts.length >= 2) {
    hfOrg = parts[0];
    hfRepo = parts[parts.length - 1].replace(/\.(yaml|yml)$/, "");
    r.hf_id = `${hfOrg}/${hfRepo}`;
  }
  // Pre-render the canonical deploy command (default variant) so agents don't
  // have to reimplement command-synthesis. Mirrors the website's default
  // selections.
  const parentHfId = `${hfOrg}/${hfRepo}`;
  const defaultRecommended = renderAndWriteVariant(r, "default", parentHfId, strategies, taxonomy);
  if (defaultRecommended) r.recommended_command = defaultRecommended;
  // Each preset at each point, written into the subtree rebuilt above, with a
  // `json:` pointer on the recipe's own preset entry.
  for (const [key, link] of Object.entries(writePresetRenderings(r, parentHfId, strategies, taxonomy))) {
    r.presets[key] = { ...r.presets[key], json: link };
  }

  // Promote each variant whose `model_id` points at a distinct HF repo to its
  // own top-level JSON endpoint, mirroring the HF URL convention. `default` is
  // included: a recipe may default to a differently-named checkpoint (this
  // recipe's own id stays the parent path, rendered above), and that HF id
  // needs an endpoint too — a default pointing back at the parent id is caught
  // by the collision check below. Renderings for promoted variants are gathered
  // here and written after the parent JSON so we can store `json:` pointers in
  // parent.variants.<v>.
  const promotedRenderings = [];  // { variantKey, variantHfId, recommended }
  for (const [variantKey, variantCfg] of Object.entries(r.variants || {})) {
    const variantModelId = variantCfg?.model_id;
    if (!variantModelId || typeof variantModelId !== "string" || !variantModelId.includes("/")) continue;
    if (allRecipeHfIds.has(variantModelId)) {
      // A standalone recipe at models/<variantModelId>.yaml exists — let it win.
      console.warn(`  ⚠ skipping variant promotion: ${variantModelId} (variant of ${parentHfId}) collides with standalone recipe`);
      collisionCount++;
      continue;
    }
    const variantRecommended = renderAndWriteVariant(r, variantKey, variantModelId, strategies, taxonomy);
    if (!variantRecommended) continue;
    promotedRenderings.push({
      variantKey, variantModelId, recommended: variantRecommended,
      install: variantCfg.install ? synthesizeInstall(r, variantCfg) : undefined,
    });
  }

  // Annotate parent variants with `json:` pointers so consumers can navigate.
  if (r.variants) {
    for (const { variantKey, variantModelId } of promotedRenderings) {
      r.variants[variantKey] = { ...r.variants[variantKey], json: `/${variantModelId}.json` };
    }
  }

  // All renderings need the raw install toggles; serialize them only afterward.
  // Replace `model.install` with the synthesized commands so
  // JSON consumers see the rendered one-liners (pip + docker) plus any
  // `extras` from `dependencies`. The raw toggle config is internal-only.
  // `dependencies` is the YAML-authored source — drop it from the public JSON
  // since `model.install.extras` already carries the same data verbatim.
  const install = synthesizeInstall(r);
  if (r.model) {
    const { install: _drop, ...rest } = r.model;
    r.model = install ? { ...rest, install } : rest;
  } else if (install) {
    r.model = { install };
  }
  delete r.dependencies;

  // strategy_overrides, compatible_strategies, default_strategy are synthesis
  // inputs whose effects are already baked into recommended_command + the
  // per-strategy alternative files (whose keys are the compatible_strategies
  // set; the recommended one comes from default_strategy). Drop AFTER all
  // renderings (parent + promoted) have run — buildVariantRendering reads
  // them. The YAML on GitHub is the source of truth for anyone re-synthesizing.
  delete r.strategy_overrides;
  delete r.compatible_strategies;
  delete r.default_strategy;
  // Which repo owns this recipe (and which overlays merged into it), so agents
  // can trace a command back to the YAML that produced it.
  r.source = provenance[parentHfId] || null;
  // JSON at /<org>/<repo>.json — mirrors HF URL scheme.
  const out = writeRecipeJson(parentHfId, r);
  recipes.push(out);

  // Now write the promoted-variant JSONs. Each is a self-contained recipe
  // shape with hf_id / model.model_id rewritten and `variants.default` set to
  // the variant's config (model_id stripped — it's at top-level now). The
  // sibling-variant info is lost on purpose; consumers can follow
  // `meta.derived_from` back to the parent for the full family view.
  for (const { variantKey, variantModelId, recommended, install } of promotedRenderings) {
    const variantCfg = r.variants?.[variantKey] || {};
    const { model_id: _vMid, json: _vJson, ...variantCore } = variantCfg;
    const promoted = {
      ...r,  // share meta/features/guide/etc. with the parent
      hf_id: variantModelId,
      meta: { ...r.meta, derived_from: parentHfId, variant: variantKey },
      model: {
        ...r.model,
        model_id: variantModelId,
        ...(install && { install }),
      },
      variants: { default: variantCore },
      recommended_command: recommended,
    };
    writeRecipeJson(variantModelId, promoted);
    promotedIndexEntries.push({
      hf_id: variantModelId,
      title: promoted.meta?.title,
      provider: promoted.meta?.provider,
      derived_from: parentHfId,
    });
    promotedCount++;
  }
}

// /models.json — slim discovery index (~5 KB). For agents that want to
// enumerate recipes and follow links; per-recipe data lives at
// `/<hf_id>.json` (the `json` pointer below). Promoted variants (e.g.
// zai-org/GLM-5.1-FP8) appear alongside parents so consumers can discover
// them directly.
const index = [
  ...recipes.map((r) => ({
    hf_id: r.hf_id,
    title: r.meta.title,
    provider: r.meta.provider,
    url: `/${r.hf_id}`,
    json: `/${r.hf_id}.json`,
  })),
  ...promotedIndexEntries.map((e) => ({
    hf_id: e.hf_id,
    title: e.title,
    provider: e.provider,
    url: `/${e.derived_from}`,  // site route — variant lives under the parent page
    json: `/${e.hf_id}.json`,
    derived_from: e.derived_from,
  })),
];
writeJson("models.json", index);

// ── Quickstart ──
const quickstartDir = path.join(CONTENT, "quickstart");
if (fs.existsSync(quickstartDir)) {
  for (const file of fs.readdirSync(quickstartDir).filter((f) => f.endsWith(".yaml"))) {
    const q = normalizeDates(readYaml(path.join(quickstartDir, file)));
    writeJson(`quickstart/${q.hardware_profile}.json`, q);
  }
}

const rcCount = recipes.filter((r) => r.recommended_command).length;
const altCount = recipes.reduce(
  (n, r) => n + Object.keys(r.recommended_command?.alternatives || {}).length, 0
);
const hwIndexedCount = recipes.reduce(
  (n, r) => n + Object.keys(r.recommended_command?.by_hardware || {}).length, 0
);
console.log(
  `✓ JSON API: ${recipes.length} models (${rcCount} with recommended_command, ${altCount} default-hw alternatives, ${hwIndexedCount} per-hw renderings), ${promotedCount} promoted variants, ${servingStrategyCount} strategies, ${Object.keys(kvStoreDeployments).length} kv-store deployments, ${platformsCount} platforms` +
  (collisionCount ? ` (${collisionCount} variant collision${collisionCount > 1 ? "s" : ""} skipped)` : "")
);
console.log(`  /models.json`);
console.log(`  /{hf_id}.json                                (e.g. /moonshotai/Kimi-K2.5.json)`);
console.log(`  /{hf_id}/strategies/{strategy}.json          (default-hw alternatives)`);
console.log(`  /{hf_id}/hw/{hw}.json                        (per-hardware rendering)`);
console.log(`  /{hf_id}/hw/{hw}/strategies/{strategy}.json  (per-(hw, strategy) alternatives)`);
console.log(`  /{variant_hf_id}.json                        (promoted variants, e.g. /zai-org/GLM-5.1-FP8.json)`);
console.log(`  /strategies.json`);
console.log(`  /kv_store.json`);
console.log(`  /kv_store/{id}.json                          (per-deployment spec)`);
console.log(`  /taxonomy.json`);
console.log(`  /platforms.json`);
