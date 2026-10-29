import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { test } from "node:test";
import yaml from "js-yaml";
import Ajv from "ajv";
import { pdRoleNodes, pdRoleParallelism, resolveCommand } from "../src/lib/command-synthesis.js";
import { buildRecipeSchema } from "../src/lib/recipe-schema.js";
import { contentDir } from "../src/lib/content-root.js";
import { applyOverlay, resolveScope, splitOverlay } from "../src/lib/overlay-merge.js";
import { BASELINE, applyPresetArgOps, effectivePreset, presetSelect, resolvePresetKey, validatePresets, withPreset } from "../src/lib/recipe-presets.js";

const read = (file) => yaml.load(fs.readFileSync(path.join(contentDir(), file), "utf8"));
const taxonomy = read("taxonomy.yaml");
// Serving strategies plus the KV-store deployments, merged as the build does.
const strategies = Object.fromEntries(["strategies", "kv_store"].flatMap((dir) => fs.readdirSync(path.join(contentDir(), dir))
  .filter((file) => file.endsWith(".yaml"))
  .map((file) => { const s = read(`${dir}/${file}`); return [s.name, s]; })));

function yamlFiles(dir) {
  return fs.readdirSync(dir, { withFileTypes: true }).flatMap((e) => {
    const full = path.join(dir, e.name);
    return e.isDirectory() ? yamlFiles(full) : /\.ya?ml$/.test(e.name) ? [full] : [];
  });
}

const render = (recipe, strategy, hw, preset) => {
  const r = resolveCommand(recipe, "default", strategy, hw, [], strategies, taxonomy,
    [], 1, null, {}, null, null, undefined, null, "http", null, preset);
  if (!r) return r;
  const { preset: _p, ...rest } = r;
  return JSON.parse(JSON.stringify(rest));
};

// Move every exact-GPU strategy block (and its role blocks) into a
// `default: true` preset: the shape a recipe migrates to.
function migrate(recipe) {
  const out = structuredClone(recipe);
  const moved = [];
  for (const [s, so] of Object.entries(out.strategy_overrides || {})) {
    const ids = new Set([
      ...Object.keys(so.hardware_overrides || {}),
      ...["prefill", "decode"].flatMap((r) => [
        ...Object.keys(so[r]?.hardware_overrides || {}),
        ...["nodes", "parallelism"].flatMap((k) => (so[r]?.[k] && typeof so[r][k] === "object" ? Object.keys(so[r][k]) : [])),
      ]),
    ].filter((k) => taxonomy.hardware_profiles[k]));
    for (const hw of ids) {
      const p = { label: `${hw} ${s}`, hardware: hw, strategy: s, default: true };
      const ho = so.hardware_overrides?.[hw];
      if (ho?.extra_args) p.args = ho.extra_args;
      if (ho?.extra_env) p.env = ho.extra_env;
      delete so.hardware_overrides?.[hw];
      for (const role of ["prefill", "decode"]) {
        const rho = so[role]?.hardware_overrides?.[hw];
        const r = {};
        if (rho?.extra_args) r.args = rho.extra_args;
        if (rho?.extra_env) r.env = rho.extra_env;
        // Only args/env move into this synthetic preset. Keep role-specific
        // operations (such as remove_args and feature_overrides) in their
        // original layer rather than silently dropping their behavior.
        if (rho) {
          // YAML aliases can share one role block between hardware ids.
          const { extra_args, extra_env, ...remaining } = rho;
          if (Object.keys(remaining).length) so[role].hardware_overrides[hw] = remaining;
          else delete so[role].hardware_overrides[hw];
        }
        for (const k of ["nodes", "parallelism"]) {
          const map = so[role]?.[k];
          if (map && typeof map === "object" && Object.hasOwn(map, hw)) { r[k] = map[hw]; delete map[hw]; }
        }
        if (Object.keys(r).length) (p.roles ??= {})[role] = r;
      }
      (out.presets ??= {})[`${hw}_${s}`] = p;
      moved.push([s, hw]);
    }
  }
  return { recipe: out, moved };
}

test("moving exact-GPU strategy blocks into default presets renders identically", () => {
  let checked = 0;
  for (const file of yamlFiles(path.join(contentDir(), "models"))) {
    const recipe = yaml.load(fs.readFileSync(file, "utf8"));
    const { recipe: migrated, moved } = migrate(recipe);
    for (const [s, hw] of moved) {
      assert.deepEqual(render(migrated, s, hw), render(recipe, s, hw), `${file} ${s} ${hw}`);
      if (s === "pd_cluster") {
        const mounted = withPreset(migrated, resolvePresetKey(migrated, hw, s), hw, s);
        const prof = taxonomy.hardware_profiles[hw];
        for (const role of ["prefill", "decode"]) {
          assert.equal(pdRoleNodes(mounted, role, prof, hw), pdRoleNodes(recipe, role, prof, hw), `${file} ${hw} ${role} nodes`);
          assert.equal(pdRoleParallelism(mounted, strategies.pd_cluster, role, prof, hw),
            pdRoleParallelism(recipe, strategies.pd_cluster, role, prof, hw), `${file} ${hw} ${role} parallelism`);
        }
      }
      checked++;
    }
  }
  assert.ok(checked > 0, "no exact-GPU strategy blocks found to migrate");
});

const recipe = {
  hf_id: "test/presets",
  model: { model_id: "test/presets", base_args: ["--max-model-len", "65536"] },
  variants: { default: { precision: "fp8", vram_minimum_gb: 100 } },
  compatible_strategies: ["single_node_tp", "pd_cluster"],
  presets: {
    agentx: {
      label: "AgentX", hardware: "gb200", strategy: "single_node_tp", default: true,
      args: ["--max-num-batched-tokens", "1024"], env: { NCCL_NET_PLUGIN: "none" },
    },
    latency: { label: "Low latency", hardware: "gb200", args: ["--max-num-seqs", "4"] },
    pd: {
      label: "PD", hardware: "gb200", strategy: "pd_cluster",
      roles: { decode: { args: ["--max-num-batched-tokens", "512"] } },
    },
  },
};

test("a default preset applies; baseline and other presets are picked explicitly", () => {
  const def = resolveCommand(recipe, "default", "single_node_tp", "gb200", [], strategies, taxonomy);
  assert.equal(def.preset.selected, "agentx");
  assert.deepEqual(def.preset.options.map((p) => p.key), ["agentx", "latency"]);
  assert.match(def.command, /--max-num-batched-tokens 1024/);
  assert.equal(def.env.NCCL_NET_PLUGIN, "none");

  const base = render(recipe, "single_node_tp", "gb200", BASELINE);
  assert.doesNotMatch(base.command, /--max-num-batched-tokens/);
  assert.equal(base.env.NCCL_NET_PLUGIN, undefined);

  assert.match(render(recipe, "single_node_tp", "gb200", "latency").command, /--max-num-seqs 4/);
  // A preset for another strategy or GPU is ignored.
  assert.doesNotMatch(render(recipe, "single_node_tp", "gb200", "pd").command, /512/);
  assert.equal(resolveCommand(recipe, "default", "single_node_tp", "h200", [], strategies, taxonomy).preset, null);
});

test("a pd_cluster preset reaches only the role it names", () => {
  const r = render(recipe, "pd_cluster", "gb200", "pd");
  assert.match(r.decode.command, /--max-num-batched-tokens 512/);
  assert.doesNotMatch(r.prefill.command, /--max-num-batched-tokens 512/);
});

test("preset validation", () => {
  const errs = (presets) => validatePresets({ ...recipe, presets }, taxonomy, ["single_node_tp", "pd_cluster"]);
  assert.deepEqual(errs(recipe.presets), []);
  assert.ok(errs({ a: { label: "a", hardware: "h999" } }).some((e) => /unknown hardware/.test(e)));
  assert.ok(errs({ a: { label: "a", hardware: "gb200", strategy: "multi_node_dep" } }).some((e) => /not offered/.test(e)));
  assert.ok(errs({ a: { label: "a", hardware: "gb200", roles: { decode: {} } } }).some((e) => /roles/.test(e)));
  assert.ok(errs({
    a: { label: "a", hardware: "gb200", default: true },
    b: { label: "b", hardware: "gb200", default: true },
  }).some((e) => /already the default/.test(e)));
  assert.ok(errs({ [BASELINE]: { label: "x", hardware: "gb200" } }).some((e) => /reserved/.test(e)));
});

// A lane-shaped preset: points, arg removal, JSON edits, variant scope,
// builder choices and an image pin.
const lane = {
  hf_id: "test/lane",
  model: { model_id: "test/lane", base_args: ["--block-size", "256", "--max-model-len", "65536"] },
  variants: { default: { precision: "fp8", vram_minimum_gb: 100 }, nvfp4: { precision: "nvfp4", vram_minimum_gb: 60 } },
  compatible_strategies: ["single_node_tp", "pd_cluster"],
  features: {
    spec_decoding: { modes: { mtp: { args: ["--speculative-config", '{"method":"mtp","num_speculative_tokens":2,"draft_x":true}'] } } },
    tool_calling: { args: ["--enable-auto-tool-choice"] },
  },
  presets: {
    agg: {
      label: "Agg", hardware: "b200", strategy: "single_node_tp", variants: ["nvfp4"], default: true,
      image: { vllm: "vllm/vllm-openai:nightly-abc@sha256:00", dynamo: "ffff" },
      select: { frontend: "python", features: { spec_decoding: "mtp", tool_calling: false } },
      args: ["--enable-prefix-caching"],
      remove_args: ["--block-size"],
      json_args: { "--speculative-config": { num_speculative_tokens: 4, draft_x: null }, "--compilation-config": { cudagraph_capture_sizes: [1, 2] } },
      points: {
        c1: { label: "C1", concurrency: 1, args: ["--max-num-seqs", "2"] },
        c8: { label: "C8", concurrency: 8, default: true, args: ["--max-num-seqs", "16"], select: { features: { tool_calling: true } } },
      },
    },
    pd: {
      label: "PD", hardware: "b200", strategy: "pd_cluster",
      roles: { decode: { nodes: 2, args: ["--max-num-seqs", "8"] } },
      points: {
        c16: { label: "C16", roles: { decode: { args: ["--max-num-seqs", "32"] } } },
        c32: { label: "C32", roles: { decode: { nodes: 4, args: ["--max-num-seqs", "64"], json_args: { "--compilation-config": { mode: 3 } } } } },
      },
    },
  },
};
const laneRender = (variant, strategy, preset, point, feats = ["spec_decoding"]) => resolveCommand(lane, variant, strategy, "b200", feats, strategies, taxonomy,
  [], 1, null, { spec_decoding: "mtp" }, null, null, undefined, null, "http", null, preset, point);
const flagValue = (argv, f) => { const i = argv.lastIndexOf(f); return i < 0 ? undefined : argv[i + 1]; };

test("points: the default point applies, a pick selects another, and both layer on the preset", () => {
  const def = laneRender("nvfp4", "single_node_tp");
  assert.equal(def.preset.selected, "agg");
  assert.equal(def.preset.point.selected, "c8");
  assert.deepEqual(def.preset.point.options.map((pt) => pt.key), ["c1", "c8"]);
  assert.equal(flagValue(def.argv, "--max-num-seqs"), "16");
  assert.ok(def.argv.includes("--enable-prefix-caching"));
  assert.equal(flagValue(laneRender("nvfp4", "single_node_tp", undefined, "c1").argv, "--max-num-seqs"), "2");
  // An unknown point falls back to the default one.
  assert.equal(laneRender("nvfp4", "single_node_tp", undefined, "nope").preset.point.selected, "c8");
});

test("remove_args drops inherited flags; json_args edits keys inside JSON flags, features included", () => {
  const { argv } = laneRender("nvfp4", "single_node_tp");
  assert.equal(argv.includes("--block-size"), false);
  assert.equal(flagValue(argv, "--max-model-len"), "65536");
  assert.deepEqual(JSON.parse(flagValue(argv, "--speculative-config")), { method: "mtp", num_speculative_tokens: 4 });
  assert.deepEqual(JSON.parse(flagValue(argv, "--compilation-config")), { cudagraph_capture_sizes: [1, 2] });
  // Baseline keeps the inherited flag and the feature's own JSON.
  const base = laneRender("nvfp4", "single_node_tp", BASELINE);
  assert.equal(flagValue(base.argv, "--block-size"), "256");
  assert.equal(JSON.parse(flagValue(base.argv, "--speculative-config")).num_speculative_tokens, 2);
  // A toggle-owned flag is only edited: spec decoding off leaves no partial
  // --speculative-config, while a flag nothing toggles is still added.
  const off = laneRender("nvfp4", "single_node_tp", undefined, undefined, []);
  assert.equal(off.argv.includes("--speculative-config"), false);
  assert.deepEqual(JSON.parse(flagValue(off.argv, "--compilation-config")), { cudagraph_capture_sizes: [1, 2] });
  // A json_args edit on a flag whose value isn't JSON is an error, not a skip.
  assert.throws(() => applyPresetArgOps(["--block-size", "128"], { json_args: { "--block-size": { x: 1 } } }), /not a JSON object/);
  assert.throws(() => applyPresetArgOps(["--kv-cache-dtype", "fp8"], { json_args: { "--kv-cache-dtype": { x: 1 } } }), /not a JSON object/);
  // Unit level: `_` and `-` match alike, and only the last occurrence is edited.
  assert.deepEqual(
    applyPresetArgOps(["--block_size", "1", "--x-cfg", '{"a":1}', "--x_cfg", '{"a":2}'], { remove_args: ["--block-size"], json_args: { "--x-cfg": { b: 3 } } }),
    ["--x-cfg", '{"a":1}', "--x_cfg", '{"a":2,"b":3}'],
  );
});

test("variants scope a preset; select and image travel with the point", () => {
  assert.equal(laneRender("default", "single_node_tp").preset, null);
  const r = laneRender("nvfp4", "single_node_tp");
  assert.deepEqual(r.preset.select, { frontend: "python", features: { spec_decoding: "mtp", tool_calling: true } });
  assert.deepEqual(laneRender("nvfp4", "single_node_tp", undefined, "c1").preset.select.features, { spec_decoding: "mtp", tool_calling: false });
  assert.deepEqual(presetSelect(lane, "agg", "c1").features.tool_calling, false);
  assert.equal(r.preset.image.vllm, "vllm/vllm-openai:nightly-abc@sha256:00");
  const mounted = withPreset(lane, "agg", "b200", "single_node_tp", { variantKey: "nvfp4" });
  assert.equal(mounted.variants.nvfp4.hardware_overrides.b200.docker_image, "vllm/vllm-openai:nightly-abc@sha256:00");
  assert.equal(mounted.variants.default.hardware_overrides, undefined);
});

test("pd_cluster points change role args, topology and role JSON edits", () => {
  const eff = effectivePreset(lane.presets.pd, "c32");
  assert.equal(eff.roles.decode.nodes, 4);
  assert.deepEqual(eff.roles.decode.args, ["--max-num-seqs", "8", "--max-num-seqs", "64"]);
  const m = withPreset(lane, "pd", "b200", "pd_cluster", { point: "c32" });
  assert.equal(pdRoleNodes(m, "decode", taxonomy.hardware_profiles.b200, "b200"), 4);
  assert.equal(pdRoleNodes(withPreset(lane, "pd", "b200", "pd_cluster", { point: "c16" }), "decode", taxonomy.hardware_profiles.b200, "b200"), 2);
  const r = resolveCommand(lane, "default", "pd_cluster", "b200", [], strategies, taxonomy,
    [], 1, { prefill: { nodes: 1, rank: 0, parallelism: "tp" }, decode: { nodes: 4, rank: 0, parallelism: "tp" } }, {}, null, null, undefined, null, "http", null, "pd", "c32");
  assert.equal(flagValue(r.decode.argv, "--max-num-seqs"), "64");
  assert.equal(JSON.parse(flagValue(r.decode.argv, "--compilation-config")).mode, 3);
  assert.equal(r.prefill.argv.includes("--max-num-seqs"), false);
});

test("a point's image fields layer over the preset's", () => {
  const p = { image: { vllm: "img:1", dynamo: "d1" }, points: { a: { label: "a", image: { dynamo: "d2" } }, b: { label: "b", image: "img:2" } } };
  assert.deepEqual(effectivePreset(p, "a").image, { vllm: "img:1", dynamo: "d2" });
  assert.deepEqual(effectivePreset(p, "b").image, { vllm: "img:2", dynamo: "d1" });
});

test("select.kv_offload accepts only known options", () => {
  const kvIds = Object.values(strategies).filter((st) => st.deploy_type === "kv_store_lb").map((st) => st.name);
  assert.ok(kvIds.length > 0);
  const errs = (kv) => validatePresets({ ...lane, presets: { a: { label: "a", hardware: "b200", strategy: "single_node_tp", select: { kv_offload: kv } } } },
    taxonomy, ["single_node_tp", "pd_cluster"], kvIds);
  assert.deepEqual(errs(kvIds[0]), []);
  assert.deepEqual(errs(Object.keys(taxonomy.kv_offload)[0]), []);
  assert.deepEqual(errs("off"), []);
  assert.ok(errs("kv_store_typo_mooncake").some((e) => /unknown option/.test(e)));
  const validate = new Ajv({ allErrors: true, strict: false }).compile(buildRecipeSchema({ taxonomy, strategies }));
  const withKv = (kv) => validate({ ...structuredClone(lane), meta: { title: "t", slug: "t", provider: "p", description: "d", date_added: "2026-01-01", date_updated: "2026-01-01", tasks: ["text"] },
    presets: { a: { label: "a", hardware: "b200", select: { kv_offload: kv } } } }) || validate.errors.map((e) => e.instancePath).join(",");
  assert.match(String(withKv("kv_store_typo_mooncake")), /kv_offload/);
});

test("tp sets the single-node TP size; points can change it", () => {
  const r = {
    ...structuredClone(lane),
    presets: {
      tp: {
        label: "TP", hardware: "b200", strategy: "single_node_tp", variants: ["nvfp4"], default: true, tp: 4,
        points: { a: { label: "A" }, b: { label: "B", tp: 2 } },
      },
    },
  };
  const tpOf = (variant, point, preset) => flagValue(resolveCommand(r, variant, "single_node_tp", "b200", [], strategies, taxonomy,
    [], 1, null, {}, null, null, undefined, null, "http", null, preset, point).argv, "--tensor-parallel-size");
  assert.equal(tpOf("nvfp4", "a"), "4");
  assert.equal(tpOf("nvfp4", "b"), "2");
  // Baseline keeps the auto-fit TP (the NVFP4 fixture fits one GPU).
  assert.equal(tpOf("nvfp4", "a", BASELINE), "1");
  // Another variant keeps its own TP: the preset only mounts onto the variants it covers.
  assert.equal(tpOf("default", "a"), tpOf("default", "a", BASELINE));
});

test("an Ascend PD preset accepts and renders separate router and connector keys", () => {
  const model = read("models/moonshotai/Kimi-K3.yaml");
  for (const field of ["date_added", "date_updated"]) {
    if (model.meta[field] instanceof Date) model.meta[field] = model.meta[field].toISOString().slice(0, 10);
  }
  model.presets = { a5: {
    label: "A5 PD", hardware: "ascend_950dt", strategy: "pd_cluster",
    select: { pd_router: "ascend", pd_connector: "mooncakev2" },
  } };
  const validate = new Ajv({ allErrors: true, strict: false }).compile(buildRecipeSchema({ taxonomy, strategies }));
  assert.ok(validate(model), JSON.stringify(validate.errors));
  const errors = () => validatePresets(model, taxonomy, Object.keys(strategies), [], strategies.pd_cluster.connectors);
  assert.deepEqual(errors(), []);
  const result = resolveCommand(model, "ascend_mxfp4", "pd_cluster", "ascend_950dt", [], strategies, taxonomy,
    [], 1, null, {}, null, null, undefined, "ascend", "http", null, "a5", null, "mooncakev2");
  assert.equal(result.pdRouter, "ascend");
  assert.equal(result.kvConnector.selected, "mooncakev2");
  model.presets.a5.select.pd_router = "vllm";
  assert.ok(errors().some((error) => /doesn't work behind pd_router vllm/.test(error)));
});

test("a DEP preset narrowed below the node also narrows the router", () => {
  const r = {
    ...structuredClone(lane),
    compatible_strategies: ["single_node_dep"],
    presets: {
      dep: {
        label: "DEP", hardware: "b200", strategy: "single_node_dep", default: true, args: ["--data-parallel-size", "2"],
        points: { a: { label: "A" }, b: { label: "B", args: ["--data-parallel-size", "4"] } },
      },
    },
  };
  const at = (point, preset) => {
    const res = resolveCommand(r, "default", "single_node_dep", "b200", [], strategies, taxonomy,
      [], 1, null, {}, null, null, undefined, null, "http", null, preset, point);
    const router = res.companions.find((c) => c.feature === "strategy:single_node_dep").command;
    return [flagValue(res.argv, "--data-parallel-size"), router.match(/--intra-node-data-parallel-size (\d+)/)[1]];
  };
  assert.deepEqual(at("a"), ["2", "2"]);
  assert.deepEqual(at("b"), ["4", "4"]);
  assert.deepEqual(at("a", BASELINE), ["8", "8"]);
});

test("preset validation: points, select, arg ops and variants", () => {
  const errs = (presets) => validatePresets({ ...lane, presets }, taxonomy, ["single_node_tp", "pd_cluster"]);
  assert.deepEqual(errs(lane.presets), []);
  const one = (extra) => errs({ a: { label: "a", hardware: "b200", strategy: "single_node_tp", ...extra } });
  assert.ok(one({ variants: ["int4"] }).some((e) => /no variant "int4"/.test(e)));
  assert.ok(one({ select: { pd_router: "dynamo" } }).some((e) => /pd_router/.test(e)));
  assert.ok(one({ select: { frontend: "go" } }).some((e) => /unknown frontend/.test(e)));
  assert.ok(one({ select: { features: { nope: true } } }).some((e) => /no feature "nope"/.test(e)));
  assert.ok(one({ select: { features: { spec_decoding: "dflash" } } }).some((e) => /no mode "dflash"/.test(e)));
  assert.ok(one({ args: ["--x", "1"], remove_args: ["--x"] }).some((e) => /both added and removed/.test(e)));
  assert.ok(one({ json_args: { "--x": "1" } }).some((e) => /must be an object/.test(e)));
  assert.ok(one({ points: { a: { label: "a", default: true }, b: { label: "b", default: true } } }).some((e) => /more than one default point/.test(e)));
  assert.ok(one({ points: { a: { label: "a", roles: { decode: {} } } } }).some((e) => /only a pd_cluster preset has roles/.test(e)));
  assert.ok(errs({ a: { label: "a", hardware: "b200", strategy: "pd_cluster", tp: 4 } }).some((e) => /only a single_node_tp preset sets tp/.test(e)));
  assert.ok(one({ tp: 99 }).some((e) => /exceeds/.test(e)));
  assert.ok(one({ select: { nodes: 2 } }).some((e) => /only a multi_node_\* preset sets the node count/.test(e)));
  assert.ok(one({ points: { a: { label: "a", tp: 99 } } }).some((e) => /exceeds/.test(e)));
});

test("an overlay adds only presets for its own hardware, and split moves them back", () => {
  const ctx = resolveScope({ brands: ["Huawei"] }, taxonomy);
  const opts = { ctx, source: "vllm-ascend" };
  const ascend = { presets: { a3: { label: "A3", hardware: "ascend_910c", runbook: "…" } } };
  const merged = applyOverlay(recipe, ascend, opts);
  assert.deepEqual(merged.errors, []);
  assert.equal(merged.recipe.presets.a3.hardware, "ascend_910c");
  assert.ok(applyOverlay(recipe, { presets: { x: { label: "x", hardware: "gb200" } } }, opts).errors.length);
  assert.ok(applyOverlay(recipe, { presets: { agentx: { label: "x", hardware: "ascend_910c" } } }, opts).errors.length);
  const { core, overlay } = splitOverlay(merged.recipe, opts);
  assert.deepEqual(overlay.presets, ascend.presets);
  assert.deepEqual(core.presets, recipe.presets);
});
