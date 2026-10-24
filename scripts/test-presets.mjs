import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { test } from "node:test";
import yaml from "js-yaml";
import { pdRoleNodes, pdRoleParallelism, resolveCommand } from "../src/lib/command-synthesis.js";
import { contentDir } from "../src/lib/content-root.js";
import { applyOverlay, resolveScope, splitOverlay } from "../src/lib/overlay-merge.js";
import { BASELINE, applyPresetArgOps, effectivePreset, presetSelect, resolvePresetKey, validatePresets, withPreset } from "../src/lib/recipe-presets.js";

const read = (file) => yaml.load(fs.readFileSync(path.join(contentDir(), file), "utf8"));
const taxonomy = read("taxonomy.yaml");
const strategies = Object.fromEntries(fs.readdirSync(path.join(contentDir(), "strategies"))
  .filter((file) => file.endsWith(".yaml"))
  .map((file) => { const s = read(`strategies/${file}`); return [s.name, s]; }));

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
        delete so[role]?.hardware_overrides?.[hw];
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
