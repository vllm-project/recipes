import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { test } from "node:test";
import yaml from "js-yaml";
import { pdRoleNodes, pdRoleParallelism, resolveCommand } from "../src/lib/command-synthesis.js";
import { contentDir } from "../src/lib/content-root.js";
import { applyOverlay, resolveScope, splitOverlay } from "../src/lib/overlay-merge.js";
import { BASELINE, resolveProfileKey, validateProfiles, withProfile } from "../src/lib/recipe-profiles.js";

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

const render = (recipe, strategy, hw, profile) => {
  const r = resolveCommand(recipe, "default", strategy, hw, [], strategies, taxonomy,
    [], 1, null, {}, null, null, undefined, null, "http", null, profile);
  if (!r) return r;
  const { profile: _p, ...rest } = r;
  return JSON.parse(JSON.stringify(rest));
};

// Move every exact-GPU strategy block (and its role blocks) into a
// `default: true` profile: the shape a recipe migrates to.
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
      (out.profiles ??= {})[`${hw}_${s}`] = p;
      moved.push([s, hw]);
    }
  }
  return { recipe: out, moved };
}

test("moving exact-GPU strategy blocks into default profiles renders identically", () => {
  let checked = 0;
  for (const file of yamlFiles(path.join(contentDir(), "models"))) {
    const recipe = yaml.load(fs.readFileSync(file, "utf8"));
    const { recipe: migrated, moved } = migrate(recipe);
    for (const [s, hw] of moved) {
      assert.deepEqual(render(migrated, s, hw), render(recipe, s, hw), `${file} ${s} ${hw}`);
      if (s === "pd_cluster") {
        const mounted = withProfile(migrated, resolveProfileKey(migrated, hw, s), hw, s);
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
  hf_id: "test/profiles",
  model: { model_id: "test/profiles", base_args: ["--max-model-len", "65536"] },
  variants: { default: { precision: "fp8", vram_minimum_gb: 100 } },
  compatible_strategies: ["single_node_tp", "pd_cluster"],
  profiles: {
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

test("a default profile applies; baseline and other profiles are picked explicitly", () => {
  const def = resolveCommand(recipe, "default", "single_node_tp", "gb200", [], strategies, taxonomy);
  assert.equal(def.profile.selected, "agentx");
  assert.deepEqual(def.profile.options.map((p) => p.key), ["agentx", "latency"]);
  assert.match(def.command, /--max-num-batched-tokens 1024/);
  assert.equal(def.env.NCCL_NET_PLUGIN, "none");

  const base = render(recipe, "single_node_tp", "gb200", BASELINE);
  assert.doesNotMatch(base.command, /--max-num-batched-tokens/);
  assert.equal(base.env.NCCL_NET_PLUGIN, undefined);

  assert.match(render(recipe, "single_node_tp", "gb200", "latency").command, /--max-num-seqs 4/);
  // A profile for another strategy or GPU is ignored.
  assert.doesNotMatch(render(recipe, "single_node_tp", "gb200", "pd").command, /512/);
  assert.equal(resolveCommand(recipe, "default", "single_node_tp", "h200", [], strategies, taxonomy).profile, null);
});

test("a pd_cluster profile reaches only the role it names", () => {
  const r = render(recipe, "pd_cluster", "gb200", "pd");
  assert.match(r.decode.command, /--max-num-batched-tokens 512/);
  assert.doesNotMatch(r.prefill.command, /--max-num-batched-tokens 512/);
});

test("profile validation", () => {
  const errs = (profiles) => validateProfiles({ ...recipe, profiles }, taxonomy, ["single_node_tp", "pd_cluster"]);
  assert.deepEqual(errs(recipe.profiles), []);
  assert.ok(errs({ a: { label: "a", hardware: "h999" } }).some((e) => /unknown hardware/.test(e)));
  assert.ok(errs({ a: { label: "a", hardware: "gb200", strategy: "multi_node_dep" } }).some((e) => /not offered/.test(e)));
  assert.ok(errs({ a: { label: "a", hardware: "gb200", roles: { decode: {} } } }).some((e) => /roles/.test(e)));
  assert.ok(errs({
    a: { label: "a", hardware: "gb200", default: true },
    b: { label: "b", hardware: "gb200", default: true },
  }).some((e) => /already the default/.test(e)));
  assert.ok(errs({ [BASELINE]: { label: "x", hardware: "gb200" } }).some((e) => /reserved/.test(e)));
});

test("an overlay adds only profiles for its own hardware, and split moves them back", () => {
  const ctx = resolveScope({ brands: ["Huawei"] }, taxonomy);
  const opts = { ctx, source: "vllm-ascend" };
  const ascend = { profiles: { a3: { label: "A3", hardware: "ascend_910c", runbook: "…" } } };
  const merged = applyOverlay(recipe, ascend, opts);
  assert.deepEqual(merged.errors, []);
  assert.equal(merged.recipe.profiles.a3.hardware, "ascend_910c");
  assert.ok(applyOverlay(recipe, { profiles: { x: { label: "x", hardware: "gb200" } } }, opts).errors.length);
  assert.ok(applyOverlay(recipe, { profiles: { agentx: { label: "x", hardware: "ascend_910c" } } }, opts).errors.length);
  const { core, overlay } = splitOverlay(merged.recipe, opts);
  assert.deepEqual(overlay.profiles, ascend.profiles);
  assert.deepEqual(core.profiles, recipe.profiles);
});
