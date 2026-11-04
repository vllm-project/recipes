import assert from "node:assert/strict";
import fs from "node:fs";
import { test } from "node:test";
import yaml from "js-yaml";
import { resolveCommand, resolveKvOffloadOption, isKvOffloadBrandSupported, kvOffloadInstallSteps, kvOffloadOptionErrors } from "../src/lib/command-synthesis.js";
import { resolveSelection } from "../src/lib/recipe-render.js";
import { selectionMarkdown } from "../src/lib/recipe-markdown.js";

// The vocabulary (taxonomy + strategies) always comes from this repo's root.
const root = new URL("../", import.meta.url);
const read = (file) => yaml.load(fs.readFileSync(new URL(file, root), "utf8"));
const taxonomy = read("taxonomy.yaml");
const strategies = Object.fromEntries(fs.readdirSync(new URL("strategies/", root))
  .filter((file) => file.endsWith(".yaml"))
  .map((file) => { const strategy = read(`strategies/${file}`); return [strategy.name, strategy]; }));
const hw = (id) => taxonomy.hardware_profiles[id];

const recipe = (extra = {}) => ({
  hf_id: "test/kv-offload",
  model: { model_id: "test/kv-offload", base_args: [] },
  // Bigger than one H100, so single_node_tp shards across the whole node.
  variants: { default: { precision: "bf16", vram_minimum_gb: 200 } },
  compatible_strategies: ["single_node_tp", "multi_node_tp"],
  // IAXL is fail-closed: it renders only where a recipe records a verified run.
  kv_offload_support: { iaxl: "verified" },
  ...extra,
});
const run = (r, hwId, kvOffload, tax = taxonomy) =>
  resolveCommand(r, "default", "single_node_tp", hwId, [], strategies, tax, [], 1, null, {}, kvOffload);
const flag = (argv, key) => (argv.includes(key) ? argv[argv.lastIndexOf(key) + 1] : undefined);

test("hardware_overrides replace top-level keys and merge env", () => {
  const opt = { description: "base", install: "base", env: { A: "1", B: "1" }, hardware_overrides: { cpu: { install: "cpu", env: { B: "2" } } } };
  const cpu = resolveKvOffloadOption(opt, hw("xeon6"), "xeon6");
  assert.equal(cpu.install, "cpu");
  assert.equal(cpu.description, "base");
  assert.deepEqual(cpu.env, { A: "1", B: "2" });
  assert.equal(resolveKvOffloadOption(opt, hw("h100"), "h100"), opt);
});

test("IAXL hardware gate: Xeon CPUs and NVIDIA, minus Grace/Vera-based nodes and Intel XPU", () => {
  const iaxl = taxonomy.kv_offload.iaxl;
  for (const id of ["xeon6", "h100", "b200", "rtx_pro_6000"]) assert.ok(isKvOffloadBrandSupported(iaxl, hw(id), id), id);
  for (const id of ["gb200", "gb300", "dgx_station_gb300", "dgx_spark_gb10", "vr200", "mi300x", "arc_pro_b60"]) assert.ok(!isKvOffloadBrandSupported(iaxl, hw(id), id), id);
});

test("IAXL on Xeon CPU renders the CPU build, its env and the setvars.sh setup", () => {
  const r = run(recipe(), "xeon6", "iaxl");
  assert.equal(r.env.DEVICE, "cpu");
  assert.equal(r.env.TP_SIZE, flag(r.argv, "--tensor-parallel-size"));
  assert.equal(r.env.IAXL_DSA_GD_ENABLE, undefined);
  // CPU sizes the pool explicitly; vLLM's CPU engine shares the host RAM.
  assert.equal(r.env.IAXL_DDR_POOL_SIZE_GB, "16");
  assert.match(r.setup, /source setvars\.sh/);
  assert.match(r.setup, /export LD_PRELOAD="\$IAXL_LD_PRELOAD/);
  // Chained to vllm serve, so a failing setvars.sh stops the launch.
  assert.match(r.setup, /source setvars\.sh && export LD_PRELOAD=\S+ &&$/);
  assert.match(r.argv.join(" "), /KVShrinkConnector/);
  assert.ok(r.argv.includes("--no-enable-prefix-caching"));
});

test("IAXL on NVIDIA fills {tp_size} from the engine's TP and keeps DSA off by default", () => {
  const r = run(recipe(), "h100", "iaxl");
  assert.equal(flag(r.argv, "--tensor-parallel-size"), "8");
  assert.equal(r.env.TP_SIZE, "8");
  assert.equal(r.env.DEVICE, "cuda");
  assert.equal(r.env.IAXL_DSA_GD_ENABLE, "0");
  // GPU servers keep IAXL's own pool default (80% of host RAM across ranks).
  assert.equal(r.env.IAXL_DDR_POOL_SIZE_GB, undefined);
});

test("IAXL is fail-closed: off without a verified run, when unsupported and on Grace nodes, with no trace", () => {
  assert.equal(taxonomy.kv_offload.iaxl.requires_opt_in, true);
  assert.match(run(recipe(), "h100", "iaxl").argv.join(" "), /KVShrinkConnector/);
  const cases = [
    [recipe({ kv_offload_support: undefined }), "h100"],
    [recipe({ kv_offload_support: {} }), "xeon6"],
    [recipe({ kv_offload_support: { iaxl: "unsupported" } }), "h100"],
    [recipe(), "gb200"],
  ];
  for (const [r, id] of cases) {
    const out = run(r, id, "iaxl");
    assert.ok(!out.argv.join(" ").includes("KVShrinkConnector"), id);
    assert.equal(out.setup, undefined, id);
    assert.equal(out.env.TP_SIZE, undefined, id);
  }
});

test("a hardware override of a gating key gates synthesis like the builder", () => {
  const tax = {
    ...taxonomy,
    kv_offload: {
      ...taxonomy.kv_offload,
      gated: { args: ["--gated"], hardware_overrides: { h100: { strategies: ["multi_node_tp"] } } },
    },
  };
  assert.ok(!run(recipe(), "h100", "gated", tax).argv.includes("--gated"));
  assert.ok(run(recipe(), "h200", "gated", tax).argv.includes("--gated"));
});

test("install renders as dependency-style steps: a bare string or { note, command } entries", () => {
  assert.deepEqual(kvOffloadInstallSteps(taxonomy.kv_offload.lmcache), [{ command: taxonomy.kv_offload.lmcache.install }]);
  assert.deepEqual(kvOffloadInstallSteps({}), []);
  const steps = (id) => kvOffloadInstallSteps(resolveKvOffloadOption(taxonomy.kv_offload.iaxl, hw(id), id));
  const cpu = steps("xeon6"), gpu = steps("h100");
  for (const s of [...cpu, ...gpu]) assert.ok(s.note && s.command && s.command === s.command.trimEnd());
  assert.match(cpu[0].command, /^sudo apt-get install -y /);
  assert.match(cpu.at(-1).command, /DEVICE=cpu /);
  // NVIDIA builds GDRCopy's gdrapi, UCX and NIXL, pinned, before IAXL itself.
  assert.match(gpu[0].command, /^sudo apt-get install -y .* rdma-core /);
  assert.match(gpu[1].command, /libgdrapi_2\.6-1_amd64/);
  assert.match(gpu[2].command, /--branch v1\.21\.0 https:\/\/github\.com\/openucx\/ucx/);
  assert.match(gpu[3].command, /--branch v1\.2\.0 https:\/\/github\.com\/ai-dynamo\/nixl/);
  assert.match(gpu.at(-1).command, /DEVICE=cuda /);
  // Both builds check out the same pinned IAXL commit, never a moving branch.
  const pins = [cpu, gpu].map((list) => list.find((s) => s.command.includes("intel-accel-for-llm.git")).command.match(/checkout -q ([0-9a-f]{40})$/)?.[1]);
  assert.ok(pins[0] && pins[0] === pins[1], "IAXL checkout pinned to one full commit");
});

// The `.md` / `.json` pages and the static JSON API render through
// resolveSelection / renderCommand, not the builder: they must carry the same
// setup lines and install steps, and no docker equivalent for IAXL.
test("agent renderings carry IAXL's setup and install steps, and no docker command", () => {
  const r = {
    ...recipe(),
    meta: { title: "t", hardware: { xeon6: "verified" } },
    variants: { default: { precision: "bf16", vram_minimum_gb: 20, supported_hardware: ["xeon6", "h100"] } },
  };
  const page = (q) => resolveSelection(r, new URLSearchParams(q), strategies, taxonomy);
  const iaxl = page("hardware=xeon6&strategy=single_node_tp&kv_offload=iaxl").rendering;
  assert.equal(iaxl.hardware, "xeon6");
  assert.match(iaxl.setup, /source setvars\.sh/);
  assert.equal(iaxl.kv_offload_install.length, 3);
  assert.equal(iaxl.docker_command, null);
  assert.equal(iaxl.docker_image, null);

  const urls = { page: "https://x.test/test/kv-offload", md: "https://x.test/test/kv-offload.md", json: "https://x.test/test/kv-offload.json" };
  const md = selectionMarkdown(r, page("hardware=xeon6&strategy=single_node_tp&kv_offload=iaxl"), { strategies, taxonomy, urls });
  const exp = md.indexOf("export DEVICE=cpu"), src = md.indexOf("source setvars.sh"), serve = md.indexOf("vllm serve");
  assert.ok(exp >= 0 && exp < src && src < serve, "exports, then setvars.sh, then vllm serve");
  assert.match(md, /&&\nvllm serve /, "setup chained straight into vllm serve");
  assert.ok(md.includes("IAXL_QATLIB=intree uv pip install --no-build-isolation -e ."));
  assert.ok(!md.includes("Docker equivalent"));

  // Options with a published image keep their docker equivalent.
  const lmcache = page("hardware=h100&strategy=single_node_tp&kv_offload=lmcache").rendering;
  assert.match(lmcache.docker_command, /docker run/);
  assert.equal(lmcache.setup, undefined);
});

// Options that render only from a pip install (`docker: false`) can't run on a
// Docker-only recipe: synthesis drops them and the page query says why.
test("an option without a published image is unavailable where the recipe has no pip install", () => {
  const base = {
    ...recipe(),
    meta: { title: "t", hardware: { xeon6: "verified" } },
    variants: { default: { precision: "bf16", vram_minimum_gb: 20, supported_hardware: ["xeon6", "h100"] } },
  };
  const dockerOnly = { ...base, model: { ...base.model, install: { pip: false } } };
  const perHw = { ...base, hardware_overrides: { xeon6: { install: { pip: false } } } };
  for (const [r, label] of [[dockerOnly, "model.install.pip: false"], [perHw, "hardware_overrides.xeon6.install.pip: false"]]) {
    assert.ok(!run(r, "xeon6", "iaxl").argv.join(" ").includes("KVShrinkConnector"), label);
    const sel = resolveSelection(r, new URLSearchParams("hardware=xeon6&strategy=single_node_tp&kv_offload=iaxl"), strategies, taxonomy).selection;
    assert.ok(sel.adjusted.some((a) => a.param === "kv_offload" && a.requested === "iaxl" && a.used === "off"), label);
  }
  // LMCache ships in the vLLM images, so a Docker-only recipe keeps it.
  assert.match(run(dockerOnly, "h100", "lmcache").argv.join(" "), /LMCacheMPConnector/);
  assert.match(run(base, "xeon6", "iaxl").argv.join(" "), /KVShrinkConnector/);
});

// A page query must not claim an option the command leaves out.
test("resolveSelection gates per hardware like synthesis", () => {
  const r = {
    ...recipe(),
    meta: { title: "t" },
    variants: { default: { precision: "bf16", vram_minimum_gb: 20 } },
  };
  const sel = (hwId) => resolveSelection(r, new URLSearchParams(`hardware=${hwId}&strategy=single_node_tp&kv_offload=iaxl`), strategies, taxonomy);
  const gb = sel("gb200");
  assert.equal(gb.selection.hardware, "gb200");
  assert.ok(gb.selection.adjusted.some((a) => a.param === "kv_offload" && a.used === "off"));
  assert.ok(!gb.rendering.command.includes("KVShrinkConnector"));
  const h100 = sel("h100");
  assert.ok(!h100.selection.adjusted.some((a) => a.param === "kv_offload"));
  assert.match(h100.rendering.command, /KVShrinkConnector/);
});

test("setup and {tp_size} are rejected outside single-node strategies", () => {
  assert.deepEqual(kvOffloadOptionErrors(taxonomy, strategies), []);
  const check = (opt) => kvOffloadOptionErrors({ kv_offload: { x: opt } }, strategies);
  assert.deepEqual(check({ strategies: ["single_node_tp"], setup: "source a.sh" }), []);
  assert.equal(check({ setup: "source a.sh" }).length, 1, "no strategies allowlist");
  assert.equal(check({ strategies: ["single_node_tp", "multi_node_tp"], env: { N: "{tp_size}" } }).length, 1);
  assert.equal(check({ strategies: ["single_node_tp"], pd: { roles: ["prefill"] }, env: { N: "{tp_size}" } }).length, 1);
  assert.equal(check({ strategies: ["single_node_tp"], hardware_overrides: { cpu: { setup: "source a.sh\n# done" } } }).length, 1, "setup ending in a comment");
  assert.deepEqual(check({ args: ["--x"] }), [], "options without setup / {tp_size} are unrestricted");
});
