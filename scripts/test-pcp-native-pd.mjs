import assert from "node:assert/strict";
import { test } from "node:test";
import { resolveCommand } from "../src/lib/command-synthesis.js";
import { loadStrategies } from "../src/lib/strategies.js";
import { loadTaxonomy } from "../src/lib/taxonomy.js";
import { renderCommand, resolveSelection } from "../src/lib/recipe-render.js";
import { selectionMarkdown } from "../src/lib/recipe-markdown.js";

const strategies = loadStrategies();
const taxonomy = loadTaxonomy();
const key = "gb300_pd_pcp_tp";

// Topology of vLLM's Mistral-Large-4-1T-A52B gb300_pd_pcp_tp preset
// (vllm@601066c9): producer TP1×PCP4×DCP4, consumer TP4 with kv_both.
function recipe() {
  return {
    hf_id: "test/native-pcp", model: { model_id: "test/native-pcp", architecture: "moe" },
    variants: { default: { precision: "fp8", vram_minimum_gb: 64 } },
    compatible_strategies: ["multi_node_tp", "multi_node_tep", "multi_node_dep", "pd_cluster"],
    presets: { [key]: {
      hardware: "gb300", strategy: "pd_cluster",
      roles: {
        prefill: { nodes: 1, parallelism: "tp", tp: 1, pcp: 4, args: ["-dcp", "4", "--enable-expert-parallel"] },
        decode: { nodes: 1, parallelism: "tp", json_args: { "--kv-transfer-config": { NixlConnector: { kv_role: "kv_both" } } } },
      },
    } },
  };
}
function render(r, opts = {}) {
  return renderCommand(r, "default", "pd_cluster", "gb300", 2, [], strategies, taxonomy,
    r.presets[key].roles, null, key, undefined, true, { pdRouter: "vllm", pdConnector: "nixl", ...opts });
}
function value(args, flag) {
  return args[args.indexOf(flag) + 1];
}
function cliOptions(argv) {
  const options = {};
  for (let i = 3; i < argv.length; i++) {
    const flag = argv[i];
    options[flag] = argv[i + 1] && !argv[i + 1].startsWith("--") ? argv[++i] : true;
  }
  return options;
}

test("native NIXL uses one ordinary vLLM engine for the Mistral GB300 PCP+DCP Prefill layout", () => {
  const r = recipe();
  const structured = render(r);
  assert.equal(structured.prefill.pcp, 4);
  assert.equal(structured.prefill.tp, 1);
  assert.equal(structured.prefill.pp, 1);
  assert.equal(structured.decode.tp, 4);
  assert.equal(structured.decode.pcp, 1);
  assert.equal(structured.pd_router, "vllm");
  assert.equal(structured.pd_connector, "nixl");
  assert.equal(structured.prefill.rank_commands, undefined);
  for (const role of [structured.prefill, structured.decode]) {
    assert.deepEqual(role.argv.slice(0, 2), ["vllm", "serve"]);
    assert.ok(!role.argv.includes("--data-parallel-external-lb"));
    assert.ok(!role.docker_argv.includes("/dev/devmm_svm"));
    assert.match(role.docker_command, /--network host/);
    assert.equal(value(role.docker_argv, "--network"), "host");
    assert.ok(!role.docker_argv.includes("-p"));
  }
  assert.equal(value(structured.prefill.argv, "--tensor-parallel-size"), "1");
  assert.equal(value(structured.prefill.argv, "--prefill-context-parallel-size"), "4");
  assert.equal(value(structured.prefill.argv, "--decode-context-parallel-size"), "4");
  assert.ok(!structured.decode.argv.includes("--prefill-context-parallel-size"));
  assert.match(structured.router.command, /--prefill http:\/\/\$PREFILL_NODE_1:8001/);
  assert.match(structured.router.command, /--decode http:\/\/\$DECODE_NODE_1:8002/);
  assert.equal(JSON.parse(value(structured.decode.argv, "--kv-transfer-config")).kv_role, "kv_both");

  // A pre-existing recipe with raw args can migrate without changing CLI values.
  const old = recipe();
  const p = old.presets[key].roles.prefill;
  delete p.tp;
  delete p.pcp;
  p.args = ["--tensor-parallel-size", "1", "--prefill-context-parallel-size", "4",
    "--decode-context-parallel-size", "4", "--enable-expert-parallel"];
  const previous = render(old);
  assert.deepEqual(cliOptions(structured.prefill.argv), cliOptions(previous.prefill.argv));
  assert.deepEqual(structured.decode.argv, previous.decode.argv);
  assert.equal(structured.router.command, previous.router.command);
});

test("PD role sizes resolve compute architecture after exact hardware and before default", () => {
  const r = recipe();
  delete r.presets;
  const query = new URLSearchParams({ hardware: "gb300", strategy: "pd_cluster", nodes: "3" });
  function prefill(sizes) {
    r.strategy_overrides = { pd_cluster: {
      prefill: { parallelism: "tp", ...sizes }, decode: { nodes: 1, parallelism: "tp" },
    } };
    return resolveSelection(r, query, strategies, taxonomy).rendering.prefill;
  }

  const sizes = { nodes: 2, tp: 2, pp: 1, pcp: 4 };
  const fallback = { nodes: 1, tp: 1, pp: 2, pcp: 1 };
  const scalar = prefill(sizes);
  assert.equal(value(scalar.argv, "--prefill-context-parallel-size"), "4");
  for (const exact of [false, true]) {
    const mapped = Object.fromEntries(Object.entries(sizes).map(([axis, size]) => [
      axis, exact
        ? { gb300: size, "arch:sm103": fallback[axis] }
        : { "arch:sm103": size, default: fallback[axis] },
    ]));
    assert.deepEqual(prefill(mapped), scalar);
  }
});

test("a native PCP preset stays on vLLM Router and NIXL across shared links", () => {
  const r = recipe();
  for (const mode of ["tp", "tep", "dep"]) {
    const query = new URLSearchParams({ hardware: "gb300", strategy: "pd_cluster", preset: key,
      prefill_mode: mode, pd_router: "ascend", pd_connector: "mooncake_ascend" });
    const { rendering, selection } = resolveSelection(r, query, strategies, taxonomy);
    assert.equal(rendering.prefill.parallelism, mode);
    assert.equal(rendering.pd_router, "vllm");
    assert.equal(rendering.pd_connector, "nixl");
    assert.deepEqual(selection.adjusted.map(({ param, used }) => [param, used]),
      [["pd_router", "vllm"], ["pd_connector", "nixl"]]);
  }
});

test("unqualified native PCP offload links keep the direct NIXL command and explain the adjustment", () => {
  const r = recipe();
  const q = new URLSearchParams({ hardware: "gb300", strategy: "pd_cluster", preset: key });
  const direct = resolveSelection(r, q, strategies, taxonomy);
  assert.equal(direct.rendering.prefill.pcp, 4);
  for (const tier of ["simple", "kv_store_centralized_mooncake", "kv_store_distributed_mooncake"]) {
    q.set("kv_offload", tier);
    const result = resolveSelection(r, q, strategies, taxonomy);
    assert.deepEqual(result.rendering, direct.rendering);
    assert.equal(result.selection.kv_offload, null);
    const adjustment = result.selection.adjusted.find(({ param }) => param === "kv_offload");
    assert.equal(adjustment.requested, tier);
    assert.equal(adjustment.used, "off");
    assert.match(adjustment.reason, /PCP/);
  }
});

test("native PCP synthesis reports incompatible KV offload as a PD error", () => {
  const r = recipe();
  assert.throws(
    () => resolveCommand(r, "default", "pd_cluster", "gb300", [], strategies, taxonomy,
      [], 2, null, {}, "simple", null, undefined, "vllm", "http", null, key, null, "nixl"),
    { name: "RangeError", message: /^PD prefill: KV offload/ },
  );
});

test("a native PD PCP preset's own offload defaults fall back to direct NIXL in JSON and Markdown", () => {
  const r = recipe();
  const p = r.presets[key];
  p.select = { kv_offload: "simple" };
  p.points = {
    inherited: { default: true },
    store: { select: { kv_offload: "kv_store_distributed_mooncake" } },
    off: { select: { kv_offload: false } },
    ordinary: { roles: { prefill: { tp: 4, pcp: 1 } } },
  };
  const q = new URLSearchParams({ hardware: "gb300", strategy: "pd_cluster", preset: key });
  for (const [point, link, requested] of [
    ["inherited", null, "simple"],
    ["store", null, "kv_store_distributed_mooncake"],
    ["store", "simple", "simple"],
    ["store", "off", null],
    ["off", null, null],
  ]) {
    q.set("point", point);
    if (link) q.set("kv_offload", link);
    else q.delete("kv_offload");
    const resolved = resolveSelection(r, q, strategies, taxonomy);
    const { rendering, selection } = resolved;
    assert.equal(selection.kv_offload, null);
    assert.equal(rendering.prefill.pcp, 4);
    const kv = JSON.parse(value(rendering.prefill.argv, "--kv-transfer-config"));
    assert.equal(kv.kv_connector, "NixlConnector");
    assert.equal(kv.kv_role, "kv_producer");
    const markdown = selectionMarkdown(r, resolved, { strategies, taxonomy,
      urls: { page: "https://example.org/test/native-pcp", json: "https://example.org/test/native-pcp.json" } });
    assert.ok(markdown.includes(rendering.prefill.command));
    const adjustment = selection.adjusted.find(({ param }) => param === "kv_offload");
    if (requested) {
      assert.equal(adjustment.requested, requested);
      assert.equal(adjustment.used, "off");
      assert.match(adjustment.reason, /PCP/);
      assert.ok(markdown.includes(`> **Note:** requested \`kv_offload=${requested}\``));
    } else {
      assert.equal(adjustment, undefined);
      assert.ok(!markdown.includes("> **Note:** requested `kv_offload="));
    }
  }

  q.set("point", "ordinary");
  const ordinary = resolveSelection(r, q, strategies, taxonomy);
  assert.equal(ordinary.selection.kv_offload, "simple");
  const transfer = JSON.parse(value(ordinary.rendering.prefill.argv, "--kv-transfer-config"));
  assert.equal(transfer.kv_connector, "MultiConnector");
  assert.deepEqual(transfer.kv_connector_extra_config.connectors.map(c => c.kv_connector),
    ["NixlConnector", "SimpleCPUOffloadConnector"]);
  assert.throws(() => render(r), /Invalid preset.*KV offload is unavailable for native PCP/);
});

test("a TP preset can set a width at PCP1 without inheriting widths from a different role mode", () => {
  const r = recipe();
  r.strategy_overrides = { pd_cluster: {
    prefill: { parallelism: "tep", tp: 8 }, decode: { parallelism: "dep", tp: 1 },
  } };
  const p = r.presets[key];
  p.roles = { prefill: { nodes: 4, parallelism: "tp" }, decode: { nodes: 4, parallelism: "tp" } };
  p.points = {
    default: { label: "Full pools", default: true },
    explicit: { label: "Explicit", roles: { prefill: { tp: 8 } } },
  };
  const q = new URLSearchParams({ hardware: "gb300", strategy: "pd_cluster", preset: key });
  const widths = () => {
    const { rendering } = resolveSelection(r, q, strategies, taxonomy);
    return [rendering.prefill, rendering.decode].map(role => Number(value(role.argv, "--tensor-parallel-size")));
  };
  // Kimi-K3 switches both pools to TP across four trays. Its inherited
  // TEP/DEP defaults would incorrectly produce TP8 and TP1.
  assert.deepEqual(widths(), [16, 16]);
  q.set("point", "explicit");
  assert.deepEqual(widths(), [8, 16]);

  p.roles.prefill.tp = 8;
  p.points.resized = { label: "Resize", roles: { prefill: { nodes: 3 } } };
  q.set("point", "resized");
  assert.deepEqual(widths(), [12, 16]);
  p.points.changedMode = { label: "Switch to TEP", roles: { prefill: { parallelism: "tep" } } };
  q.set("point", "changedMode");
  assert.deepEqual(widths(), [16, 16]);
});

test("the producer's PCP size cannot activate a legacy TP width on the consumer", () => {
  const r = recipe();
  delete r.presets[key].roles.decode.parallelism;
  r.strategy_overrides = { pd_cluster: { decode: { parallelism: "tp", tp: 1 } } };
  const result = render(r);
  assert.equal(result.prefill.pcp, 4);
  assert.equal(result.decode.tp, 4);
});

test("resizing a PCP pool rederives TP when its preset inherits the strategy mode", () => {
  const r = recipe();
  delete r.presets[key].roles.prefill.parallelism;
  r.presets[key].roles.prefill.args = [];
  r.strategy_overrides = { pd_cluster: { prefill: { parallelism: "tp" } } };
  const q = new URLSearchParams({ hardware: "gb300", strategy: "pd_cluster", preset: key });
  assert.equal(resolveSelection(r, q, strategies, taxonomy).rendering.prefill.tp, 1);
  q.set("prefill_nodes", "2");
  const { prefill } = resolveSelection(r, q, strategies, taxonomy).rendering;
  assert.equal(prefill.nodes, 2);
  assert.equal(prefill.tp, 2);
  assert.equal(prefill.pcp, 4);
});

test("a PCP preset keeps TP for a device remap and rederives it when the pool grows", () => {
  const r = recipe();
  const preset = r.presets[key];
  preset.hardware = "h200";
  preset.roles.prefill = { nodes: 1, parallelism: "tp", devices: [0, 2, 4, 6], tp: 1, pcp: 2 };
  preset.points = {
    sparse: { default: true },
    full: { roles: { prefill: { devices: [0, 1, 2, 3, 4, 5, 6, 7] } } },
  };
  const query = new URLSearchParams({ hardware: "h200", strategy: "pd_cluster", preset: key });
  const prefill = () => resolveSelection(r, query, strategies, taxonomy).rendering.prefill;
  query.set("prefill_devices", "1,3,5,7");
  const remapped = prefill();
  assert.equal(remapped.tp, 1);
  assert.equal(remapped.env.CUDA_VISIBLE_DEVICES, "1,3,5,7");
  query.set("prefill_devices", "0-7");
  assert.equal(value(prefill().argv, "--tensor-parallel-size"), "4");
  query.delete("prefill_devices");
  query.set("point", "full");
  assert.equal(prefill().tp, 4);
});

test("points only inherit a PD role TP at its authored mode and TP/TEP pool size", () => {
  const r = recipe();
  const p = r.presets[key];
  p.roles.prefill = { nodes: 1, tp: 1, pcp: 2 };
  p.points = {
    same: { roles: { prefill: { parallelism: "tep" } } },
    resized: { roles: { prefill: { nodes: 2 } } },
    switched: { roles: { prefill: { parallelism: "tp" } } },
    pinned: { roles: { prefill: { nodes: 2, tp: 1 } } },
  };
  // The preset inherits this mode through a compute-architecture override.
  const base = { parallelism: { "arch:sm103": "tep", default: "tp" }, pcp: 2, tp: 1 };
  r.strategy_overrides = { pd_cluster: { prefill: base } };
  const q = new URLSearchParams({ hardware: "gb300", strategy: "pd_cluster", preset: key });
  function pool(point) {
    q.set("point", point);
    return resolveSelection(r, q, strategies, taxonomy).rendering.prefill;
  }
  assert.equal(pool("same").tp, 1);
  assert.equal(pool("resized").tp, 4);
  assert.equal(pool("switched").tp, 2);
  assert.equal(pool("pinned").tp, 1);

  base.parallelism = "dep";
  p.roles.prefill.tp = 2;
  const dep = pool("resized");
  assert.equal(dep.nodes, 2);
  assert.equal(dep.tp, 2);
  assert.equal(dep.dpLocal, 1);
  assert.equal(dep.dpSize, 2);
});

test("native NIXL PCP shares TP/TEP engines across nodes and registers just the head at its served port", () => {
  const r = recipe();
  const p = r.presets[key].roles.prefill;
  Object.assign(p, { nodes: 2, parallelism: "tep", args: ["--port=7201"] });
  delete p.tp;
  const result = render(r);
  assert.equal(result.prefill.tp, 2);
  assert.equal(result.prefill.node_commands.length, 2);
  const follower = result.prefill.node_commands[1].argv;
  assert.equal(value(follower, "--node-rank"), "1");
  assert.ok(follower.includes("--headless"));
  assert.ok(follower.includes("--enable-expert-parallel"));
  assert.match(result.router.command, /--prefill http:\/\/\$PREFILL_NODE_1:7201/);
  assert.doesNotMatch(result.router.command, /PREFILL_NODE_2/);
});

test("native DEP sizes internal DP from TP × PCP without an Ascend rank launcher", () => {
  const r = recipe();
  Object.assign(r.presets[key].roles.prefill, { parallelism: "dep", pcp: 2, args: [] });
  const result = render(r);
  assert.equal(result.prefill.dpLocal, 2);
  assert.equal(value(result.prefill.argv, "--data-parallel-size-local"), "2");
  assert.equal(result.prefill.rank_commands, undefined);
  assert.deepEqual(result.prefill.argv.slice(0, 2), ["vllm", "serve"]);
  assert.ok(!result.prefill.argv.includes("--data-parallel-start-rank"));
  assert.match(result.router.command, /--intra-node-data-parallel-size 2/);
  r.presets[key].roles.prefill.nodes = 2;
  const cluster = render(r);
  const next = cluster.prefill.node_commands[1].argv;
  assert.equal(value(next, "--data-parallel-start-rank"), "2");
  assert.equal(value(next, "--data-parallel-size"), "4");
  assert.ok(next.includes("--data-parallel-hybrid-lb"));
  assert.match(cluster.router.command, /PREFILL_NODE_2:8001/);

  r.presets[key].roles.prefill.devices = [1, 3];
  const subset = render(r).prefill;
  assert.equal(subset.dpLocal, 1);
  assert.equal(subset.dpSize, 2);
  assert.equal(value(subset.node_commands[1].argv, "--data-parallel-start-rank"), "1");
});

test("native PD rejects contradictory device, port, connector and DCP settings", () => {
  for (const [role, delta, expected] of [
    ["prefill", { tp: 2 }, /evenly/],
    ["prefill", { nodes: 0 }, /dedicated/],
    ["decode", { nodes: 0 }, /dedicated/],
    ["prefill", { args: ["--port=65536"] }, /HTTP port/],
    ["prefill", { args: ["--data-parallel-size", "2"] }, /topology/],
    ["prefill", { args: ["--data-parallel-rank", "1"] }, /topology/],
    ["prefill", { args: ["-n", "2"] }, /topology/],
    ["prefill", { args: ["-r=1"] }, /topology/],
    ["prefill", { parallelism: "dep", args: ["--data-parallel-external-lb"] }, /topology/],
    ["prefill", { parallelism: "dep", args: ["--data-parallel-multi-port-external-lb"] }, /topology/],
    ["prefill", { args: ["--data-parallel-hybrid-lb"] }, /topology/],
    ["prefill", { args: ["--decode_context_parallel_size=2"] }, /DCP equal to PCP/],
    ["prefill", { args: ["-dcp", "4", "--dcp_comm_backend=a2a"] }, /ag_rs/],
    ["prefill", { json_args: { "--kv-transfer-config": { NixlConnector: { kv_role: "kv_both" } } } }, /kv_producer/],
    ["prefill", { json_args: { "--kv-transfer-config": { NixlConnector: { kv_connector_extra_config: { bidirectional_kv_xfer: true } } } } }, /bidirectional/],
    ["decode", { json_args: { "--kv-transfer-config": { NixlConnector: { kv_role: "kv_producer" } } } }, /kv_consumer/],
    ["decode", { tp: 3 }, /transfer TP/],
  ]) {
    const r = recipe();
    Object.assign(r.presets[key].roles[role], delta);
    assert.throws(() => render(r), expected);
  }
  const dcp = recipe();
  dcp.presets[key].roles.prefill.args = [];
  dcp.presets[key].roles.decode.args = ["-dcp", "2"];
  assert.throws(() => render(dcp), /replicated PCP/);
  assert.throws(() => render(recipe(), { pdConnector: "mooncake" }), /NIXL/);
});

test("restating a structured PD role mode in a preset or point preserves its strategy TP", () => {
  const r = recipe();
  r.strategy_overrides = { pd_cluster: {
    prefill: { nodes: 1, parallelism: { "arch:sm103": "tp", default: "tep" }, tp: 1, pcp: 2 },
    decode: { nodes: 1, parallelism: "tp" },
  } };
  const preset = r.presets[key];
  preset.roles = { prefill: { parallelism: "tp" } };
  preset.points = {
    base: { default: true },
    same: { roles: { prefill: { parallelism: "tp" } } },
    changed: { roles: { prefill: { parallelism: "tep" } } },
  };
  const query = new URLSearchParams({ hardware: "gb300", strategy: "pd_cluster", preset: "baseline" });
  const prefill = () => resolveSelection(r, query, strategies, taxonomy).rendering.prefill;
  const baseline = prefill();
  assert.equal(value(baseline.argv, "--tensor-parallel-size"), "1");
  query.set("preset", key);
  assert.deepEqual(prefill().argv, baseline.argv);
  query.set("point", "same");
  assert.deepEqual(prefill().argv, baseline.argv);
  delete preset.roles.prefill.parallelism;
  assert.deepEqual(prefill().argv, baseline.argv);
  query.set("prefill_nodes", "2");
  assert.equal(value(prefill().argv, "--tensor-parallel-size"), "4");
  query.delete("prefill_nodes");
  query.set("point", "changed");
  assert.equal(value(prefill().argv, "--tensor-parallel-size"), "2");
});
