import assert from "node:assert/strict";
import { test } from "node:test";
import { IncompatibleDecodeTp, resolveCommand } from "../src/lib/command-synthesis.js";
import { renderCommand, resolveSelection } from "../src/lib/recipe-render.js";
import { selectionMarkdown } from "../src/lib/recipe-markdown.js";
import { loadStrategies } from "../src/lib/strategies.js";
import { loadTaxonomy } from "../src/lib/taxonomy.js";

const taxonomy = loadTaxonomy();
const strategies = loadStrategies();

function fixture(hardware, nodes = 1) {
  return {
    hf_id: "test/Ascend-PD", model: { model_id: "test/Ascend-PD-W8A8", architecture: "moe" },
    meta: { hardware: { [hardware]: "supported" } },
    variants: { default: { precision: "int8", vram_minimum_gb: 64, supported_hardware: [hardware] } },
    compatible_strategies: ["single_node_tp", "multi_node_dep", "pd_cluster"],
    presets: { ascend_pd: {
      hardware, strategy: "pd_cluster", select: { pd_router: "ascend", pd_connector: "mooncake_ascend" },
      args: ["--port", "7100", "--data-parallel-rpc-port", "12321"],
      json_args: { "--kv-transfer-config": { kv_connector: "MooncakeHybridConnector" } },
      roles: {
        prefill: { nodes, parallelism: "dep", tp: 1, pcp: 4 },
        decode: { nodes: 1, parallelism: "dep", tp: 1 },
      },
    } },
  };
}

function render(r) {
  const p = r.presets.ascend_pd;
  return renderCommand(r, "default", "pd_cluster", p.hardware, 2, [], strategies, taxonomy,
    p.roles, null, "ascend_pd", undefined, true);
}

function option(argv, flag) {
  const index = argv.indexOf(flag);
  assert.notEqual(index, -1, flag);
  return argv[index + 1];
}

test("Ascend PCP emits one standard multi-port launch per node and advertises the same layout to its peers", () => {
  for (const [hw, devices] of [["ascend_910b", 8], ["ascend_910c", 16]]) {
    const r = render(fixture(hw, 2));
    const local = devices / 4;
    for (const [name, role, count, kvRole] of [
      ["prefill", r.prefill, local, "kv_producer"], ["decode", r.decode, devices, "kv_consumer"],
    ]) {
      assert.equal(role.node_commands.length, name === "prefill" ? 2 : 1);
      for (const node of role.node_commands) {
        const argv = node.argv;
        assert.deepEqual(argv.slice(0, 3), ["vllm", "serve", "test/Ascend-PD-W8A8"]);
        assert.match(node.command, /^vllm serve /);
        assert.ok(argv.includes("--data-parallel-multi-port-external-lb"));
        assert.equal(option(argv, "--data-parallel-size-local"), String(count));
        assert.equal(option(argv, "--data-parallel-size"), String(role.nodes * count));
        assert.equal(option(argv, "--data-parallel-start-rank"), String(node.node_rank * count));
        assert.equal(node.rank_commands, undefined);
        assert.equal(node.env.ASCEND_RT_VISIBLE_DEVICES, undefined);
        assert.ok(node.docker_argv.includes("--env=ASCEND_RT_VISIBLE_DEVICES"));
        assert.ok(node.docker_command.includes("--env=ASCEND_RT_VISIBLE_DEVICES"));
        const config = JSON.parse(option(argv, "--kv-transfer-config"));
        assert.equal(config.kv_connector, "MooncakeHybridConnector");
        assert.equal(config.kv_role, kvRole);
        assert.equal(config.kv_connector_extra_config.prefill.dp_size, 2 * local);
        assert.equal(config.kv_connector_extra_config.decode.dp_size, devices);
      }
      const ports = r.router.argv.indexOf(name === "prefill" ? "--prefiller-ports" : "--decoder-ports") + 1;
      assert.deepEqual(r.router.argv.slice(ports, ports + role.nodes * count),
        Array.from({ length: role.nodes * count }, (_, i) => String(7100 + i % count)));
    }
    assert.equal(option(r.prefill.argv, "--prefill-context-parallel-size"), "4");
    assert.ok(!r.decode.argv.includes("--prefill-context-parallel-size"));
  }
});

test("the selected PCP preset keeps its frontend, router and transfer path in shared links and Markdown", () => {
  const r = fixture("ascend_910c");
  r.presets.ascend_pd.roles.prefill.env = { ASCEND_RT_VISIBLE_DEVICES: "15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0" };
  const query = new URLSearchParams({ hardware: "ascend_910c", strategy: "pd_cluster", preset: "ascend_pd",
    prefill_mode: "tp", frontend: "rust", pd_router: "vllm", pd_connector: "nixl", kv_offload: "simple" });
  const resolved = resolveSelection(r, query, strategies, taxonomy);
  const { rendering, selection } = resolved;
  for (const [param, used] of Object.entries({
    prefill_mode: "dep", frontend: "python", pd_router: "ascend", pd_connector: "mooncake_ascend", kv_offload: "off",
  })) assert.ok(selection.adjusted.some((item) => item.param === param && item.used === used), param);
  assert.equal(rendering.prefill.env.ASCEND_RT_VISIBLE_DEVICES, r.presets.ascend_pd.roles.prefill.env.ASCEND_RT_VISIBLE_DEVICES);
  const page = "https://recipes.vllm.ai/test/Ascend-PD";
  const markdown = selectionMarkdown(r, resolved, { strategies, taxonomy, urls: { page, json: page + ".json" } });
  assert.ok(markdown.includes(rendering.prefill.command));
  assert.ok(markdown.includes(rendering.decode.command));
  assert.ok(markdown.includes(rendering.router.command));
  assert.doesNotMatch(markdown, /ASCEND_PCP_LOCAL_RANK|bash -c|launch_online_dp/);
  // A published preset must declare a usable direct-transfer deployment.
  r.presets.ascend_pd.select.kv_offload = "simple";
  assert.throws(() => render(r), /Ascend PCP requires direct Mooncake/);
  query.delete("kv_offload");
  const fallback = resolveSelection(r, query, strategies, taxonomy);
  assert.deepEqual(fallback.rendering.prefill.argv, rendering.prefill.argv);
  assert.match(fallback.selection.adjusted.find((a) => a.param === "kv_offload").reason, /direct Mooncake/);
});

test("the single-endpoint A2 internal-DP layout keeps its explicit device mask", () => {
  const recipe = fixture("ascend_910b");
  const { prefill, decode } = recipe.presets.ascend_pd.roles;
  delete prefill.pcp;
  prefill.data_parallel_lb = decode.data_parallel_lb = "internal";
  const rendered = render(recipe);
  for (const worker of [rendered.prefill, rendered.decode]) {
    assert.equal(worker.env.ASCEND_RT_VISIBLE_DEVICES, "0,1,2,3,4,5,6,7");
    assert.doesNotMatch(worker.docker_command, /--env=ASCEND_RT_VISIBLE_DEVICES/);
  }
});

test("Ascend PCP links keep Decode modes compatible with the final connector and pool width", () => {
  const r = fixture("ascend_910c");
  r.compatible_strategies.push("multi_node_tep");
  const preset = r.presets.ascend_pd;
  const query = new URLSearchParams({ hardware: preset.hardware, strategy: "pd_cluster", preset: "ascend_pd" });
  const select = (mode) => {
    query.set("decode_mode", mode);
    return resolveSelection(r, query, strategies, taxonomy);
  };
  const baseline = select("dep");
  assert.ok(select("unknown").selection.adjusted.every(({ reason }) => !reason.includes("MooncakeHybridConnector")));
  for (const mode of ["tp", "tep"]) {
    const resolved = select(mode);
    const { rendering, selection } = resolved;
    assert.deepEqual(rendering, baseline.rendering);
    assert.equal(selection.pd.decode.parallelism, "dep");
    const reason = "incompatible with MooncakeHybridConnector because Decode TP exceeds Prefill TP";
    assert.deepEqual(selection.adjusted, [{ param: "decode_mode", requested: mode, used: "dep", reason }]);
    const page = "https://recipes.vllm.ai/test/Ascend-PD";
    const markdown = selectionMarkdown(r, resolved, { strategies, taxonomy, urls: { page, json: page + ".json" } });
    assert.ok(markdown.includes(`requested \`decode_mode=${mode}\` is ${reason}; rendered \`dep\` instead.`));
    const pools = { prefill: preset.roles.prefill, decode: { nodes: 1, parallelism: mode } };
    assert.throws(
      () => resolveCommand(r, "default", "pd_cluster", preset.hardware, [], strategies, taxonomy,
        [], 2, pools, {}, null, null, "python", "ascend", "http", null, "ascend_pd", undefined, "mooncake_ascend"),
      IncompatibleDecodeTp,
    );
  }
  // A narrow explicit TP remains usable at its declared node count.
  preset.roles.decode.parallelism = "tp";
  assert.equal(select("tp").rendering.decode.tp, 1);
  assert.equal(select("tep").selection.pd.decode.parallelism, "tp");
  query.set("decode_nodes", "2");
  assert.equal(select("tp").rendering.decode.parallelism, "dep");
  // Unlike Hybrid, V2 permits the wider Decode pool.
  preset.json_args["--kv-transfer-config"].kv_connector = "MooncakeConnectorV2";
  for (const mode of ["tp", "tep"]) {
    const { rendering, selection } = select(mode);
    assert.equal(rendering.decode.parallelism, mode);
    assert.equal(rendering.decode.tp, 32);
    assert.deepEqual(selection.adjusted, []);
  }
  preset.json_args["--kv-transfer-config"].kv_connector = "MooncakeHybridConnector";
  delete preset.roles.decode.tp;
  assert.throws(() => render(r), /Prefill TP >= Decode TP/);
});

test("Ascend PCP validates the final connector and addresses after preset edits", () => {
  const recipe = fixture("ascend_910c");
  const preset = recipe.presets.ascend_pd;
  preset.roles.decode.json_args = { "--kv-transfer-config": { kv_connector_extra_config: { prefill: { dp_size: 16 } } } };
  assert.throws(() => render(recipe), /must match the role topology/);
  preset.roles.decode.json_args = { "--kv-transfer-config": { kv_connector: "MultiConnector",
    kv_connector_extra_config: { connectors: [{ kv_connector: "MooncakeHybridConnector", kv_role: "kv_consumer" }] } } };
  assert.throws(() => render(recipe), /matching direct/);
  delete preset.roles.decode.json_args;
  preset.roles.prefill.args = ["--port", "65535"];
  assert.throws(() => render(recipe), /HTTP ports/);
  preset.roles.prefill.args = ["-dpp=12345"];
  const argv = render(recipe).prefill.argv;
  assert.equal(argv.filter((arg) => arg === "--data-parallel-rpc-port").length, 1);
  assert.equal(option(argv, "--data-parallel-rpc-port"), "12345");
  preset.roles.prefill.args = ["--data-parallel-rank", "0"];
  assert.throws(() => render(recipe), /conflict with the selected DP mode/);
});
