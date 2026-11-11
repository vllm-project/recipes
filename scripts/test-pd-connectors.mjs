import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { test } from "node:test";
import yaml from "js-yaml";
import { buildDockerArgv, buildDockerRun, computeDockerMeta, pdConnectorKey, resolveCommand } from "../src/lib/command-synthesis.js";
import { applyPresetArgOps, validatePresets } from "../src/lib/recipe-presets.js";
import { contentDir } from "../src/lib/content-root.js";
import { ascendPdPools, ascendPdRecipe } from "./fixtures/ascend-pd.mjs";

// Recipes live in their source repos; read the assembled tree (run
// build-recipes-api or assemble-content first).
const read = (file) => yaml.load(fs.readFileSync(path.join(contentDir(), file), "utf8"));
const recipe = read("models/XiaomiMiMo/MiMo-V2.6-Flash-RL.yaml");
const taxonomy = read("taxonomy.yaml");
const strategies = Object.fromEntries(["strategies", "kv_store"].flatMap((dir) => fs.readdirSync(path.join(contentDir(), dir))
  .filter((file) => file.endsWith(".yaml"))
  .map((file) => { const s = read(`${dir}/${file}`); return [s.name, s]; })));
// The connector catalog belongs to this renderer, not a remote content source.
strategies.pd_cluster = yaml.load(fs.readFileSync(new URL("../strategies/pd_cluster.yaml", import.meta.url), "utf8"));
const flag = (argv, key) => argv[argv.lastIndexOf(key) + 1];
const pools = { prefill: { nodes: 1 }, decode: { nodes: 1 } };
const resolve = (strats, kvOffload = null) =>
  resolveCommand(recipe, "default", "pd_cluster", "gb300", [], strats, taxonomy, [], 1, pools, {}, kvOffload);
// resolveCommand's positional tail: … frontend, pdRouter, pdTransport, dcp, preset, point, pdConnector.
const pick = ({ hw = "gb300", connector = null, router = null, kv = null } = {}) =>
  resolveCommand(recipe, "default", "pd_cluster", hw, [], strategies, taxonomy, [], 1, pools, {}, kv, null, undefined, router, "http", null, undefined, undefined, connector);
const kvCfg = (r, role) => JSON.parse(flag(r[role].argv, "--kv-transfer-config"));

test("the default connector is the one marked default, else the first", () => {
  const pd = strategies.pd_cluster;
  assert.equal(pdConnectorKey(pd), "nixl");
  assert.equal(pdConnectorKey(pd, "nope"), "nixl");
  assert.equal(pdConnectorKey({ connectors: { a: {}, b: {} } }), "a");
  assert.equal(pdConnectorKey({ connectors: { a: {}, b: { default: true } } }), "b");
  assert.equal(pdConnectorKey({}), null);
});

test("each role renders its connector's config and env", () => {
  const r = resolve(strategies);
  for (const [role, kvRole, port] of [["prefill", "kv_producer", "5557"], ["decode", "kv_consumer", "5558"]]) {
    assert.deepEqual(JSON.parse(flag(r[role].argv, "--kv-transfer-config")),
      { kv_connector: "NixlConnector", kv_role: kvRole, kv_load_failure_policy: "fail" });
    assert.equal(r[role].env.VLLM_NIXL_SIDE_CHANNEL_PORT, port);
  }
});

test("the Mooncake store wraps whichever transfer connector is selected", () => {
  // A second connector made the default stands in for Mooncake / MoRI-IO.
  const other = {
    default: true,
    prefill: { config: { kv_connector: "OtherConnector", kv_role: "kv_producer" }, env: { OTHER_PORT: "1" } },
    decode: { config: { kv_connector: "OtherConnector", kv_role: "kv_consumer" } },
  };
  const strats = { ...strategies, pd_cluster: { ...strategies.pd_cluster, connectors: { nixl: { ...strategies.pd_cluster.connectors.nixl, default: false }, other } } };
  const plain = resolve(strats);
  assert.equal(JSON.parse(flag(plain.prefill.argv, "--kv-transfer-config")).kv_connector, "OtherConnector");
  assert.equal(plain.prefill.env.OTHER_PORT, "1");
  assert.equal(plain.prefill.env.VLLM_NIXL_SIDE_CHANNEL_PORT, undefined);

  const stored = resolve(strats, "kv_store_distributed_mooncake");
  for (const role of ["prefill", "decode"]) {
    const cfg = JSON.parse(flag(stored[role].argv, "--kv-transfer-config"));
    assert.equal(cfg.kv_connector, "MultiConnector");
    assert.deepEqual(cfg.kv_connector_extra_config.connectors.map((c) => c.kv_connector), ["OtherConnector", "MooncakeStoreConnector"]);
    assert.equal(stored[role].argv.filter((t) => t === "--kv-transfer-config").length, 1);
  }
});

test("Mooncake: producer/consumer workers, bootstrap port on prefill, mooncake router with bootstrap endpoints", () => {
  const r = pick({ connector: "mooncake" });
  assert.equal(r.kvConnector.selected, "mooncake");
  assert.equal(r.kvConnector.unavailableReason, null);
  assert.deepEqual([kvCfg(r, "prefill").kv_connector, kvCfg(r, "prefill").kv_role, kvCfg(r, "decode").kv_role],
    ["MooncakeConnector", "kv_producer", "kv_consumer"]);
  assert.equal(r.prefill.env.VLLM_MOONCAKE_BOOTSTRAP_PORT, "8998");
  assert.equal(r.prefill.env.VLLM_NIXL_SIDE_CHANNEL_PORT, undefined);
  assert.match(r.router.command, /--kv-connector mooncake/);
  assert.match(r.router.command, /--prefill http:\/\/\$PREFILL_NODE_1:8001 8998 \\/);
  assert.match(r.router.command, /--decode http:\/\/\$DECODE_NODE_1:8002 \\/);
  assert.ok(r.kvConnector.install.some((s) => /mooncake-transfer-engine/.test(s.command)));
});

test("MoRI-IO: workers register with the router, which discovers them; addresses stay shell-expandable", () => {
  const r = pick({ hw: "mi355x", connector: "moriio" });
  assert.equal(r.kvConnector.selected, "moriio");
  for (const [role, port] of [["prefill", "8001"], ["decode", "8002"]]) {
    const extra = kvCfg(r, role).kv_connector_extra_config;
    assert.equal(kvCfg(r, role).kv_connector, "MoRIIOConnector");
    assert.equal(extra.proxy_ip, "$ROUTER_HOST");
    assert.equal(extra.http_port, port, "{port} resolves to the role's served port");
    // The rendered command splices the placeholder out of the single quotes.
    assert.ok(r[role].command.includes(`"proxy_ip":"'"$ROUTER_HOST"'"`), r[role].command);
  }
  assert.notEqual(kvCfg(r, "prefill").kv_connector_extra_config.handshake_port, kvCfg(r, "decode").kv_connector_extra_config.handshake_port);
  assert.match(r.router.command, /--kv-connector moriio \\\n\s+--vllm-discovery-address 0\.0\.0\.0:36367/);
  assert.doesNotMatch(r.router.command, /--prefill|--decode/);
  assert.equal(r.kvConnector.docker.host_network, true);
});

test("a connector the hardware or router can't run falls back to NIXL with the reason", () => {
  const amdOnly = pick({ hw: "gb300", connector: "moriio" });
  assert.equal(amdOnly.kvConnector.selected, "nixl");
  assert.match(amdOnly.kvConnector.unavailableReason, /AMD/);
  const dynamo = pick({ connector: "mooncake", router: "dynamo" });
  assert.equal(dynamo.kvConnector.selected, "nixl");
  assert.match(dynamo.kvConnector.unavailableReason, /vLLM Router/);
});

test("Simple CPU offload composes on prefill only, around the selected connector", () => {
  const r = pick({ hw: "mi355x", connector: "moriio", kv: "simple" });
  assert.deepEqual(kvCfg(r, "prefill").kv_connector_extra_config.connectors.map((c) => c.kv_connector),
    ["MoRIIOConnector", "SimpleCPUOffloadConnector"]);
  assert.equal(kvCfg(r, "decode").kv_connector, "MoRIIOConnector");
  // The tier's child still has its {port} resolved.
  assert.equal(kvCfg(r, "prefill").kv_connector_extra_config.connectors[0].kv_connector_extra_config.http_port, "8001");
  // LMCache has no `pd.roles`, so it never reaches the PD commands.
  assert.equal(kvCfg(pick({ kv: "lmcache" }), "prefill").kv_connector, "NixlConnector");
});

test("json_args addresses a MultiConnector child by connector name", () => {
  const multi = JSON.stringify({ kv_connector: "MultiConnector", kv_role: "kv_both", kv_connector_extra_config: { connectors: [
    { kv_connector: "MoRIIOConnector", kv_role: "kv_producer", kv_connector_extra_config: { read_mode: false, notify_port: "1" } },
    { kv_connector: "SimpleCPUOffloadConnector", kv_role: "kv_both", kv_connector_extra_config: { cpu_bytes_to_use_per_rank: 1, lazy_offload: false } },
  ] } });
  const [, out] = applyPresetArgOps(["--kv-transfer-config", multi], { json_args: { "--kv-transfer-config": {
    MoRIIOConnector: { kv_connector_extra_config: { read_mode: true, notify_port: null } },
    SimpleCPUOffloadConnector: { kv_connector_extra_config: { cpu_bytes_to_use_per_rank: null, cpu_bytes_to_use: 9 } },
  } } });
  const [mori, cpu] = JSON.parse(out).kv_connector_extra_config.connectors;
  assert.deepEqual(mori.kv_connector_extra_config, { read_mode: true });
  assert.deepEqual(cpu.kv_connector_extra_config, { lazy_offload: false, cpu_bytes_to_use: 9 });
  // On a plain config the name matches the value itself.
  const [, plain] = applyPresetArgOps(["--kv-transfer-config", '{"kv_connector":"NixlConnector","kv_role":"kv_producer"}'],
    { json_args: { "--kv-transfer-config": { NixlConnector: { kv_connector_extra_config: { num_threads: 4 } } } } });
  assert.equal(JSON.parse(plain).kv_connector_extra_config.num_threads, 4);
  assert.throws(() => applyPresetArgOps(["--kv-transfer-config", multi], { json_args: { "--kv-transfer-config": { MooncakeConnector: { x: 1 } } } }), /has no MooncakeConnector/);
});

test("select.pd_connector is validated against the connector's hardware and router", () => {
  const errs = (hardware, select) => validatePresets({ ...recipe, presets: { a: { label: "a", hardware, strategy: "pd_cluster", select } } },
    taxonomy, ["pd_cluster", "single_node_tp"], [], strategies.pd_cluster.connectors);
  assert.deepEqual(errs("mi355x", { pd_connector: "moriio" }), []);
  assert.ok(errs("gb300", { pd_connector: "moriio" }).some((e) => /doesn't run on gb300/.test(e)));
  assert.ok(errs("gb300", { pd_connector: "mooncake", pd_router: "dynamo" }).some((e) => /behind pd_router dynamo/.test(e)));
  assert.ok(errs("gb300", { pd_connector: "nope" }).some((e) => /unknown connector/.test(e)));
});

test("RDMA connectors add host networking and locked memory to the worker containers", () => {
  const r = pick({ connector: "mooncake" });
  const meta = computeDockerMeta(recipe, recipe.variants.default, taxonomy.hardware_profiles.gb300, "gb300");
  const { docker } = r.kvConnector;
  const cmd = buildDockerRun({ command: r.prefill.command, env: r.prefill.env, image: meta.image, gpuFlags: meta.gpuFlags, hostNetwork: docker.host_network, extraFlags: docker.flags });
  assert.match(cmd, /--network host/);
  assert.match(cmd, /--ulimit memlock=-1/);
  const argv = buildDockerArgv({ argv: r.prefill.argv, env: r.prefill.env, meta, hostNetwork: docker.host_network, extraFlags: docker.flags });
  assert.ok(argv.includes("memlock=-1") && argv.includes("host") && !argv.includes("-p"));
});

test("a role override that pins another connector keeps it, and the pick falls back with the reason", () => {
  const pinned = { ...recipe, strategy_overrides: { ...recipe.strategy_overrides, pd_cluster: { ...recipe.strategy_overrides?.pd_cluster,
    prefill: { ...recipe.strategy_overrides?.pd_cluster?.prefill, hardware_overrides: { mi355x: { extra_args: ["--kv-transfer-config", '{"kv_connector":"NixlConnector","kv_role":"kv_producer"}'] } } } } } };
  const r = resolveCommand(pinned, "default", "pd_cluster", "mi355x", [], strategies, taxonomy, [], 1, pools, {}, null, null, undefined, null, "http", null, undefined, undefined, "moriio");
  assert.equal(r.kvConnector.selected, "nixl");
  assert.match(r.kvConnector.unavailableReason, /prefill command .* sets its own --kv-transfer-config/);
  assert.doesNotMatch(r.router.command, /moriio/);
});

test("a pool's tp takes a hardware-keyed map, read by the commands and the router alike", () => {
  const keyedTp = { ...recipe, compatible_strategies: [...recipe.compatible_strategies, "multi_node_dep", "pd_cluster"], strategy_overrides: { ...recipe.strategy_overrides, pd_cluster: {
    decode: { parallelism: "dep", tp: { default: 1, b200: 4 } } } } };
  const depPools = { prefill: { nodes: 1 }, decode: { nodes: 2, parallelism: "dep" } };
  const on = (hw) => resolveCommand(keyedTp, "default", "pd_cluster", hw, [], strategies, taxonomy, [], 1, depPools, {});
  const b200 = on("b200");
  assert.equal(flag(b200.decode.argv, "--tensor-parallel-size"), "4");
  assert.equal(flag(b200.decode.argv, "--data-parallel-size"), "4");
  assert.equal(flag(b200.decode.argv, "--data-parallel-size-local"), "2");
  const b300 = on("b300");
  assert.equal(flag(b300.decode.argv, "--tensor-parallel-size"), "1");
  assert.equal(flag(b300.decode.argv, "--data-parallel-size"), "16");
  assert.equal(flag(b300.decode.argv, "--data-parallel-size-local"), "8");
  assert.match(b200.router.command, /--intra-node-data-parallel-size 2/);
  assert.match(b300.router.command, /--intra-node-data-parallel-size 8/);
});

const ascend = (r, rank = 0, connector = "mooncakev2", strats = strategies, features = [], pools = ascendPdPools(rank)) =>
  resolveCommand(r, "default", "pd_cluster", "ascend_950dt", features, strats, taxonomy, [], 1,
    pools, {}, null, null, "python", "ascend", "http", null, "ascend_pd", undefined, connector);

test("TP/TEP derives its width from the whole pool as PD grows from four to eight nodes", () => {
  const fixture = ascendPdRecipe();
  fixture.compatible_strategies.push("single_node_tep");
  // A TP hint retained from DEP must not pin a TP/TEP pool's width.
  fixture.presets.ascend_pd.roles.prefill.tp = 8;
  for (const parallelism of ["tp", "tep"]) {
    for (const nodes of [4, 8]) {
      const pools = { prefill: { nodes, rank: nodes - 1, parallelism }, decode: { nodes, rank: nodes - 1, parallelism: "dep" } };
      const r = ascend(fixture, nodes - 1, "mooncakev2", strategies, [], pools);
      assert.equal(r.prefill.tp, nodes * 2);
      assert.equal(r.prefill.tp * r.prefill.pp, r.prefill.poolGpus);
      assert.equal(flag(r.prefill.argv, "--tensor-parallel-size"), String(nodes * 2));
      assert.equal(flag(r.prefill.argv, "--node-rank"), String(nodes - 1));
      assert.equal(r.prefill.argv.includes("--enable-expert-parallel"), parallelism === "tep");
      assert.equal(r.decode.tp, 4);
      assert.equal(r.decode.dpLocal, 2);
      assert.equal(r.decode.dpSize, nodes * 2);
      assert.equal(flag(r.decode.argv, "--data-parallel-start-rank"), String((nodes - 1) * 2));
      assert.deepEqual(kvCfg(r, "decode").kv_connector_extra_config, {
        prefill: { dp_size: 1, tp_size: nodes * 2, pp_size: 4 },
        decode: { dp_size: nodes * 2, tp_size: 4, pp_size: 1 },
      });
      assert.equal(r.router.argv.slice(r.router.argv.indexOf("--decoder-ports") + 1).length, nodes * 2);
    }
  }
});

test("Ascend DEP uses native external DP modes with every rank and endpoint", () => {
  for (const tp of [8, 4, 2]) {
    const fixture = ascendPdRecipe();
    fixture.presets.ascend_pd.roles.decode.tp = tp;
    const local = 8 / tp;
    for (let rank = 0; rank < 4; rank++) {
      const r = ascend(fixture, rank);
      const argv = r.decode.argv;
      assert.equal(flag(argv, "--tensor-parallel-size"), String(tp));
      assert.equal(flag(argv, "--data-parallel-size-local"), String(local));
      assert.equal(flag(argv, "--data-parallel-size"), String(4 * local));
      assert.ok(!argv.includes("--data-parallel-hybrid-lb"));
      assert.equal(argv.includes("--data-parallel-multi-port-external-lb"), local > 1);
      assert.equal(argv.includes("--data-parallel-external-lb"), local === 1);
      const rankFlag = local > 1 ? "--data-parallel-start-rank" : "--data-parallel-rank";
      assert.equal(flag(argv, rankFlag), String(rank * local));
      assert.ok(!argv.includes(local > 1 ? "--data-parallel-rank" : "--data-parallel-start-rank"));
      assert.equal(flag(argv, "--data-parallel-address"), "$DECODE_NODE_1");
      assert.equal(r.decode.env.HCCL_IF_IP, `$DECODE_NODE_${rank + 1}`);
      assert.equal(flag(r.prefill.argv, "--tensor-parallel-size"), "8");
      assert.equal(flag(r.prefill.argv, "--pipeline-parallel-size"), "4");
      assert.equal(flag(r.prefill.argv, "--nnodes"), "4");
      assert.equal(flag(r.prefill.argv, "--node-rank"), String(rank));
      assert.equal(flag(r.prefill.argv, "--master-addr"), "$PREFILL_NODE_1");
      assert.equal(r.prefill.argv.includes("--headless"), rank > 0);
      assert.equal(r.prefill.env.VLLM_HOST_IP, `$PREFILL_NODE_${rank + 1}`);
      // Only the TP/PP serving head is an HTTP backend; every external DP
      // rank is one, including those sharing a physical decode node.
      assert.equal(r.router.argv.filter((arg) => arg === "$PREFILL_NODE_1").length, 1);
      for (let node = 1; node <= 4; node++) {
        assert.equal(r.router.argv.filter((arg) => arg === `$DECODE_NODE_${node}`).length, local);
      }
      const ports = r.router.argv.slice(r.router.argv.indexOf("--decoder-ports") + 1);
      assert.deepEqual(ports, Array.from({ length: 4 }, () => Array.from({ length: local }, (_, i) => String(8002 + i))).flat());
      assert.doesNotMatch(r.router.command, /launch_online_dp|run_dp_template/);
    }
  }
});

test("PD topology placeholders, role edits and connector selection share the final pool layout", () => {
  const fixture = ascendPdRecipe();
  const r = ascend(fixture, 0, "mooncakev2", strategies, ["spec_decoding"]);
  for (const role of ["prefill", "decode"]) {
    assert.deepEqual(kvCfg(r, role).kv_connector_extra_config, {
      prefill: { dp_size: 1, tp_size: 8, pp_size: 4 },
      decode: { dp_size: 8, tp_size: 4, pp_size: 1 },
    });
    assert.equal(r[role].argv[2], "test/Ascend-PD-W8A8");
  }
  assert.ok(!r.prefill.argv.includes("--max-model-len"));
  assert.equal(flag(r.decode.argv, "--max-model-len"), "1024");
  assert.equal(JSON.parse(flag(r.decode.argv, "--speculative-config")).num_speculative_tokens, 3);
  for (const role of ["prefill", "decode"]) assert.ok(!ascend(fixture)[role].argv.includes("--speculative-config"));

  const custom = structuredClone(strategies);
  custom.pd_cluster.connectors.other = {
    brands: ["Huawei"], routers: ["ascend"],
    prefill: { config: { kv_connector: "OtherConnector", kv_role: "kv_producer", kv_connector_extra_config: { tp: "{prefill_tp_size}" } } },
    decode: { config: { kv_connector: "OtherConnector", kv_role: "kv_consumer" } },
  };
  const other = ascend(fixture, 0, "other", custom);
  assert.equal(other.kvConnector.selected, "other");
  assert.deepEqual(kvCfg(other, "prefill"), { kv_connector: "OtherConnector", kv_role: "kv_producer", kv_connector_extra_config: { tp: 8 } });
  assert.equal(kvCfg(other, "decode").kv_connector, "OtherConnector");

  for (const pp of [5, 3]) {
    fixture.presets.ascend_pd.roles.prefill.pp = pp;
    assert.throws(() => ascend(fixture), RangeError, "invalid TP/PP capacity must not produce commands");
  }
  const inconsistent = ascendPdRecipe();
  inconsistent.presets.ascend_pd.roles.decode.args = ["--tensor-parallel-size", "2"];
  assert.throws(() => ascend(inconsistent), RangeError, "late flags must not disagree with the router and connector topology");
});
