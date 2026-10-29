import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { test } from "node:test";
import yaml from "js-yaml";
import { buildDockerArgv, buildDockerRun, computeDockerMeta, normalizePdRoleNodes, pdConnectorKey, pdConnectorUnavailableReason, pdConnectorsFor, pdRoleModes, pdRoleNodesPerInstance, resolveCommand } from "../src/lib/command-synthesis.js";
import { applyPresetArgOps, validatePresets } from "../src/lib/recipe-presets.js";
import { contentDir } from "../src/lib/content-root.js";

// Recipes live in their source repos; read the assembled tree (run
// build-recipes-api or assemble-content first).
const read = (file) => yaml.load(fs.readFileSync(path.join(contentDir(), file), "utf8"));
const recipe = read("models/XiaomiMiMo/MiMo-V2.6-Flash-RL.yaml");
const taxonomy = read("taxonomy.yaml");
const strategies = Object.fromEntries(["strategies", "kv_store"].flatMap((dir) => fs.readdirSync(path.join(contentDir(), dir))
  .filter((file) => file.endsWith(".yaml"))
  .map((file) => { const s = read(`${dir}/${file}`); return [s.name, s]; })));
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

// Four-node Kimi K3 groups are model instances, rather than one TP/DP group
// spanning the whole pool. A pool of eight nodes therefore runs two complete
// instances with separate rank-0/master endpoints.
const ascendRecipe = {
  ...read("models/moonshotai/Kimi-K3.yaml"),
  strategy_overrides: { pd_cluster: {
    connectors: { ascend_950dt: ["mooncakev2"] },
    prefill: {
      nodes: { ascend_950dt: 4 }, nodes_per_instance: { ascend_950dt: 4 },
      parallelism: { ascend_950dt: "tp" }, tp: { ascend_950dt: 8 }, pp: { ascend_950dt: 4 },
    },
    decode: {
      nodes: { ascend_950dt: 4 }, nodes_per_instance: { ascend_950dt: 4 },
      parallelism: { ascend_950dt: "dep" }, tp: { ascend_950dt: 8 }, pp: { ascend_950dt: 1 },
    },
  } },
};
const a5 = taxonomy.hardware_profiles.ascend_950dt;
const pickAscend = ({ model = ascendRecipe, features = [], pools = null, connector = null, router = null, frontend = undefined } = {}) =>
  resolveCommand(model, "ascend_mxfp4", "pd_cluster", "ascend_950dt", features, strategies, taxonomy, [], 1,
    pools, {}, null, null, frontend, router, "http", null, undefined, undefined, connector);

test("Ascend group constraints apply only to the specified hardware and role", () => {
  assert.equal(pdRoleNodesPerInstance(ascendRecipe, "prefill", a5, "ascend_950dt"), 4);
  assert.equal(pdRoleNodesPerInstance(ascendRecipe, "decode", a5, "ascend_950dt"), 4);
  assert.deepEqual(pdRoleModes(ascendRecipe, "prefill", a5, "ascend_950dt"), ["tp"]);
  assert.deepEqual(pdRoleModes(ascendRecipe, "decode", a5, "ascend_950dt"), ["dep"]);
  assert.deepEqual(pdConnectorsFor(ascendRecipe, a5, "ascend_950dt"), ["mooncakev2"]);
  assert.equal(pdConnectorsFor(ascendRecipe, taxonomy.hardware_profiles.gb300, "gb300"), null);
});

test("Ascend pool sizes normalize to complete four-node groups within the UI limit", () => {
  for (const role of ["prefill", "decode"]) {
    for (const [input, expected] of [[undefined, 4], [null, 4], [NaN, 4], [Infinity, 4], [-Infinity, 4],
      [-4, 4], [0, 4], [1, 4], [3, 4], [4, 4], [4.1, 8], [5, 8], [15.2, 16], [17, 16], [1e6, 16]]) {
      assert.equal(normalizePdRoleNodes(ascendRecipe, role, a5, "ascend_950dt", input), expected, `${role}: ${input}`);
    }
  }
});

test("Ascend defaults produce a four-node prefill group and a four-node decode group", () => {
  const r = pickAscend();
  assert.equal(r.orchestrator, "ascend-proxy");
  assert.equal(r.kvConnector.selected, "mooncakev2");
  assert.equal(r.prefill.nodes, 4);
  assert.equal(r.decode.nodes, 4);
  assert.equal(flag(r.prefill.argv, "--tensor-parallel-size"), "8");
  assert.equal(flag(r.prefill.argv, "--pipeline-parallel-size"), "4");
  assert.equal(flag(r.prefill.argv, "--nnodes"), "4");
  assert.equal(flag(r.decode.argv, "--tensor-parallel-size"), "8");
  assert.equal(flag(r.decode.argv, "--data-parallel-size"), "4");
  assert.equal(flag(r.decode.argv, "--data-parallel-size-local"), "1");
  assert.ok(!r.decode.argv.includes("--data-parallel-hybrid-lb"));
});

test("Ascend Router is a distinct selection paired with Mooncake V2", () => {
  for (const connector of [null, "mooncakev2", "nixl"]) {
    const r = pickAscend({ router: "ascend", connector });
    assert.equal(r.pdRouter, "ascend");
    assert.equal(r.orchestrator, "ascend-proxy");
    assert.equal(r.kvConnector.selected, "mooncakev2");
    assert.equal(r.routerUnavailableReason, undefined);
    assert.ok(r.ascendScripts && r.startServer, "the independent router option retains the five-step launch");
  }
});

test("adding Ascend Router retains existing GPU router and connector commands", () => {
  for (const hw of ["gb300", "mi355x"]) {
    const baseline = pick({ hw });
    const rejected = pick({ hw, router: "ascend" });
    assert.equal(rejected.pdRouter, "vllm");
    assert.match(rejected.routerUnavailableReason, /Huawei/);
    for (const role of ["prefill", "decode"]) {
      assert.deepEqual(rejected[role].argv, baseline[role].argv);
      assert.deepEqual(rejected[role].env, baseline[role].env);
    }
    assert.deepEqual(rejected.router, baseline.router);
    assert.equal(rejected.kvConnector.selected, baseline.kvConnector.selected);
  }
  assert.equal(pick({ connector: "mooncake" }).kvConnector.selected, "mooncake");
  assert.equal(pick({ hw: "mi355x", connector: "moriio" }).kvConnector.selected, "moriio");
});

test("the full connector catalog explains incompatible Ascend combinations", () => {
  const connectors = strategies.pd_cluster.connectors;
  for (const key of ["nixl", "mooncake", "moriio", "mooncakev2"]) assert.ok(connectors[key]);
  assert.ok(strategies.pd_cluster.dynamo && strategies.pd_cluster.smg && strategies.pd_cluster.ascend);
  assert.equal(pdConnectorUnavailableReason(connectors.mooncakev2, a5, "ascend", ascendRecipe, "ascend_950dt", "mooncakev2"), null);
  assert.match(pdConnectorUnavailableReason(connectors.mooncakev2, a5, "vllm"), /Ascend Router/);
  for (const key of ["nixl", "mooncake", "moriio"]) {
    assert.match(pdConnectorUnavailableReason(connectors[key], a5, "ascend", ascendRecipe, "ascend_950dt", key), /requires mooncakev2/);
  }
});

test("Ascend replicated groups restart ranks at zero and use each group's own master", () => {
  for (const [rank, localRank] of [[4, 0], [5, 1], [7, 3]]) {
    const r = pickAscend({ pools: { prefill: { nodes: 8, rank }, decode: { nodes: 8, rank } } });
    assert.equal(flag(r.prefill.argv, "--tensor-parallel-size"), "8");
    assert.equal(flag(r.prefill.argv, "--pipeline-parallel-size"), "4");
    assert.equal(flag(r.prefill.argv, "--nnodes"), "4");
    assert.equal(flag(r.prefill.argv, "--node-rank"), String(localRank));
    assert.equal(flag(r.prefill.argv, "--master-addr"), "$PREFILL_NODE_5");
    assert.ok(!r.prefill.argv.includes("--headless"));
    assert.equal(flag(r.decode.argv, "--data-parallel-size"), "4");
    assert.equal(flag(r.decode.argv, "--data-parallel-size-local"), "1");
    assert.equal(flag(r.decode.argv, "--data-parallel-rank"), String(localRank));
    assert.equal(flag(r.decode.argv, "--data-parallel-address"), "$DECODE_NODE_5");
    assert.ok(!r.decode.argv.includes("--headless"));
    assert.ok(!r.decode.argv.includes("--data-parallel-hybrid-lb"));
  }
});

test("Ascend Mooncake V2 carries the role and per-instance topology without CUDA transfer settings", () => {
  const r = pickAscend({ pools: { prefill: { nodes: 8 }, decode: { nodes: 12 } } });
  for (const [role, kvRole] of [["prefill", "kv_producer"], ["decode", "kv_consumer"]]) {
    const cfg = kvCfg(r, role);
    assert.equal(cfg.kv_connector, "MooncakeConnectorV2");
    assert.equal(cfg.kv_role, kvRole);
    assert.deepEqual(cfg.kv_connector_extra_config, {
      prefill: { dp_size: 1, tp_size: 8, pp_size: 4 }, decode: { dp_size: 4, tp_size: 8 },
    });
    assert.match(String(cfg.kv_port), /^\d+$/);
    assert.ok(Number(cfg.kv_port) >= 20000 + a5.gpu_count * 1000 && Number(cfg.kv_port) <= 65535,
      "KV side-channel port must avoid AscendDirectTransport's per-NPU reserved range");
    assert.equal(r[role].argv.filter((arg) => arg === "--kv-transfer-config").length, 1);
    assert.deepEqual(Object.keys(r[role].env).filter((key) => /^(?:NCCL_|CUDA_|UCX_)|NIXL|BOOTSTRAP/.test(key)), []);
  }
  assert.notEqual(kvCfg(r, "prefill").kv_port, kvCfg(r, "decode").kv_port);
});

test("Ascend proxy registers every decode endpoint and only prefill group leaders", () => {
  const r = pickAscend({ pools: { prefill: { nodes: 8 }, decode: { nodes: 12 } } });
  assert.match(r.router.command, /load_balance_proxy_server_example\.py/);
  const tokens = r.router.command.replace(/\\\r?\n/g, " ").trim().split(/\s+/);
  const list = (key) => {
    const start = tokens.indexOf(key);
    assert.ok(start >= 0, `${key}: ${r.router.command}`);
    const end = tokens.findIndex((token, i) => i > start && token.startsWith("--"));
    return tokens.slice(start + 1, end < 0 ? undefined : end).map((token) => token.replace(/^["']|["']$/g, ""));
  };
  assert.deepEqual(list("--prefiller-hosts"), ["$PREFILL_NODE_1", "$PREFILL_NODE_5"]);
  assert.deepEqual(list("--decoder-hosts"), Array.from({ length: 12 }, (_, i) => `$DECODE_NODE_${i + 1}`));
  assert.deepEqual(list("--prefiller-ports"), ["8001", "8001"]);
  assert.deepEqual(list("--decoder-ports"), Array(12).fill("8002"));
  assert.doesNotMatch(r.router.command, /bootstrap|--kv-connector|vllm-router|smg launch|dynamo\.frontend/);
});

test("stale URL connector, router, frontend and pool-mode selections cannot escape Ascend constraints", () => {
  for (const router of ["smg", "dynamo", "vllm"]) {
    const r = pickAscend({ router, connector: "nixl", frontend: "rust", pools: {
      prefill: { nodes: 5, parallelism: "dep" }, decode: { nodes: 3, parallelism: "tp" },
    } });
    assert.equal(r.orchestrator, "ascend-proxy");
    assert.equal(r.frontendRequired, "python");
    assert.equal(r.kvConnector.selected, "mooncakev2");
    assert.equal(r.prefill.nodes, 8);
    assert.equal(r.decode.nodes, 4);
    assert.equal(flag(r.prefill.argv, "--pipeline-parallel-size"), "4");
    assert.equal(flag(r.decode.argv, "--data-parallel-size"), "4");
    for (const role of ["prefill", "decode"]) assert.notEqual(r[role].env.VLLM_USE_RUST_FRONTEND, "1");
  }
});

test("the assembled Kimi K3 recipe supplies the A5 PD service settings and role-specific draft execution", () => {
  const model = read("models/moonshotai/Kimi-K3.yaml");
  const r = pickAscend({ model, features: ["spec_decoding"] });
  assert.equal(r.orchestrator, "ascend-proxy");
  assert.equal(r.prefill.nodes, 4);
  assert.equal(r.decode.nodes, 4);
  for (const role of ["prefill", "decode"]) {
    const worker = r[role];
    assert.equal(kvCfg(r, role).kv_connector, "MooncakeConnectorV2");
    assert.equal(flag(worker.argv, "--max-model-len"), "262144");
    assert.equal(flag(worker.argv, "--max-num-seqs"), "32");
    assert.equal(Number(flag(worker.argv, "--gpu-memory-utilization")), 0.8);
    assert.equal(worker.env.VLLM_USE_V2_MODEL_RUNNER, "1");
    assert.equal(worker.env.HCCL_OP_EXPANSION_MODE, "CCU_SCHED");
    assert.equal(worker.env.HCCL_BUFFSIZE, "800");
    assert.ok(!worker.argv.includes("--data-parallel-hybrid-lb"));
    assert.ok(!worker.argv.includes("--no-enable-prefix-caching"));
    assert.ok(!worker.argv.includes("--disable-flashinfer-autotune"));
    const draft = JSON.parse(flag(worker.argv, "--speculative-config"));
    assert.equal(draft.model, "Inferact/Kimi-K3-DSpark-Block5");
    assert.equal(draft.draft_tensor_parallel_size, 8);
    assert.equal(draft.enforce_eager, role === "prefill");
  }
  assert.ok(r.prefill.argv.includes("--enforce-eager"));
  assert.equal(flag(r.prefill.argv, "--data-parallel-size"), "1");
  assert.equal(flag(r.prefill.argv, "--distributed-executor-backend"), "mp");
  assert.equal(flag(r.prefill.argv, "--master-port"), "41000");
  assert.equal(r.prefill.env.VLLM_PP_LAYER_PARTITION, "24,24,24,21");
  assert.equal(flag(r.prefill.argv, "--max-num-batched-tokens"), "16384");
  assert.equal(flag(r.decode.argv, "--max-num-batched-tokens"), "768");
  assert.equal(flag(r.decode.argv, "--block-size"), "768");
  assert.equal(flag(r.decode.argv, "--cp-kv-cache-interleave-size"), "768");
});

// Mixed TP+DP serving has one public API on node 0. This differs from the
// external-LB PD decode pool, whose per-rank HTTP endpoints all serve requests.
const kimiK3 = read("models/moonshotai/Kimi-K3.yaml");
const mixed = (hardware, variant = "default", model = kimiK3) => resolveCommand(model, variant, "multi_node_tp_dp", hardware,
  [], strategies, taxonomy, [], 4);

for (const [hardware, variant, tp] of [["ascend_950dt", "ascend_mxfp4", 8], ["ascend_910c", "ascend_w4a8", 16]]) {
  test(`${hardware} mixed TP+DP exposes the API only on the head node`, () => {
    assert.deepEqual(kimiK3.strategy_overrides.multi_node_tp_dp.hardware_overrides[hardware].worker_args, ["--headless"],
      "the owning recipe must declare its worker behavior");
    const r = mixed(hardware, variant);
    assert.equal(r.deployType, "multi_node");
    assert.equal(r.workerArgvs.length, 3);
    assert.ok(!r.headArgv.includes("--headless"));
    for (const argv of [r.headArgv, ...r.workerArgvs]) {
      assert.equal(flag(argv, "--tensor-parallel-size"), String(tp));
      assert.equal(flag(argv, "--data-parallel-size"), "4");
      assert.equal(flag(argv, "--data-parallel-size-local"), "1");
    }
    for (const [index, argv] of r.workerArgvs.entries()) {
      assert.ok(argv.includes("--headless"), `node ${index + 1} must join without an API server`);
      assert.equal(flag(argv, "--data-parallel-start-rank"), String(index + 1));
      assert.equal(flag(argv, "--data-parallel-address"), flag(r.headArgv, "--data-parallel-address"));
    }
  });

  test(`${hardware} mixed worker behavior follows the source declaration`, () => {
    const model = structuredClone(kimiK3);
    delete model.strategy_overrides.multi_node_tp_dp.hardware_overrides[hardware].worker_args;
    const r = mixed(hardware, variant, model);
    assert.ok(!r.headArgv.includes("--headless"));
    for (const argv of r.workerArgvs) assert.ok(!argv.includes("--headless"), "hardware brand must not imply worker flags");
  });
}

for (const hardware of ["h200", "gb300", "mi355x"]) {
  test(`${hardware} mixed TP+DP keeps its existing worker API behavior`, () => {
    const r = mixed(hardware);
    assert.equal(r.workerArgvs.length, 3);
    for (const argv of [r.headArgv, ...r.workerArgvs]) {
      assert.ok(!argv.includes("--headless"));
      assert.equal(flag(argv, "--tensor-parallel-size"), String(taxonomy.hardware_profiles[hardware].gpu_count));
      assert.equal(flag(argv, "--data-parallel-size"), "4");
      assert.equal(flag(argv, "--data-parallel-size-local"), "1");
    }
    for (const [index, argv] of r.workerArgvs.entries()) {
      assert.equal(flag(argv, "--data-parallel-start-rank"), String(index + 1));
    }
  });
}

test("Ascend PD roles omit headless on every rank while mixed workers remain headless", () => {
  const model = structuredClone(kimiK3);
  // Even a declaration on a PD strategy is not a multi-node worker launch.
  model.strategy_overrides.pd_cluster.hardware_overrides.ascend_950dt = { worker_args: ["--headless"] };
  for (const rank of [0, 1, 2, 3]) {
    const r = pickAscend({ model, pools: { prefill: { nodes: 4, rank }, decode: { nodes: 4, rank } } });
    assert.ok(!r.prefill.argv.includes("--headless"), `prefill rank ${rank} must omit mixed-only worker behavior`);
    assert.doesNotMatch(r.prefill.templateCommand, /headless|HEADLESS_ARGS/);
    assert.doesNotMatch(r.decode.templateCommand, /headless|HEADLESS_ARGS/);
    assert.ok(!r.decode.argv.includes("--headless"), `decode rank ${rank} must serve its proxy endpoint`);
    assert.ok(!r.decode.argv.includes("--data-parallel-hybrid-lb"));
    assert.equal(flag(r.decode.argv, "--data-parallel-rank"), String(rank));
    assert.equal(flag(r.decode.argv, "--port"), "8002");
    for (const node of [1, 2, 3, 4]) assert.ok(r.router.argv.includes(`$DECODE_NODE_${node}`));
  }
});

// Execute generated shell with fake model/launcher executables. This catches
// quoting, positional-argument and rank bugs without starting an NPU service.
function withAscendLaunchFixture(run) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "recipes-ascend-launch-"));
  const bash = process.env.RECIPES_TEST_BASH || "bash";
  const env = { ...process.env, DECODE_DP_RPC_PORT: "41992" };
  const bin = path.join(dir, "mock-bin");
  fs.mkdirSync(bin);
  fs.writeFileSync(path.join(bin, "vllm"), "#!/usr/bin/env bash\nprintf '%s\\n' \"$HCCL_IF_IP\" \"$HCCL_SOCKET_IFNAME\" \"$GLOO_SOCKET_IFNAME\" \"$VLLM_HOST_IP\" > model-env.txt\nprintf '%s\\0' \"$@\"\n", { mode: 0o755 });
  fs.writeFileSync(path.join(bin, "python"), "#!/usr/bin/env bash\nprintf '__CALL__\\0%s\\0' \"$PWD\"\nprintf '%s\\0' \"$@\"\n", { mode: 0o755 });
  const shell = (command, args = []) => {
    const r = spawnSync(bash, ["-c", 'export PATH="$PWD/mock-bin:$PATH"\n' + command, "test", ...args],
      { cwd: dir, env, encoding: "utf8", timeout: 10000 });
    assert.ifError(r.error);
    assert.equal(r.status, 0, r.stderr || command);
    return r.stdout;
  };
  try {
    // Git Bash needs an executable bit even when Windows ignored writeFile's mode.
    shell("chmod +x mock-bin/vllm mock-bin/python");
    run({ dir, env, shell });
  } finally {
    assert.equal(path.dirname(dir), path.resolve(os.tmpdir()));
    assert.ok(path.basename(dir).startsWith("recipes-ascend-launch-"));
    fs.rmSync(dir, { recursive: true, force: true });
  }
}

test("A5 Scripts downloads only the launcher, with templates and proxy in their own steps", () => {
  const r = pickAscend({ model: kimiK3 });
  assert.match(r.ascendScripts.command, /examples\/external_online_dp\/launch_online_dp\.py/);
  assert.match(r.ascendScripts.command, /^curl --fail --location --output launch_online_dp\.py https:\/\/raw\.githubusercontent\.com\/vllm-project\/vllm-ascend\/v0\.23\.0\//);
  assert.doesNotMatch(r.ascendScripts.command, /docker|\bcp\b|mkdir|load_balance_proxy|run_dp_template|\n/);
  assert.equal(r.ascendScripts.containerCommand, undefined);
  assert.equal(r.ascendScripts.files.length, 1);
  assert.match(r.startServer.command, /launch_online_dp\.py/);
  assert.deepEqual(r.startServer.commands.map(({ role }) => role), ["prefill", "decode"]);
  assert.doesNotMatch(r.startServer.command, /ASCEND_PD_ROLE|NIC_NAME|local_ip|\bcase\b/);
  assert.match(r.router.command, /load_balance_proxy_server_example\.py/);
  assert.equal(r.kvConnector.selected, "mooncakev2");
  for (const role of ["prefill", "decode"]) {
    assert.match(r[role].templateCommand, /^cat > \.\/run_dp_template\.sh/);
    assert.match(r[role].templateCommand, /nic_name="<NIC_NAME>"\nlocal_ip="<LOCAL_IP>"/);
    assert.match(r[role].templateCommand, /unset ftp_proxy FTP_PROXY https_proxy HTTPS_PROXY http_proxy HTTP_PROXY LD_PRELOAD/);
    assert.match(r[role].command, /vllm serve/);
    assert.equal(kvCfg(r, role).kv_connector, "MooncakeConnectorV2");
  }
  withAscendLaunchFixture(({ shell }) => {
    for (const command of [r.ascendScripts.command, ...r.startServer.commands.map(({ command }) => command), r.router.command]) {
      shell('bash -n -c "$1"', [command]);
    }
  });
});

test("reusable A5 templates preserve PP ranks, external DP and enabled draft settings at runtime", () => {
  // One PD template covers a group head and followers, without mixed-only
  // worker flags on either role.
  const r = pickAscend({ model: kimiK3, features: ["spec_decoding"], pools: {
    prefill: { nodes: 8, rank: 7 }, decode: { nodes: 8, rank: 7 },
  } });
  withAscendLaunchFixture(({ dir, shell }) => {
    for (const role of ["prefill", "decode"]) {
      const kvPort = role === "prefill" ? "37001" : "37101";
      shell(r[role].templateCommand.replaceAll("<NIC_NAME>", "eth0").replaceAll("<LOCAL_IP>", "192.0.2.99")
        .replace(/^KV_PORT=.*$/m, `KV_PORT=${kvPort}`));
      const script = "run_dp_template.sh";
      assert.ok(fs.existsSync(path.join(dir, script)));
      shell('bash -n "$1"', [script]);
      for (const rank of [0, 3]) {
        const prefill = role === "prefill";
        const args = ["0,1,2,3,4,5,6,7", prefill ? "8171" : "8272", prefill ? "1" : "4",
          String(rank), prefill ? "192.0.2.15" : "192.0.2.25", prefill ? "41991" : "41992", "8"];
        const argv = shell('bash "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8"', [script, ...args]).split("\0").filter(Boolean);
        assert.equal(argv[0], "serve");
        assert.equal(flag(argv, "--tensor-parallel-size"), "8");
        assert.equal(flag(argv, "--data-parallel-size"), prefill ? "1" : "4");
        assert.equal(flag(argv, "--port"), prefill ? "8171" : "8272");
        assert.ok(argv.includes("--enable-expert-parallel"));
        const transfer = JSON.parse(flag(argv, "--kv-transfer-config"));
        assert.equal(transfer.kv_connector, "MooncakeConnectorV2");
        assert.equal(transfer.kv_port, kvPort, "the template's KV_PORT is editable");
        assert.equal(transfer.kv_connector_extra_config.prefill.tp_size, 8);
        assert.equal(transfer.kv_connector_extra_config.decode.tp_size, 8);
        const draft = JSON.parse(flag(argv, "--speculative-config"));
        assert.equal(draft.model, "Inferact/Kimi-K3-DSpark-Block5");
        assert.equal(draft.draft_tensor_parallel_size, 8);
        assert.equal(draft.enforce_eager, prefill);
        assert.deepEqual(fs.readFileSync(path.join(dir, "model-env.txt"), "utf8").trim().split(/\r?\n/),
          ["192.0.2.99", "eth0", "eth0", "192.0.2.99"], "template uses manually replaced node IP and interface");
        if (prefill) {
          assert.equal(flag(argv, "--pipeline-parallel-size"), "4");
          assert.equal(flag(argv, "--nnodes"), "4");
          assert.equal(flag(argv, "--node-rank"), String(rank));
          assert.equal(flag(argv, "--master-addr"), "192.0.2.15");
          assert.equal(flag(argv, "--master-port"), "41991");
          assert.ok(!argv.includes("--headless"));
        } else {
          assert.equal(flag(argv, "--data-parallel-rank"), String(rank));
          assert.equal(flag(argv, "--data-parallel-address"), "192.0.2.25");
          assert.equal(flag(argv, "--data-parallel-rpc-port"), "41992");
          assert.ok(!argv.includes("--headless"));
          assert.ok(!argv.includes("--data-parallel-hybrid-lb"));
        }
      }
    }
  });
});

test("A5 server launch uses group-local ranks and distinct masters for replicated groups", () => {
  const r = pickAscend({ model: kimiK3, pools: {
    prefill: { nodes: 12, rank: 11 }, decode: { nodes: 8, rank: 7 },
  } });
  withAscendLaunchFixture(({ dir, env, shell }) => {
    for (const role of ["PREFILL", "DECODE"]) {
      for (let node = 1; node <= 12; node++) env[role + "_NODE_" + node] = (role === "PREFILL" ? "192.0.2." : "198.51.100.") + node;
    }
    const calls = [];
    for (const role of ["prefill", "decode"]) {
      const selected = shell(r.startServer.commands.find((entry) => entry.role === role).command).split("__CALL__\0").filter(Boolean)
        .map((call) => call.split("\0").filter(Boolean));
      assert.equal(selected.length, 1, "each independent command starts one role");
      calls.push(...selected);
    }
    assert.equal(calls.length, 2);
    for (const [index, call] of calls.entries()) {
      const prefill = index === 0;
      assert.equal(path.basename(call[0]), path.basename(dir), "launcher and template share the current directory");
      assert.equal(path.basename(call[1]), "launch_online_dp.py");
      assert.equal(flag(call, "--dp-size"), prefill ? "1" : "4");
      assert.equal(flag(call, "--tp-size"), "8");
      assert.equal(flag(call, "--dp-size-local"), "1");
      assert.equal(flag(call, "--dp-rank-start"), "3");
      assert.equal(flag(call, "--dp-address"), prefill ? "192.0.2.9" : "198.51.100.5");
      assert.equal(flag(call, "--dp-rpc-port"), prefill ? "41000" : "41992");
    }
  });
});

test("the five-step payload stays within A5 disaggregated serving", () => {
  const results = [
    mixed("ascend_950dt", "ascend_mxfp4"),
    resolveCommand(kimiK3, "ascend_w4a8", "pd_cluster", "ascend_910c", [], strategies, taxonomy, [], 1, pools),
    ...["h200", "gb300", "mi355x"].map((hw) =>
      resolveCommand(kimiK3, "default", "pd_cluster", hw, [], strategies, taxonomy, [], 1, pools)),
  ];
  for (const r of results) {
    assert.ok(!r.ascendScripts);
    assert.ok(!r.startServer);
    assert.ok(!r.prefill?.templateCommand);
    assert.ok(!r.decode?.templateCommand);
  }
});
