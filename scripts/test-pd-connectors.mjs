import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { test } from "node:test";
import yaml from "js-yaml";
import { buildDockerArgv, buildDockerRun, computeDockerMeta, pdConnectorKey, resolveCommand } from "../src/lib/command-synthesis.js";
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

test("the Ascend router rewrites both pools to vllm-ascend external DP + MooncakeHybridConnector", () => {
  // Read the repo's own copy, not the assembled tree: once a recipe is synced
  // to the vllm-project/vllm source the local copy defers to it, so the A3
  // pool overrides under test travel with this repo's PR first.
  const ds = yaml.load(fs.readFileSync(path.resolve("models/deepseek-ai/DeepSeek-V4-Flash.yaml"), "utf8"));
  const r = resolveCommand(ds, "ascend_w8a8", "pd_cluster", "ascend_910c", [], strategies, taxonomy, [], 1, pools, {}, null, null, undefined, "ascend");
  assert.equal(r.orchestrator, "ascend");
  // Prefill pool (A3 Mooncake PD layout: dp4 x tp4): the external-DP
  // rendezvous takes rank + rpc-port, and the HTTP port moves to 7100.
  assert.equal(flag(r.prefill.argv, "--tensor-parallel-size"), "4");
  assert.equal(flag(r.prefill.argv, "--data-parallel-size"), "4");
  assert.equal(flag(r.prefill.argv, "--data-parallel-rank"), "0");
  assert.equal(flag(r.prefill.argv, "--data-parallel-rpc-port"), "12321");
  assert.equal(flag(r.prefill.argv, "--port"), "7100");
  assert.deepEqual(JSON.parse(flag(r.prefill.argv, "--kv-transfer-config")), {
    kv_connector: "MooncakeHybridConnector",
    kv_role: "kv_producer",
    kv_port: "36000",
    engine_id: "0",
    kv_connector_extra_config: { prefill: { dp_size: 4, tp_size: 4 }, decode: { dp_size: 16, tp_size: 1 } },
  });
  // Decode pool (dp16 x tp1).
  assert.equal(flag(r.decode.argv, "--tensor-parallel-size"), "1");
  assert.equal(flag(r.decode.argv, "--data-parallel-size"), "16");
  const dcfg = JSON.parse(flag(r.decode.argv, "--kv-transfer-config"));
  assert.equal(dcfg.kv_role, "kv_consumer");
  assert.equal(dcfg.kv_port, "36100");
  assert.equal(dcfg.engine_id, "1");
  // A3 pool tuning (role hardware_overrides) overrides the GB200 vllm_args
  // and the variant's single-node values via last-wins dedupe.
  assert.equal(flag(r.prefill.argv, "--max-num-seqs"), "16");
  assert.equal(flag(r.prefill.argv, "--max-num-batched-tokens"), "8192");
  assert.equal(flag(r.decode.argv, "--max-num-seqs"), "60");
  assert.equal(flag(r.decode.argv, "--max-num-batched-tokens"), "120");
  assert.ok(JSON.parse(flag(r.prefill.argv, "--additional-config")).enable_dsa_cp);
  assert.ok(JSON.parse(flag(r.decode.argv, "--additional-config")).recompute_scheduler_enable);
  // NPU visibility replaces CUDA, NVIDIA NCCL knobs are dropped (HCCL has its
  // own), decode runs HCCL_BUFFSIZE=1024 while prefill keeps the variant's 2560.
  for (const role of ["prefill", "decode"]) {
    assert.equal(r[role].env.ASCEND_RT_VISIBLE_DEVICES, "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15");
    assert.equal(r[role].env.CUDA_VISIBLE_DEVICES, undefined);
    assert.equal(r[role].env.VLLM_USE_NCCL_SYMM_MEM, undefined);
    for (const k of Object.keys(r[role].env)) assert.ok(!k.startsWith("NCCL_"), `NCCL var leaked: ${k}`);
  }
  assert.equal(r.prefill.env.HCCL_BUFFSIZE, "2560");
  assert.equal(r.decode.env.HCCL_BUFFSIZE, "1024");
  // Scripts / Start_Server / proxy payloads carry the image-matching tag.
  assert.ok(r.ascendScripts.launcher.includes("/blob/v0.23.0/"), "launcher tag must match the pulled image");
  assert.ok(r.ascendScripts.proxy.includes("/blob/v0.23.0/"), "proxy tag must match the pulled image");
  assert.ok(r.startServer.command.includes("launch_online_dp.py"));
  assert.ok(r.router.command.includes("load_balance_proxy_server_example.py"));
  assert.equal(r.router.link, r.ascendScripts.proxy);
  // A connector pick (nixl / mooncake / moriio) must not leak into the Ascend
  // router: the transfer config and env belong to the strategy alone.
  const picked = resolveCommand(ds, "ascend_w8a8", "pd_cluster", "ascend_910c", [], strategies, taxonomy, [], 1,
    pools, {}, null, null, undefined, "ascend", "http", null, undefined, undefined, "mooncake");
  assert.equal(picked.kvConnector, null);
  assert.equal(picked.prefill.env.VLLM_MOONCAKE_BOOTSTRAP_PORT, undefined);
  assert.equal(picked.prefill.env.VLLM_NIXL_SIDE_CHANNEL_PORT, undefined);
  assert.equal(picked.prefill.argv.filter((t) => t === "--kv-transfer-config").length, 1);
  assert.equal(JSON.parse(flag(picked.prefill.argv, "--kv-transfer-config")).kv_connector, "MooncakeHybridConnector");
});
