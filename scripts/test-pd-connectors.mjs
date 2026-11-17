import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { test } from "node:test";
import yaml from "js-yaml";
import { buildDockerArgv, buildDockerRun, computeDockerMeta, pdConnectorKey, pdConnectorUnavailableReason, resolveCommand } from "../src/lib/command-synthesis.js";
import { applyPresetArgOps, validatePresets } from "../src/lib/recipe-presets.js";
import { renderCommand } from "../src/lib/recipe-render.js";
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
  for (const hw of ["ascend_950dt", "ascend_910c"]) {
    const r = pick({ hw });
    assert.equal(r.pdRouter, "vllm", hw);
    assert.equal(r.kvConnector.selected, "nixl", hw);
    assert.equal(pdConnectorUnavailableReason(pd.connectors[r.kvConnector.selected],
      taxonomy.hardware_profiles[hw], r.pdRouter), null, hw);
  }
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

test("a connector edit is skipped and recorded when a collector is passed", () => {
  const nixl = '{"kv_connector":"NixlConnector","kv_role":"kv_producer"}';
  const edit = { "--kv-transfer-config": { MooncakeStoreConnector: { x: 1 }, NixlConnector: { kv_load_failure_policy: "recompute" } } };
  const missing = new Set();
  const [, out] = applyPresetArgOps(["--kv-transfer-config", nixl], { json_args: edit }, undefined, missing);
  // The connector that is there still takes its edit.
  assert.deepEqual(JSON.parse(out), { kv_connector: "NixlConnector", kv_role: "kv_producer", kv_load_failure_policy: "recompute" });
  assert.deepEqual([...missing], ["json_args: --kv-transfer-config has no MooncakeStoreConnector"]);
  // No collector: still an error.
  assert.throws(() => applyPresetArgOps(["--kv-transfer-config", nixl], { json_args: edit }), /has no MooncakeStoreConnector/);
  // An absent flag isn't added just to hold skipped connector edits.
  const absent = new Set();
  assert.deepEqual(applyPresetArgOps(["--port", "8000"], { json_args: { "--kv-transfer-config": { MooncakeStoreConnector: { x: 1 } } } }, undefined, absent), ["--port", "8000"]);
  assert.equal(absent.size, 1);
});

test("a preset's Mooncake-store edit waits for its KV Offload pick instead of breaking the command", () => {
  // Kimi-K3.1-Pro's GB300 PD lanes: select.kv_offload turns the store on, and
  // json_args tunes its MooncakeStoreConnector. The builder renders the preset
  // before it applies `select`, and the reader may switch KV Offload off.
  const lane = { ...recipe, presets: { lane: {
    label: "lane", hardware: "gb300", strategy: "pd_cluster",
    select: { kv_offload: "kv_store_distributed_mooncake" },
    json_args: { "--kv-transfer-config": { MooncakeStoreConnector: { kv_load_failure_policy: "recompute" } } },
  } } };
  const run = (kv) => resolveCommand(lane, "default", "pd_cluster", "gb300", [], strategies, taxonomy, [], 1, pools, {}, kv,
    null, undefined, null, "http", null, "lane", undefined, null);

  const off = run(null);
  assert.equal(off.preset.selected, "lane");
  assert.deepEqual(off.preset.select, { kv_offload: "kv_store_distributed_mooncake" });
  assert.deepEqual(off.presetWarnings, ["json_args: --kv-transfer-config has no MooncakeStoreConnector"]);
  for (const role of ["prefill", "decode"]) assert.equal(kvCfg(off, role).kv_connector, "NixlConnector");

  const on = run("kv_store_distributed_mooncake");
  assert.equal(on.presetWarnings, undefined);
  for (const role of ["prefill", "decode"]) {
    const store = kvCfg(on, role).kv_connector_extra_config.connectors.find((c) => c.kv_connector === "MooncakeStoreConnector");
    assert.equal(store.kv_load_failure_policy, "recompute", role);
  }

  // The build renders presets with their own `select`; a lane whose `select`
  // doesn't add the connector it edits still fails there.
  assert.doesNotThrow(() => renderCommand(lane, "default", "pd_cluster", "gb300", 1, [], strategies, taxonomy, pools, null, "lane", undefined, true));
  const noSelect = { ...lane, presets: { lane: { ...lane.presets.lane, select: undefined } } };
  assert.throws(() => renderCommand(noSelect, "default", "pd_cluster", "gb300", 1, [], strategies, taxonomy, pools, null, "lane", undefined, true),
    /Invalid preset lane .*has no MooncakeStoreConnector/);
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

const ascend = (r, rank = 0, connector = "mooncake_ascend", strats = strategies, features = [], pools = ascendPdPools(rank), kv = null) =>
  resolveCommand(r, "default", "pd_cluster", "ascend_950dt", features, strats, taxonomy, [], 1,
    pools, {}, kv, null, "python", "ascend", "http", null, "ascend_pd", undefined, connector);

test("Ascend catalog defaults survive absent overrides and reject invalid final configs", () => {
  const defaults = ascend(ascendPdRecipe());
  for (const role of ["prefill", "decode"]) assert.equal(kvCfg(defaults, role).kv_connector, "MooncakeConnectorV2");
  for (const value of ["not-json", "null", "[]", "42", "{}", '{"kv_connector":""}', '{"kv_connector":3}']) {
    const fixture = ascendPdRecipe();
    fixture.presets.ascend_pd.roles.prefill.args = ["--kv-transfer-config", value];
    assert.throws(() => ascend(fixture),
      (error) => error instanceof RangeError && /^PD prefill:.*--kv-transfer-config/.test(error.message), value);
  }
  for (const kv of [null, "simple"]) {
    const fixture = ascendPdRecipe();
    fixture.presets.ascend_pd.roles.prefill.json_args = { "--kv-transfer-config": { MooncakeConnectorV2: { kv_connector: null } } };
    assert.throws(() => ascend(fixture, 0, "mooncake_ascend", strategies, [], ascendPdPools(), kv), RangeError);
  }
});

test("Ascend catalog configs use ordinary named preset edits for implementations and offload", () => {
  for (const name of ["MooncakeConnectorV1", "MooncakeConnectorV2", "MooncakeHybridConnector", "vendor.custom_v2"]) {
    for (const kv of [null, "simple"]) {
      const fixture = ascendPdRecipe();
      const preset = fixture.presets.ascend_pd;
      preset.json_args = { "--kv-transfer-config": { MooncakeConnectorV2: { kv_connector_extra_config: { order: "all" } } } };
      for (const role of ["prefill", "decode"]) {
        preset.roles[role].json_args["--kv-transfer-config"] = { MooncakeConnectorV2: {
          kv_connector: name, kv_port: null, kv_connector_extra_config: { order: "role", prefill: null },
        } };
      }
      if (kv) preset.roles.prefill.json_args["--kv-transfer-config"].SimpleCPUOffloadConnector = {
        kv_connector_extra_config: { cpu_bytes_to_use_per_rank: null, cpu_bytes_to_use: 9 },
      };
      const rendered = ascend(fixture, 0, "mooncake_ascend", strategies, [], ascendPdPools(), kv);
      assert.equal(rendered.kvConnector.selected, "mooncake_ascend");
      assert.equal(rendered.kvConnector.unavailableReason, null);
      for (const role of ["prefill", "decode"]) {
        const cfg = kvCfg(rendered, role);
        const transfer = kv && role === "prefill" ? cfg.kv_connector_extra_config.connectors[0] : cfg;
        assert.equal(transfer.kv_connector, name);
        assert.equal(transfer.kv_port, undefined);
        assert.deepEqual(transfer.kv_connector_extra_config, { order: "role", decode: { dp_size: 8, tp_size: 4, pp_size: 1 } });
        assert.equal(rendered[role].argv.filter((arg) => arg === "--kv-transfer-config").length, 1);
      }
      if (kv) {
        const cfg = kvCfg(rendered, "prefill");
        assert.equal(cfg.kv_connector, "MultiConnector");
        const cpu = cfg.kv_connector_extra_config.connectors[1];
        assert.equal(cpu.kv_connector, "SimpleCPUOffloadConnector");
        assert.deepEqual(cpu.kv_connector_extra_config, { lazy_offload: false, cpu_bytes_to_use: 9 });
      }
    }
  }
});

test("Ascend PD requires each selection's declaration in the active hardware and variant preset", () => {
  const cases = [
    { name: "no preset", edit: (r) => { r.presets = {}; } },
    { name: "no select", edit: (r) => { delete r.presets.ascend_pd.select; } },
    { name: "no router declaration", edit: (r) => { delete r.presets.ascend_pd.select.pd_router; } },
    { name: "no connector declaration", router: "ascend", edit: (r) => { delete r.presets.ascend_pd.select.pd_connector; } },
    { name: "Baseline", preset: "baseline" },
    { name: "another Huawei profile", hw: "ascend_910c" },
    { name: "declarations on non-Huawei hardware", hw: "h200", edit: (r) => { r.presets.ascend_pd.hardware = "h200"; } },
    { name: "another variant", variant: "other", edit: (r) => {
      r.variants.other = { ...r.variants.default };
      r.presets.ascend_pd.variants = ["default"];
    } },
  ];
  for (const { name, edit, hw = "ascend_950dt", variant = "default", preset = "ascend_pd", router = "vllm" } of cases) {
    const fixture = ascendPdRecipe();
    edit?.(fixture);
    const r = resolveCommand(fixture, variant, "pd_cluster", hw, [], strategies, taxonomy, [], 1,
      ascendPdPools(), {}, null, null, "python", "ascend", "http", null, preset, undefined, "mooncake_ascend");
    assert.equal(r.pdRouter, router, name);
    assert.equal(r.kvConnector.selected, "nixl", name);
    assert.equal(!!r.routerUnavailableReason, router !== "ascend", name);
    assert.ok(r.kvConnector.unavailableReason, name);
  }
});

test("Ascend PD uses the current point's inherited declarations, not another point's", () => {
  const fixture = ascendPdRecipe();
  delete fixture.presets.ascend_pd.select.pd_connector;
  fixture.presets.ascend_pd.points = {
    qualified: { default: true, select: { pd_connector: "mooncake_ascend" } },
    unconfigured: {},
    overridden: { select: { pd_router: "vllm", pd_connector: "mooncake_ascend" } },
  };
  for (const [point, router, connector] of [[undefined, "ascend", "mooncake_ascend"], ["qualified", "ascend", "mooncake_ascend"],
    ["unconfigured", "ascend", "nixl"], ["overridden", "vllm", "nixl"]]) {
    const r = resolveCommand(fixture, "default", "pd_cluster", "ascend_950dt", [], strategies, taxonomy, [], 1,
      ascendPdPools(), {}, null, null, "python", "ascend", "http", null, "ascend_pd", point, "mooncake_ascend");
    assert.equal(r.pdRouter, router, point);
    assert.equal(r.kvConnector.selected, connector, point);
  }
});

test("TP/TEP derives its width from the whole pool as PD grows from four to eight nodes", () => {
  const fixture = ascendPdRecipe();
  fixture.compatible_strategies.push("single_node_tep");
  // A TP hint retained from DEP must not pin a TP/TEP pool's width.
  fixture.presets.ascend_pd.roles.prefill.tp = 8;
  for (const parallelism of ["tp", "tep"]) {
    for (const nodes of [4, 8]) {
      const pools = { prefill: { nodes, rank: nodes - 1, parallelism }, decode: { nodes, rank: nodes - 1, parallelism: "dep" } };
      const r = ascend(fixture, nodes - 1, "mooncake_ascend", strategies, [], pools);
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
  for (const [nodes, tp] of [[1, 8], [1, 4], [1, 2], [4, 8], [4, 4], [4, 2]]) {
    const fixture = ascendPdRecipe();
    fixture.presets.ascend_pd.roles.decode.tp = tp;
    const local = 8 / tp;
    for (let rank = 0; rank < nodes; rank++) {
      const pools = ascendPdPools(rank);
      pools.decode.nodes = nodes;
      const r = ascend(fixture, rank, "mooncake_ascend", strategies, [], pools);
      const argv = r.decode.argv;
      assert.equal(flag(argv, "--tensor-parallel-size"), String(tp));
      assert.equal(flag(argv, "--data-parallel-size-local"), String(local));
      assert.equal(flag(argv, "--data-parallel-size"), String(nodes * local));
      assert.ok(!argv.includes("--data-parallel-hybrid-lb"));
      assert.equal(argv.includes("--data-parallel-multi-port-external-lb"), local > 1);
      assert.equal(argv.includes("--data-parallel-external-lb"), local === 1 && nodes > 1);
      const rankFlag = local > 1 ? "--data-parallel-start-rank" : "--data-parallel-rank";
      if (nodes * local > 1) assert.equal(flag(argv, rankFlag), String(rank * local));
      else {
        assert.ok(!argv.includes("--data-parallel-rank") && !argv.includes("--data-parallel-start-rank"));
        assert.ok(argv.includes("--enable-expert-parallel"));
        const inconsistent = structuredClone(fixture);
        inconsistent.presets.ascend_pd.roles.decode.args = ["--data-parallel-external-lb"];
        assert.throws(() => ascend(inconsistent, rank, "mooncake_ascend", strategies, [], pools), RangeError);
      }
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
      for (let node = 1; node <= nodes; node++) {
        assert.equal(r.router.argv.filter((arg) => arg === `$DECODE_NODE_${node}`).length, local);
      }
      const ports = r.router.argv.slice(r.router.argv.indexOf("--decoder-ports") + 1);
      assert.deepEqual(ports, Array.from({ length: nodes }, () => Array.from({ length: local }, (_, i) => String(8002 + i))).flat());
      assert.equal(kvCfg(r, "decode").kv_connector_extra_config.decode.dp_size, nodes * local);
      assert.doesNotMatch(r.router.command, /launch_online_dp|run_dp_template/);
    }
  }
});

test("PD topology placeholders, role edits and connector selection share the final pool layout", () => {
  const fixture = ascendPdRecipe();
  const r = ascend(fixture, 0, "mooncake_ascend", strategies, ["spec_decoding"]);
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
  const stored = ascend(fixture, 0, "mooncake_ascend", strategies, [], ascendPdPools(), "simple");
  assert.deepEqual(kvCfg(stored, "prefill").kv_connector_extra_config.connectors.map((c) => c.kv_connector),
    ["MooncakeConnectorV2", "SimpleCPUOffloadConnector"]);
  assert.equal(kvCfg(stored, "decode").kv_connector, "MooncakeConnectorV2");

  const custom = structuredClone(strategies);
  custom.pd_cluster.connectors.other = {
    brands: ["Huawei"], routers: ["ascend"],
    prefill: { config: { kv_connector: "OtherConnector", kv_role: "kv_producer", kv_connector_extra_config: { tp: "{prefill_tp_size}" } } },
    decode: { config: { kv_connector: "OtherConnector", kv_role: "kv_consumer" } },
  };
  const catalogFixture = ascendPdRecipe();
  for (const role of ["prefill", "decode"]) delete catalogFixture.presets.ascend_pd.roles[role].json_args["--kv-transfer-config"];
  const other = ascend(catalogFixture, 0, "other", custom);
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
