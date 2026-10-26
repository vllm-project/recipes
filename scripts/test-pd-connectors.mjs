import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { test } from "node:test";
import yaml from "js-yaml";
import { pdConnectorKey, resolveCommand } from "../src/lib/command-synthesis.js";
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
