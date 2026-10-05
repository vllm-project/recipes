import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { test } from "node:test";
import yaml from "js-yaml";
import { resolveCommand } from "../src/lib/command-synthesis.js";

const read = (file) => yaml.load(fs.readFileSync(new URL(`../${file}`, import.meta.url), "utf8"));
const recipe = read("models/XiaomiMiMo/MiMo-V2.6-Flash-RL.yaml");
const taxonomy = read("taxonomy.yaml");
const strategies = Object.fromEntries(["strategies", "kv_store"].flatMap((dir) =>
  fs.readdirSync(new URL(`../${dir}`, import.meta.url))
    .filter((file) => file.endsWith(".yaml"))
    .map((file) => { const spec = read(`${dir}/${file}`); return [spec.name, spec]; })));
const resolve = ({ router = "dynamo", offload = null, pools = {}, specs = strategies } = {}) =>
  resolveCommand(recipe, "default", "pd_cluster", "gb300", ["tool_calling", "reasoning"],
    specs, taxonomy, [], 1, pools, {}, offload, null, undefined, router);
const flag = (argv, key) => {
  const index = argv.indexOf(key);
  return index === -1 ? undefined : argv[index + 1];
};

test("frontend command delivers agentic flags intact after shell expansion", () => {
  const result = resolve();
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "recipes-dynamo-"));
  try {
    fs.writeFileSync(path.join(dir, "python3"), '#!/usr/bin/env node\nconsole.log(JSON.stringify(process.argv.slice(2)));\n', { mode: 0o755 });
    const argv = JSON.parse(execFileSync("bash", ["-c", result.router.command], {
      env: { ...process.env, PATH: `${dir}${path.delimiter}${process.env.PATH || ""}`, ROUTER_PORT: "31000" }, encoding: "utf8",
    }));
    assert.deepEqual(argv.slice(0, 2), ["-m", "dynamo.frontend"]);
    assert.equal(flag(argv, "--http-port"), "31000");
    assert.equal(flag(argv, "--router-mode"), "kv");
    assert.equal(flag(argv, "--router-session-affinity-ttl-secs"), "14400");
    assert.ok(argv.includes("--no-router-kv-events"));
    for (const key of ["--active-decode-blocks-threshold", "--active-prefill-tokens-threshold", "--active-prefill-tokens-threshold-frac"]) {
      assert.ok(!argv.includes(key));
    }
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test("frontend cache environment stays separate from TP and DEP workers", () => {
  for (const parallelism of ["tp", "dep"]) {
    const result = resolve({ pools: { prefill: { nodes: 2, parallelism, rank: 1 }, decode: { nodes: 2, parallelism, rank: 1 } } });
    assert.equal(result.router.env.DYN_TOKENIZER, "fastokens");
    assert.equal(result.router.env.DYN_TOKENIZER_CACHE_BYTES, "8000000000");
    for (const key of ["DYN_ROUTER_TEMPERATURE", "DYN_ROUTER_QUEUE_THRESHOLD", "DYN_TOKENIZER_CACHE", "DYN_LOGGING_JSONL", "OTEL_EXPORT_ENABLED"]) {
      assert.equal(result.router.env[key], undefined);
    }
    for (const role of ["prefill", "decode"]) {
      assert.equal(result[role].env.ETCD_ENDPOINTS, result.router.env.ETCD_ENDPOINTS);
      assert.equal(result[role].env.DYN_REQUEST_PLANE, "tcp");
      assert.equal(result[role].env.DYN_TOKENIZER_CACHE_BYTES, undefined);
      assert.equal(result[role].env.ETCD_LEASE_TTL, undefined);
      assert.ok(!result[role].argv.includes("--port"));
      assert.equal(flag(result[role].argv, "--disaggregation-mode"), role);
      assert.equal(JSON.parse(flag(result[role].argv, "--kv-transfer-config")).kv_role, "kv_both");
    }
    assert.ok(!result.prefill.argv.includes("--router-mode"));
    assert.equal(flag(result.decode.argv, "--router-mode"), "least-loaded");
    assert.ok(!result.decode.argv.includes("--stream-interval"));
  }
});

test("agentic routing composes with Mooncake and native PD retains its settings", () => {
  const result = resolve({ offload: "kv_store_distributed_mooncake" });
  for (const role of ["prefill", "decode"]) {
    const config = JSON.parse(flag(result[role].argv, "--kv-transfer-config"));
    assert.equal(config.kv_connector, "MultiConnector");
    assert.ok(config.kv_connector_extra_config.connectors.some((c) => c.kv_connector === "MooncakeStoreConnector"));
  }
  assert.equal(flag(result.decode.argv, "--router-mode"), "least-loaded");
  const native = resolve({ router: "vllm" });
  assert.equal(native.orchestrator, "vllm-router");
  assert.ok(!native.decode.argv.includes("--router-mode"));
  assert.ok(!native.decode.argv.includes("--stream-interval"));
  assert.equal(flag(native.decode.argv, "--port"), "8002");
});
