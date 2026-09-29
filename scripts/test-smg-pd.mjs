import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { test } from "node:test";
import yaml from "js-yaml";
import { resolveCommand, buildDockerRun, computeDockerMeta } from "../src/lib/command-synthesis.js";
import { substitute } from "../src/lib/cluster-endpoints.js";

const read = (file) => yaml.load(fs.readFileSync(new URL(`../${file}`, import.meta.url), "utf8"));
const recipe = read("models/XiaomiMiMo/MiMo-V2.6-Flash-RL.yaml");
const taxonomy = read("taxonomy.yaml");
const strategies = Object.fromEntries(fs.readdirSync(new URL("../strategies", import.meta.url))
  .filter((file) => file.endsWith(".yaml"))
  .map((file) => { const strategy = read(`strategies/${file}`); return [strategy.name, strategy]; }));
const flag = (argv, key) => argv[argv.indexOf(key) + 1];
const resolve = ({ router = "smg", hardware = "gb300", pools = { prefill: { nodes: 1 }, decode: { nodes: 1 } }, offload = null, model = recipe, strategy = "pd_cluster" } = {}) =>
  resolveCommand(model, "default", strategy, hardware, ["tool_calling", "reasoning"], strategies, taxonomy, [], 1, pools, {}, offload, null, undefined, router);

test("SMG preserves MiMo TP4 HTTP workers and producer/consumer connectors", () => {
  const result = resolve();
  assert.equal(result.orchestrator, "smg");
  for (const [role, port, kvRole] of [["prefill", "8001", "kv_producer"], ["decode", "8002", "kv_consumer"]]) {
    const { argv } = result[role];
    assert.deepEqual(argv.slice(0, 3), ["vllm", "serve", "XiaomiMiMo/MiMo-V2.6-Flash-RL"]);
    assert.equal(flag(argv, "--tensor-parallel-size"), "4");
    assert.equal(flag(argv, "--port"), port);
    assert.equal(flag(argv, "--tool-call-parser"), "mimo");
    assert.equal(flag(argv, "--reasoning-parser"), "mimo");
    assert.equal(JSON.parse(flag(argv, "--kv-transfer-config")).kv_role, kvRole);
    assert.ok(!argv.includes("--disaggregation-mode"));
    assert.ok(!result[role].env.ETCD_ENDPOINTS);
  }
  assert.ok(!result.infra);
});

test("registration sends connector metadata and substitutes endpoints into valid JSON", () => {
  const result = resolve();
  assert.ok(result.registration, "SMG needs explicit connector-aware registration");
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "recipes-smg-"));
  try {
    // Capture the actual curl boundary after shell expansion, including heredoc stdin.
    fs.writeFileSync(path.join(dir, "curl"), `#!/usr/bin/env node
let body = "";
process.stdin.on("data", (chunk) => body += chunk);
process.stdin.on("end", () => console.log(JSON.stringify({args: process.argv.slice(2), body: body.trim() ? JSON.parse(body) : null})));
`, { mode: 0o755 });
    const script = substitute(result.registration.command, {
      ROUTER_HOST: "192.0.2.10", ROUTER_PORT: "30000",
      PREFILL_NODE_1: "192.0.2.11", DECODE_NODE_1: "192.0.2.12",
    });
    execFileSync("bash", ["-n"], { input: script });
    const calls = execFileSync("bash", ["-c", script], { env: { ...process.env, PATH: `${dir}:${process.env.PATH}` }, encoding: "utf8" })
      .trim().split("\n").map(JSON.parse);
    const posts = calls.filter((call) => call.body);
    assert.deepEqual(posts.map((call) => call.body), [
      { url: "http://192.0.2.11:8001", worker_type: "prefill", runtime_type: "vllm", kv_connector: "NixlConnector", kv_role: "kv_producer" },
      { url: "http://192.0.2.12:8002", worker_type: "decode", runtime_type: "vllm", kv_connector: "NixlConnector", kv_role: "kv_consumer" },
    ]);
    assert.ok(posts.every((call) => call.args.includes("http://192.0.2.10:30000/workers")));
    assert.ok(calls.at(-1).args.includes("http://192.0.2.10:30000/readiness"));
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test("multi-node TP/TEP registers only pool heads, not headless followers", () => {
  const result = resolve({ pools: { prefill: { nodes: 2, parallelism: "tep", rank: 1 }, decode: { nodes: 2, parallelism: "tp", rank: 1 } } });
  assert.equal(result.orchestrator, "smg");
  assert.ok(result.prefill.argv.includes("--headless"));
  assert.ok(result.decode.argv.includes("--headless"));
  assert.match(result.registration.command, /\$PREFILL_NODE_1:8001/);
  assert.match(result.registration.command, /\$DECODE_NODE_1:8002/);
  assert.doesNotMatch(result.registration.command, /NODE_2/);
});

test("SMG startup is separate from registration and workers retain Docker support", () => {
  const result = resolve();
  assert.match(result.router.command, /^smg launch/);
  assert.match(result.router.command, /--pd-disaggregation/);
  assert.doesNotMatch(result.router.command, /--prefill |--decode |etcd|dynamo/);
  const meta = computeDockerMeta(recipe, recipe.variants.default, taxonomy.hardware_profiles.gb300, "gb300");
  const command = buildDockerRun({ command: result.prefill.command, env: result.prefill.env, image: meta.image, gpuFlags: meta.gpuFlags });
  assert.match(command, /vllm\/vllm-openai:mimo-v26/);
  assert.match(command, /--port 8001/);
});

test("SMG remains available on AMD without inheriting Dynamo's NVIDIA gate", () => {
  const result = resolve({ hardware: "mi355x" });
  assert.equal(result.orchestrator, "smg");
  assert.equal(result.prefill.env.VLLM_ROCM_USE_AITER, "1");
});

test("SMG registration follows worker ports and rejects incompatible connector overrides", () => {
  const model = structuredClone(recipe);
  model.strategy_overrides.pd_cluster = {
    prefill: { vllm_args: ["--port", "8101"] },
    decode: { vllm_args: ["--port", "8102"] },
  };
  const result = resolve({ model });
  assert.equal(flag(result.prefill.argv, "--port"), "8101");
  assert.match(result.registration.command, /NODE_1:8101/);
  assert.match(result.registration.command, /NODE_1:8102/);
  model.strategy_overrides.pd_cluster.decode.vllm_args.push("--kv-transfer-config", '{"kv_connector":"MultiConnector","kv_role":"kv_both"}');
  const unsupported = resolve({ model });
  assert.equal(unsupported.orchestrator, "vllm-router");
  assert.match(unsupported.routerUnavailableReason, /NixlConnector/);
  assert.equal(unsupported.registration, undefined);
});

test("unsupported SMG combinations fall back with an explicit reason", () => {
  for (const options of [
    { pools: { prefill: { nodes: 2, parallelism: "dep" }, decode: { nodes: 1 } } },
    { offload: "kv_store_distributed_mooncake" },
    { model: read("models/moonshotai/Kimi-K3.yaml") },
  ]) {
    const result = resolve(options);
    assert.equal(result.orchestrator, "vllm-router");
    assert.ok(result.routerUnavailableReason);
    assert.equal(result.registration, undefined);
  }
});

test("native and Dynamo selections retain their existing behavior", () => {
  const native = resolve({ router: "vllm" });
  assert.equal(native.orchestrator, "vllm-router");
  assert.match(native.router.command, /--prefill http:\/\/\$PREFILL_NODE_1:8001/);
  assert.equal(native.registration, undefined);
  const dynamo = resolve({ router: "dynamo" });
  assert.equal(dynamo.orchestrator, "dynamo");
  assert.equal(dynamo.prefill.argv[2], "dynamo.vllm");
  assert.ok(!dynamo.prefill.argv.includes("--port"));
  assert.equal(JSON.parse(flag(dynamo.prefill.argv, "--kv-transfer-config")).kv_role, "kv_both");
  assert.equal(resolve({ strategy: "single_node_tp" }).registration, undefined);
});
