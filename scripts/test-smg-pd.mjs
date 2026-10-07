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
const resolve = ({ router = "smg", hardware = "gb300", pools = { prefill: { nodes: 1 }, decode: { nodes: 1 } }, offload = null, model = recipe, strategy = "pd_cluster", transport = "http", features = ["tool_calling", "reasoning"], variant = "default", frontend = undefined } = {}) =>
  resolveCommand(model, variant, strategy, hardware, features, strategies, taxonomy, [], 1, pools, {}, offload, null, frontend, router, transport);

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
  const routerDocker = substitute(result.router.dockerCommand, { ROUTER_PORT: "31000" });
  assert.match(routerDocker, /^docker run --rm --network host/);
  assert.match(routerDocker, /lightseekorg\/smg:1\.11\.0/);
  assert.equal(result.router.dockerInstall, "docker pull lightseekorg/smg:1.11.0");
  assert.match(routerDocker, /--pd-disaggregation/);
  assert.match(routerDocker, /--port 31000/);
  assert.doesNotMatch(routerDocker, /--gpus|vllm-openai|uv pip|smg launch/);
  execFileSync("bash", ["-n"], { input: routerDocker });
  const meta = computeDockerMeta(recipe, recipe.variants.default, taxonomy.hardware_profiles.gb300, "gb300");
  const command = buildDockerRun({ command: result.prefill.command, env: result.prefill.env, image: meta.image, gpuFlags: meta.gpuFlags });
  assert.match(command, /vllm\/vllm-openai:v0\.31\.0/);
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

const grpcModels = [
  ["moonshotai/Kimi-K3", "kimi_k3", "kimi_k3"],
  ["MiniMaxAI/MiniMax-M3", "minimax_m3", "minimax_m3"],
  ["zai-org/GLM-5.3", "glm45", "glm47_moe"],
  ["deepseek-ai/DeepSeek-V4.1-Flash", "deepseek_v41", "deepseek_v41"],
];
const grpcPools = { prefill: { nodes: 2, parallelism: "tp" }, decode: { nodes: 2, parallelism: "tep" } };
for (const [id, reasoning, tools] of grpcModels) {
  test(`SMG gRPC ${id} routes to pool heads and parses at the gateway`, () => {
    const model = read(`models/${id}.yaml`);
    const result = resolve({ model, transport: "grpc", pools: grpcPools, frontend: "rust" });
    assert.equal(result.transport, "grpc");
    assert.equal(result.orchestrator, "smg");
    assert.equal(result.registration, undefined);
    for (const [role, kvRole] of [["prefill", "kv_producer"], ["decode", "kv_consumer"]]) {
      assert.ok(result[role].argv.includes("--grpc"));
      assert.equal(flag(result[role].argv, "--host"), "0.0.0.0");
      assert.equal(result[role].env.VLLM_USE_RUST_FRONTEND, "0");
      assert.ok(!result[role].argv.includes("--tool-call-parser"));
      assert.ok(!result[role].argv.includes("--reasoning-parser"));
      assert.ok(!result[role].argv.includes("--enable-auto-tool-choice"));
      assert.equal(JSON.parse(flag(result[role].argv, "--kv-transfer-config")).kv_role, kvRole);
      if (model.features.text_only) assert.ok(!result[role].argv.includes("--language-model-only"));
    }
    assert.match(result.router.command, /--prefill grpc:\/\/\$PREFILL_NODE_1:8001/);
    assert.match(result.router.command, /--decode grpc:\/\/\$DECODE_NODE_1:8002/);
    assert.ok(result.router.command.includes(`--model-path ${id}`));
    assert.ok(result.router.command.includes(`--reasoning-parser ${reasoning}`));
    assert.ok(result.router.command.includes(`--tool-call-parser ${tools}`));
    assert.match(result.workerInstall, /smg-grpc-servicer>=/);
    assert.match(result.workerInstall, /smg-grpc-proto>=/);
    execFileSync("bash", ["-n"], { input: result.router.command });
  });
}

test("gRPC respects parser toggles, variant tokenizer IDs and headless follower ranks", () => {
  const model = read("models/moonshotai/Kimi-K3.yaml");
  const result = resolve({ model, variant: "nvfp4", transport: "grpc", features: [], pools: {
    prefill: { nodes: 2, rank: 1, parallelism: "tp" }, decode: { nodes: 2, rank: 1, parallelism: "tep" },
  } });
  assert.ok(result.prefill.argv.includes("--headless"));
  assert.ok(!result.prefill.argv.includes("--grpc"), "gRPC dispatch would bypass the headless launcher");
  assert.ok(!result.decode.argv.includes("--grpc"));
  assert.match(result.router.command, /--model-path RedHatAI\/Kimi-K3-NVFP4/);
  assert.match(result.router.command, /--reasoning-parser passthrough/);
  assert.match(result.router.command, /--tool-call-parser passthrough/);
  assert.doesNotMatch(result.router.command, /NODE_2/);
});

test("gRPC is opt-in only and unsupported selections never emit gRPC workers", () => {
  for (const id of ["XiaomiMiMo/MiMo-V2.6-Flash-RL", "zai-org/GLM-5.3-Flash"]) {
    const result = resolve({ model: read(`models/${id}.yaml`), transport: "grpc", pools: grpcPools });
    assert.ok(result.routerUnavailableReason);
    assert.ok(!result.prefill.argv.includes("--grpc"));
  }
  const model = read("models/zai-org/GLM-5.3.yaml");
  for (const options of [{}, { router: "vllm", transport: "grpc" }, { router: "dynamo", transport: "grpc" },
    { transport: "grpc", offload: "simple" }, { transport: "grpc", pools: { prefill: { parallelism: "dep" } } }]) {
    const result = resolve({ model, pools: grpcPools, ...options });
    assert.ok(!result.prefill.argv.includes("--grpc"));
  }
});

test("gRPC Docker installs dependencies inside the vLLM image and exposes NIXL on host networking", () => {
  const model = read("models/MiniMaxAI/MiniMax-M3.yaml");
  const result = resolve({ model, transport: "grpc", pools: grpcPools });
  const meta = computeDockerMeta(model, model.variants.default, taxonomy.hardware_profiles.gb300, "gb300");
  const command = buildDockerRun({ command: result.prefill.command, env: result.prefill.env,
    image: meta.image, gpuFlags: meta.gpuFlags, setupCommand: result.workerDockerSetup, hostNetwork: result.transport === "grpc" });
  assert.match(command, /--network host/);
  assert.match(command, /--entrypoint \/bin\/sh/);
  assert.match(command, /vllm\/vllm-openai:minimax-m3/);
  assert.match(command, /python3 -m pip install.*smg-grpc-servicer>=/);
  assert.match(command, /exec vllm serve/);
  assert.doesNotMatch(command, / -p /);
  const router = result.router.dockerCommand;
  assert.match(router, /--prefill grpc:/);
  assert.match(router, /-v ~\/.cache\/huggingface:\/root\/.cache\/huggingface/);
  assert.doesNotMatch(router, /--gpus|smg launch/);
  for (const script of [command, router]) execFileSync("bash", ["-n"], { input: script });
  // Exercise both shell layers without launching containers or installing packages.
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "recipes-grpc-docker-"));
  try {
    fs.writeFileSync(path.join(dir, "docker"), '#!/usr/bin/env node\nconsole.log(JSON.stringify(process.argv.slice(2)));\n', { mode: 0o755 });
    const env = { ...process.env, PATH: `${dir}:${process.env.PATH}`, PREFILL_NODE_1: "192.0.2.11", IFACE_NAME: "eth0" };
    const argv = JSON.parse(execFileSync("bash", ["-c", command], { env, encoding: "utf8" }));
    const scriptIndex = argv.indexOf("-c");
    assert.ok(scriptIndex > argv.indexOf(meta.image));
    fs.writeFileSync(path.join(dir, "python3"), '#!/bin/sh\nexit 0\n', { mode: 0o755 });
    fs.writeFileSync(path.join(dir, "vllm"), '#!/usr/bin/env node\nconsole.log(JSON.stringify(process.argv.slice(2)));\n', { mode: 0o755 });
    const worker = JSON.parse(execFileSync("sh", argv.slice(scriptIndex), { env, encoding: "utf8" }));
    assert.deepEqual(worker.slice(0, 2), ["serve", "MiniMaxAI/MiniMax-M3"]);
    assert.ok(worker.includes("--grpc"));
    assert.equal(flag(worker, "--master-addr"), "192.0.2.11");
    assert.equal(JSON.parse(flag(worker, "--kv-transfer-config")).kv_role, "kv_producer");
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test("gRPC preserves vision encoder settings while removing worker-only chat defaults", () => {
  const result = resolve({ model: read("models/MiniMaxAI/MiniMax-M3.yaml"), transport: "grpc", pools: grpcPools,
    features: ["tool_calling", "reasoning", "encoder_parallel", "thinking_always_on"] });
  for (const role of ["prefill", "decode"]) {
    assert.ok(!result[role].argv.includes("--default-chat-template-kwargs"));
    assert.equal(flag(result[role].argv, "--mm-encoder-tp-mode"), "data");
    assert.equal(flag(result[role].argv, "--mm-encoder-attn-backend"), "FLASHINFER");
    assert.equal(flag(result[role].argv, "--mm-processor-cache-type"), "shm");
    assert.ok(!result[role].argv.includes("--language-model-only"));
  }
});

test("gRPC Text Only remains an opt-in feature on both pools", () => {
  for (const id of ["moonshotai/Kimi-K3", "MiniMaxAI/MiniMax-M3", "deepseek-ai/DeepSeek-V4.1-Flash"]) {
    const model = read(`models/${id}.yaml`);
    for (const enabled of [false, true, false]) {
      const result = resolve({ model, transport: "grpc", pools: grpcPools,
        features: ["tool_calling", "reasoning", ...(enabled ? ["text_only"] : [])] });
      for (const role of ["prefill", "decode"]) {
        assert.equal(result[role].argv.includes("--language-model-only"), enabled, `${id} ${role}`);
        assert.ok(result[role].argv.includes("--grpc"));
      }
    }
  }
});
