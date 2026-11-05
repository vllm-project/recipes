// Focused contract tests: no assembled catalog, server or network needed.
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { test } from "node:test";
import yaml from "js-yaml";
import Ajv from "ajv";
import { renderRecipeClients, resolveClientTarget } from "../src/lib/recipe-clients.js";
import { buildRecipeSchema } from "../src/lib/recipe-schema.js";
import { buildVariantRendering, renderCommand, resolveSelection } from "../src/lib/recipe-render.js";
import { selectionMarkdown } from "../src/lib/recipe-markdown.js";
import { resolveCommand } from "../src/lib/command-synthesis.js";
import { resolveOmniTasks } from "../src/lib/omni-tasks.js";

const load = (url) => yaml.load(fs.readFileSync(url, "utf8"));
const taxonomy = load(new URL("../taxonomy.yaml", import.meta.url));
const strategies = Object.fromEntries(["strategies", "kv_store"].flatMap((dir) => {
  const url = new URL("../" + dir + "/", import.meta.url);
  return fs.readdirSync(url).filter((file) => file.endsWith(".yaml")).map((file) => {
    const strategy = load(new URL(file, url));
    return [strategy.name, strategy];
  });
}));
const validateSource = new Ajv({ allErrors: true, strict: false }).compile(buildRecipeSchema({ taxonomy, strategies }));
function validate(recipe) {
  // hf_id is loader metadata, not an authored YAML key.
  const { hf_id, ...source } = recipe;
  const valid = validateSource(source);
  validate.errors = validateSource.errors;
  return valid;
}
// Complete original Qwen format; the helper has no model-specific branch.
const prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n';
const suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
const instruction = "Given a web search query, retrieve relevant passages that answer the query";
const documents = [
  "The capital of China is Beijing.",
  "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
];
const requestBody = {
  query: prefix + "<Instruct>: " + instruction + "\n<Query>: What is the capital of China?\n",
  documents: documents.map((doc) => "<Document>: " + doc + suffix),
};
const fixture = {
  hf_id: "Qwen/Qwen3-Reranker-0.6B",
  meta: { title: "Reranker fixture", provider: "Qwen", description: "Client fixture", date_updated: "2026-11-04", tasks: ["embedding"] },
  model: { model_id: "Qwen/Qwen3-Reranker-0.6B", architecture: "dense", parameter_count: "0.6B", base_args: ["--runner", "pooling"] },
  client: {
    endpoint: "/v1/rerank", request_body: requestBody,
    benchmark: { backend: "vllm-rerank", dataset_name: "random-rerank", args: ["--random-input", "200", "--num-prompts", "100"] },
  },
  variants: {
    default: { precision: "bf16", vram_minimum_gb: 2 },
    alternate: { model_id: "example/alternate-checkpoint", precision: "bf16", vram_minimum_gb: 2 },
  },
  compatible_strategies: ["single_node_tp", "multi_node_tp", "pd_cluster"],
  guide: "Fixture guide.",
};
const fresh = () => structuredClone(fixture);

test("schema accepts clients and rejects incomplete, unknown or incompatible declarations", () => {
  assert.equal(validate(fresh()), true, JSON.stringify(validate.errors));
  for (const mutate of [
    (r) => { r.client.extra = true; },
    (r) => { r.client.benchmark.extra = true; },
    (r) => { delete r.client.endpoint; },
    (r) => { delete r.client.request_body; },
    (r) => { delete r.client.benchmark; },
    (r) => { delete r.client.benchmark.backend; },
    (r) => { delete r.client.benchmark.dataset_name; },
    (r) => { r.client.benchmark.backend = ""; },
    (r) => { r.client.benchmark.args = ["--x", 1]; },
    (r) => { r.client.request_body = []; },
    (r) => { r.client.request_body = {}; },
    (r) => { r.client.request_body.model = "example/wrong"; },
    (r) => { r.meta.tasks = ["omni"]; },
  ]) {
    const recipe = fresh();
    mutate(recipe);
    assert.equal(validate(recipe), false, JSON.stringify(recipe.client));
  }
  for (const endpoint of ["https://example.org/v1/rerank", "//example.org/rerank", "/v1/a b", "/v1/rerank?host=elsewhere"]) {
    const recipe = fresh();
    recipe.client.endpoint = endpoint;
    assert.equal(validate(recipe), false, endpoint);
  }
});

test("schema and helper reject benchmark target and workload overrides", () => {
  for (const arg of [
    "--model", "--model=example/wrong", "--host", "--host=wrong", "--port", "--port=9000",
    "--endpoint", "--endpoint=/wrong", "--backend", "--backend=openai", "--dataset",
    "--dataset-name", "--dataset-name=random", "--dataset_name=random", "--base-url=http://wrong", "-m", "-mexample/wrong", "--host wrong",
  ]) {
    const recipe = fresh();
    recipe.client.benchmark.args = [arg];
    assert.equal(validate(recipe), false, arg);
    assert.throws(() => renderRecipeClients(recipe, "default", {}), /must not override/, arg);
  }
  const recipe = fresh();
  recipe.client.request_body.model = "example/wrong";
  assert.throws(() => renderRecipeClients(recipe, "default", {}), /must not declare model/);
});

test("targets preserve single, multi, PD and routed/routerless KV behavior", () => {
  for (const [result, hostKey, portKey, port] of [
    [{ deployType: "single_node" }, "VLLM_HOST", "VLLM_PORT", "8000"],
    [{ deployType: "multi_node" }, "HEAD_IP", "VLLM_PORT", "8000"],
    [{ deployType: "pd_cluster" }, "ROUTER_HOST", "ROUTER_PORT", "30000"],
    [{ deployType: "kv_store_lb", router: { port: 31080 } }, "ROUTER_HOST", "ROUTER_PORT", "31080"],
    [{ deployType: "kv_store_lb", router: {} }, "ROUTER_HOST", "ROUTER_PORT", "30080"],
    [{ deployType: "kv_store_lb", router: null }, "VLLM_HOST", "VLLM_PORT", "8000"],
  ]) {
    assert.deepEqual(resolveClientTarget(result), { hostKey, portKey, host: "localhost", port });
    const endpoints = { [hostKey]: "worker.example", [portKey]: "12345", UNUSED_HOST: "wrong" };
    const clients = renderRecipeClients(fixture, "alternate", result, endpoints);
    assert.equal(clients.model_id, "example/alternate-checkpoint");
    assert.equal(clients.host, "worker.example");
    assert.equal(clients.port, "12345");
    assert.ok(clients.verify_command.includes("http://worker.example:12345/v1/rerank"));
    assert.ok(clients.benchmark_command.includes("--host worker.example"));
    assert.ok(clients.benchmark_command.includes("--port 12345"));
    assert.ok(clients.benchmark_command.includes("--model example/alternate-checkpoint"));
  }
});

const posixShell = [
  process.env.RECIPES_TEST_SHELL,
  process.platform === "win32" && path.join(process.env.ProgramFiles || "C:/Program Files", "Git/bin/bash.exe"),
  "/bin/sh",
].find((candidate) => candidate && fs.existsSync(candidate));
function shellArgs(name, command) {
  // Shadow executables and inspect real shell argv without contacting a server.
  const script = name + '() { printf "%s\\0" "$@"; }\n' + command;
  const result = spawnSync(posixShell, ["-c", script], { encoding: "utf8" });
  assert.ifError(result.error);
  assert.equal(result.status, 0, result.stderr);
  return result.stdout.split("\0").slice(0, -1);
}

test("real shell preserves Qwen data, apostrophes, newlines and literal dollars", { skip: !posixShell && "POSIX shell unavailable" }, () => {
  const recipe = fresh();
  const literal = "A document's $HEAD_IP and $(printf INJECTED)\nsecond line";
  recipe.client.request_body.documents.push("<Document>: " + literal + suffix);
  recipe.client.benchmark.args.push("--result-dir", literal);
  const clients = renderRecipeClients(recipe, "alternate", {}, { VLLM_HOST: "a host's $HOME", VLLM_PORT: "9001" });
  const curlArgs = shellArgs("curl", clients.verify_command);
  assert.ok(curlArgs.includes("--fail-with-body"));
  assert.ok(curlArgs.includes("http://a host's $HOME:9001/v1/rerank"));
  const body = JSON.parse(curlArgs[curlArgs.indexOf("--data-raw") + 1]);
  assert.deepEqual(body, { model: "example/alternate-checkpoint", ...recipe.client.request_body });
  assert.equal(body.query, requestBody.query);
  assert.equal(body.documents[0], "<Document>: " + documents[0] + suffix);
  const benchArgs = shellArgs("vllm", clients.benchmark_command);
  assert.deepEqual(benchArgs.slice(0, 2), ["bench", "serve"]);
  assert.equal(benchArgs[benchArgs.indexOf("--host") + 1], "a host's $HOME");
  assert.equal(benchArgs[benchArgs.indexOf("--model") + 1], "example/alternate-checkpoint");
  assert.equal(benchArgs[benchArgs.indexOf("--result-dir") + 1], literal);
});

test("UI helper, static variants, query JSON and Markdown generate the same clients", () => {
  for (const variant of ["default", "alternate"]) {
    for (const hardware of ["h200", "mi300x"]) {
      const built = buildVariantRendering(fixture, variant, hardware, strategies, taxonomy);
      assert.ok(built);
      for (const strategy of ["single_node_tp", "multi_node_tp"]) {
        const expected = strategy === built.recommended.strategy ? built.recommended : built.alternatives[strategy];
        assert.ok(expected, hardware + "/" + strategy);
        const query = new URLSearchParams({ hardware, variant, strategy, nodes: String(expected.node_count) });
        const resolved = resolveSelection(fixture, query, strategies, taxonomy);
        assert.deepEqual(resolved.rendering.clients, expected.clients);
        const uiResult = resolveCommand(fixture, variant, strategy, hardware, [], strategies, taxonomy, [], expected.node_count);
        assert.deepEqual(renderRecipeClients(fixture, variant, uiResult), expected.clients);
        const md = selectionMarkdown(fixture, resolved, {
          strategies, taxonomy, urls: { page: "https://example.org/Qwen/Qwen3-Reranker-0.6B", json: "https://example.org/Qwen/Qwen3-Reranker-0.6B.json" },
        });
        assert.ok(md.includes("## Verify\n"));
        assert.ok(md.includes(expected.clients.verify_command));
        assert.ok(md.includes(expected.clients.benchmark_command));
        assert.ok(md.includes("local to that browser"));
        const target = resolveClientTarget(uiResult);
        const custom = renderRecipeClients(fixture, variant, uiResult, { [target.hostKey]: "custom.example", [target.portKey]: "9002" });
        assert.equal(custom.host, "custom.example");
        assert.equal(resolved.rendering.clients.host, "localhost");
      }
    }
  }
});

test("Next production minification preserves exact client command bytes", async () => {
  // The static API imports source directly; Next's production routes minify it.
  // A fixed literal-array join was folded into a template that lost shell
  // backslash-newlines, which source-only parity tests cannot detect.
  const { default: swc } = await import("next/dist/build/swc/index.js");
  const source = fs.readFileSync(new URL("../src/lib/recipe-clients.js", import.meta.url), "utf8");
  const compiled = await swc.minify(source, { compress: true, mangle: true, module: true });
  const minified = await import("data:text/javascript;base64," + Buffer.from(compiled.code).toString("base64"));
  for (const variant of ["default", "alternate"]) {
    for (const result of [{ deployType: "single_node" }, { deployType: "pd_cluster" }]) {
      const endpoints = { VLLM_HOST: "worker.example", VLLM_PORT: "9002", ROUTER_HOST: "router.example", ROUTER_PORT: "31000" };
      const expected = renderRecipeClients(fixture, variant, result, endpoints);
      const actual = minified.renderRecipeClients(fixture, variant, result, endpoints);
      for (const key of ["verify_command", "benchmark_command"]) {
        assert.deepEqual(Buffer.from(actual[key], "utf8"), Buffer.from(expected[key], "utf8"), variant + "/" + result.deployType + "/" + key);
      }
      assert.deepEqual(actual, expected);
    }
  }
});

test("PD and routed/routerless KV selections expose the correct public clients", () => {
  for (const [query, deployType, port, hasRouter] of [
    ["strategy=pd_cluster&prefill_nodes=1&decode_nodes=1", "pd_cluster", "30000", true],
    ["strategy=single_node_tp&kv_offload=kv_store_distributed_mooncake&kv_instances=1", "kv_store_lb", "8000", false],
    ["strategy=single_node_tp&kv_offload=kv_store_distributed_mooncake&kv_instances=2", "kv_store_lb", "30080", true],
  ]) {
    const resolved = resolveSelection(fixture, new URLSearchParams("hardware=h200&variant=alternate&" + query), strategies, taxonomy);
    const rendering = resolved.rendering;
    assert.equal(rendering.deploy_type, deployType);
    assert.equal(!!rendering.router, hasRouter);
    assert.equal(rendering.clients.port, port);
    assert.equal(rendering.clients.host, "localhost");
    assert.equal(rendering.clients.model_id, "example/alternate-checkpoint");
    assert.deepEqual(rendering.clients, renderRecipeClients(fixture, "alternate", rendering));
    const md = selectionMarkdown(fixture, resolved, {
      strategies, taxonomy, urls: { page: "https://example.org/Qwen/Qwen3-Reranker-0.6B", json: "https://example.org/Qwen/Qwen3-Reranker-0.6B.json" },
    });
    assert.ok(md.includes(rendering.clients.verify_command));
    assert.ok(md.includes(rendering.clients.benchmark_command));
  }
});

test("legacy text renderings and Omni task clients remain opt-out", () => {
  const legacy = fresh();
  delete legacy.client;
  legacy.meta.tasks = ["text"];
  assert.equal(validate(legacy), true, JSON.stringify(validate.errors));
  const rendering = renderCommand(legacy, "default", "single_node_tp", "h200", 1, [], strategies, taxonomy);
  assert.equal(Object.hasOwn(rendering, "clients"), false);
  assert.equal(renderRecipeClients(legacy, "default", {}), null);
  const omni = {
    ...legacy, meta: { ...legacy.meta, tasks: ["omni"] },
    omni: { tasks: [{ id: "t2i", curl: "existing image client", benchmark: "existing image benchmark" }] },
  };
  const task = resolveOmniTasks(omni)[0];
  assert.equal(task.example({}), "existing image client");
  assert.equal(task.benchmark, "existing image benchmark");
  assert.equal(renderRecipeClients(omni, "default", {}), null);
  assert.equal(resolveSelection(omni, new URLSearchParams("hardware=h200"), strategies, taxonomy).rendering.clients, undefined);
});
