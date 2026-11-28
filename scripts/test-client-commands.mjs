import assert from "node:assert/strict";
import test from "node:test";
import { resolveClientCommands } from "../src/lib/client-commands.js";

const client = { host: "serve.example", port: "8123" };

function requestBody(command) {
  const data = command.match(/  (?:-d|--data-raw) '([\s\S]*)'$/);
  assert.ok(data, "expected a quoted JSON request body");
  return JSON.parse(data[1].replaceAll("'\\''", "'"));
}

test("the Qwen VL reranker gets a rerank request and benchmark", () => {
  const modelId = "Qwen/Qwen3-VL-Reranker-2B";
  const { verifyCmd, benchCmd } = resolveClientCommands({
    ...client,
    modelId,
    tasks: ["embedding", "multimodal"],
    argv: ["vllm", "serve", modelId, "--runner", "pooling",
      "--chat-template", "/vllm-workspace/vllm/examples/pooling/score/template/qwen3_vl_reranker.jinja"],
  });

  assert.match(verifyCmd, /^curl http:\/\/serve\.example:8123\/v1\/rerank /);
  const body = requestBody(verifyCmd);
  assert.equal(body.model, modelId);
  assert.equal(typeof body.query, "string");
  assert.equal(body.documents.length, 2);
  assert.match(benchCmd, /--backend vllm-rerank/);
  assert.match(benchCmd, /--endpoint \/v1\/rerank/);
  assert.match(benchCmd, /--dataset-name random-rerank/);
  assert.match(benchCmd, /--host serve\.example/);
  assert.match(benchCmd, /--port 8123/);
  assert.doesNotMatch(benchCmd, /--random-output-len/);
});

test("an embedding variant gets an embeddings request and benchmark", () => {
  const modelId = "jinaai/jina-embeddings-v5-text-small-retrieval";
  const { verifyCmd, benchCmd } = resolveClientCommands({
    ...client,
    modelId,
    tasks: ["embedding"],
  });

  assert.match(verifyCmd, /\/v1\/embeddings/);
  assert.deepEqual(requestBody(verifyCmd), { model: modelId, input: "Hello" });
  assert.match(benchCmd, /--backend openai-embeddings/);
  assert.match(benchCmd, /--endpoint \/v1\/embeddings/);
  assert.doesNotMatch(benchCmd, /--random-output-len/);
});

test("the selected serving arguments can specify a pooling task", () => {
  for (const argv of [
    ["--runner", "pooling", "--pooler-config", '{"task":"embed"}'],
    ["--runner=pooling", "--pooler-config.task=embed"],
    ["--convert=embed"],
  ]) {
    const { verifyCmd } = resolveClientCommands({ ...client, modelId: "org/custom-model", argv });
    assert.match(verifyCmd, /\/v1\/embeddings/);
  }
});

test("pooling tasks without a supported example do not get chat commands", () => {
  for (const [modelId, argv] of [
    ["org/classifier", ["--runner", "pooling"]],
    ["org/embedding-model", ["--runner", "pooling", "--pooler-config.task", "classify"]],
  ]) {
    assert.deepEqual(resolveClientCommands({ ...client, modelId, tasks: ["embedding"], argv }),
      { verifyCmd: null, benchCmd: null });
  }
});

test("generation recipes keep the chat example and generation benchmark", () => {
  const modelId = "Qwen/Qwen3-0.6B";
  const { verifyCmd, benchCmd } = resolveClientCommands({ ...client, modelId, tasks: ["text"] });
  assert.match(verifyCmd, /\/v1\/chat\/completions/);
  assert.deepEqual(requestBody(verifyCmd), {
    model: modelId,
    messages: [{ role: "user", content: "Hello" }],
    max_tokens: 32,
  });
  assert.match(benchCmd, /--dataset-name random /);
  assert.match(benchCmd, /--random-output-len 1024/);
});

const declaredClient = {
  endpoint: "/v2/custom-score",
  request_body: {
    query: "<Instruct>: Retrieve relevant passages.\n<Query>: What is the capital of China?\n",
    documents: ["<Document>: It's Beijing. Literal $HEAD_IP.\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"],
  },
  benchmark: {
    backend: "vllm-rerank",
    dataset_name: "random-rerank",
    args: ["--random-input-len", "200", "--num-prompts", "17"],
  },
};

test("a declared client overrides recognized pooling examples with the selected model and target", () => {
  for (const [modelId, host, port, argv] of [
    ["Qwen/Qwen3-Reranker-0.6B", "router.example", "31000", ["--runner", "pooling"]],
    ["org/alternate-embedding-model", "head.example", "9002", ["--convert=embed"]],
  ]) {
    const commands = resolveClientCommands({
      modelId, host, port, argv, tasks: ["embedding"], client: declaredClient,
    });
    assert.deepEqual(Object.keys(commands).sort(), ["benchCmd", "verifyCmd"]);
    assert.ok(commands.verifyCmd.includes("http://" + host + ":" + port + declaredClient.endpoint));
    assert.match(commands.verifyCmd, /^curl --fail-with-body -sS -X POST /);
    assert.deepEqual(requestBody(commands.verifyCmd), { model: modelId, ...declaredClient.request_body });
    assert.ok(commands.benchCmd.includes("--model " + modelId));
    assert.ok(commands.benchCmd.includes("--host " + host));
    assert.ok(commands.benchCmd.includes("--port " + port));
    assert.match(commands.benchCmd, /--endpoint \/v2\/custom-score/);
    assert.match(commands.benchCmd, /--backend vllm-rerank/);
    assert.match(commands.benchCmd, /--dataset-name random-rerank/);
    assert.match(commands.benchCmd, /--random-input-len 200/);
    assert.match(commands.benchCmd, /--num-prompts 17/);
    assert.doesNotMatch(commands.benchCmd, /--random-output-len/);
  }
});

test("a declared custom endpoint is available when the pooling heuristic has no example", () => {
  const selection = {
    ...client, modelId: "org/custom-pooling-checkpoint", tasks: ["embedding"],
    argv: ["--runner=pooling", "--pooler-config.task=classify"],
  };
  assert.deepEqual(resolveClientCommands(selection), { verifyCmd: null, benchCmd: null });
  const commands = resolveClientCommands({ ...selection, client: declaredClient });
  assert.deepEqual(requestBody(commands.verifyCmd), { model: selection.modelId, ...declaredClient.request_body });
  assert.ok(commands.verifyCmd.includes("http://serve.example:8123/v2/custom-score"));
  assert.match(commands.benchCmd, /--endpoint \/v2\/custom-score/);
  assert.doesNotMatch(commands.verifyCmd, /\/v1\/chat\/completions/);
});

test("invalid client declarations do not silently fall back to inferred examples", () => {
  const selection = { ...client, modelId: "org/embedding-model", tasks: ["embedding"] };
  const wrongModel = structuredClone(declaredClient);
  wrongModel.request_body.model = "org/wrong-model";
  assert.throws(() => resolveClientCommands({ ...selection, client: wrongModel }), /must not declare model/);
  const wrongTarget = structuredClone(declaredClient);
  wrongTarget.benchmark.args.push("--host=wrong.example");
  assert.throws(() => resolveClientCommands({ ...selection, client: wrongTarget }), /must not override/);
});

test("an explicit generate runner overrides the embedding category unless a client is declared", () => {
  const selection = {
    ...client, modelId: "org/embedding-model", tasks: ["embedding"], argv: ["--runner=generate"],
  };
  const inferred = resolveClientCommands(selection);
  assert.match(inferred.verifyCmd, /\/v1\/chat\/completions/);
  assert.match(inferred.benchCmd, /--random-output-len 1024/);
  const declared = resolveClientCommands({ ...selection, client: declaredClient });
  assert.match(declared.verifyCmd, /\/v2\/custom-score/);
  assert.deepEqual(requestBody(declared.verifyCmd), { model: selection.modelId, ...declaredClient.request_body });
});

test("the last repeated serving option determines the inferred client", () => {
  for (const [argv, endpoint] of [
    [["--runner", "pooling", "--runner=generate"], "/v1/chat/completions"],
    [["--runner=generate", "--runner", "pooling"], "/v1/embeddings"],
    [["--runner=pooling", "--pooler-config.task", "embed", "--pooler-config.task=score"], "/v1/rerank"],
    [["--runner=pooling", "--pooler-config.task=score", "--pooler-config.task", "embed"], "/v1/embeddings"],
  ]) {
    const commands = resolveClientCommands({ ...client, modelId: "org/embedding-model", tasks: ["embedding"], argv });
    assert.ok(commands.verifyCmd.includes("http://serve.example:8123" + endpoint), argv.join(" "));
  }
});
