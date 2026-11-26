import assert from "node:assert/strict";
import test from "node:test";
import { resolveClientCommands } from "../src/lib/client-commands.js";

const client = { host: "serve.example", port: "8123" };

function requestBody(command) {
  return JSON.parse(command.split("  -d '")[1].slice(0, -1));
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
