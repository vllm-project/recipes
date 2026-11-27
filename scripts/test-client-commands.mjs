import assert from "node:assert/strict";
import test from "node:test";
import { resolveClientCommands } from "../src/lib/client-commands.js";
import { resolveCommand, resolveOmniCommand } from "../src/lib/command-synthesis.js";
import { resolveOmniTasks } from "../src/lib/omni-tasks.js";

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
  assert.doesNotMatch(benchCmd, /--served-model-name/);
});

test("generated requests follow the selected hardware's first served name", () => {
  const modelId = "org/checkpoint";
  const recipe = {
    model: { model_id: modelId, base_args: ["--served-model-name", "base"] },
    variants: { default: { hardware_overrides: {
      h200: { extra_args: ["--served-model-name", "primary", "secondary"] },
    } } },
  };
  for (const [hw, alias] of [["h100", "base"], ["h200", "primary"]]) {
    const { argv } = resolveCommand(recipe, "default", "single_node_tp", hw, [],
      { single_node_tp: {} }, { hardware_profiles: { h100: {}, h200: {} } });
    const { verifyCmd, benchCmd } = resolveClientCommands({ ...client, modelId, argv });
    assert.equal(requestBody(verifyCmd).model, alias);
    assert.ok(benchCmd.includes(`--served-model-name ${alias} `));
    assert.match(benchCmd, /--model org\/checkpoint /);
  }
});

test("pooling keeps its checkpoint-specific endpoint when the alias is opaque", () => {
  for (const [modelId, endpoint] of [["org/text-embeddings", "embeddings"], ["org/text-reranker", "rerank"]]) {
    const { verifyCmd, benchCmd } = resolveClientCommands({
      ...client, modelId, tasks: ["embedding"], argv: ["--served-model-name", "old", "--served-model-name=live"],
    });
    assert.ok(verifyCmd.includes(`/v1/${endpoint} `));
    assert.equal(requestBody(verifyCmd).model, "live");
    assert.ok(benchCmd.includes(`--model ${modelId} `));
    assert.match(benchCmd, /--served-model-name live /);
  }
});

test("a PD prefill alias is used when the decode command omits it", () => {
  const { verifyCmd, benchCmd } = resolveClientCommands({
    ...client, modelId: "org/checkpoint", argv: ["--port", "8002"], prefillArgv: ["--served-model-name", "live"],
  });
  assert.equal(requestBody(verifyCmd).model, "live");
  assert.match(benchCmd, /--served-model-name live /);
});

test("Omni examples follow the active task checkpoint and served name", () => {
  const recipe = { model: { model_id: "org/base" }, omni: { tasks: [{ id: "i2i", model_id: "org/image" }] } };
  const [task] = resolveOmniTasks(recipe);
  for (const alias of [null, "image-api"]) {
    const omniTask = { ...task, extraArgs: alias ? ["--served-model-name", alias] : [] };
    const { modelId, argv } = resolveOmniCommand(recipe, "default", omniTask, {});
    const { verifyCmd, benchCmd } = resolveClientCommands({ ...client, modelId, argv, omniTask });
    assert.equal(requestBody(verifyCmd).model, alias || "org/image");
    assert.match(benchCmd, /--model org\/image /);
    assert.equal(benchCmd.includes("--served-model-name image-api"), !!alias);
  }
});
