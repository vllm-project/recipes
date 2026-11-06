import assert from "node:assert/strict";
import { test } from "node:test";
import { checkpointLink } from "../src/lib/checkpoint-link.js";

const recipe = { model: { model_id: "org/base" } };

test("checkpoint links default to the variant or model Hugging Face repository", () => {
  assert.deepEqual(checkpointLink(recipe, {}), {
    modelId: "org/base", href: "https://huggingface.co/org/base", provider: "Hugging Face",
  });
  assert.equal(checkpointLink(recipe, { model_id: "org/quantized" }).href, "https://huggingface.co/org/quantized");
  assert.equal(checkpointLink({}, {}), null);
});

test("explicit HTTPS checkpoint links identify their provider", () => {
  assert.deepEqual(checkpointLink(recipe, { model_id: "org/quantized", model_url: "https://modelscope.cn/models/org/quantized" }), {
    modelId: "org/quantized", href: "https://modelscope.cn/models/org/quantized", provider: "ModelScope",
  });
  assert.equal(checkpointLink(recipe, { model_url: "https://www.modelscope.cn/models/org/base" }).provider, "ModelScope");
  assert.equal(checkpointLink(recipe, { model_url: "https://example.org/model" }).provider, "Checkpoint");
});

test("unsafe or malformed links fall back to Hugging Face", () => {
  for (const model_url of ["javascript:alert(1)", "http://modelscope.cn/models/org/base", "https://"]) {
    assert.equal(checkpointLink(recipe, { model_url }).href, "https://huggingface.co/org/base");
  }
});
