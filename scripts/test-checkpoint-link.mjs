import assert from "node:assert/strict";
import { test } from "node:test";
import { checkpointLink, hasExternalModelReference, hasManualModelScopeEnv } from "../src/lib/checkpoint-link.js";
import { loadTaxonomy } from "../src/lib/taxonomy.js";

const taxonomy = loadTaxonomy();
const recipe = { model: { model_id: "org/base" } };

test("the variant hub determines the checkpoint link", () => {
  assert.equal(checkpointLink(recipe, {}, taxonomy).href, "https://huggingface.co/org/base");
  assert.deepEqual(checkpointLink(recipe, { model_id: "org/quantized", hub: "modelscope" }, taxonomy), {
    modelId: "org/quantized", href: "https://modelscope.cn/models/org/quantized", provider: "ModelScope", hub: "modelscope",
  });
  assert.equal(checkpointLink({}, {}, taxonomy), null);
  assert.equal(checkpointLink(recipe, { hub: "unknown" }, taxonomy), null);
});

test("independent tokenizer and draft references are detected", () => {
  assert.equal(hasExternalModelReference({ features: { spec_decoding: { args: ["--speculative-config", '{"method":"mtp"}'] } } }), false);
  assert.equal(hasExternalModelReference({ features: { spec_decoding: { args: ["--speculative-config", '{"model":"org/draft"}'] } } }), true);
  assert.equal(hasExternalModelReference({ model: { base_args: ["--tokenizer=org/tokenizer"] } }), true);
  assert.equal(hasManualModelScopeEnv({ variants: { a: { extra_env: { VLLM_USE_MODELSCOPE: "True" } } } }), true);
});
