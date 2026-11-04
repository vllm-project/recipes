import assert from "node:assert/strict";
import { test } from "node:test";
import { clientModelFor, fillClientModel } from "../src/lib/client-model.js";
const recipe = { model: { model_id: "moonshotai/Kimi-K2.6" }, variants: { w4a8_ascend: { model_id: "Eco-Tech/Kimi-K2.6-w4a8" }, fp8: { model_id: "RedHatAI/Kimi-K2.6-FP8" } } };
test("client example follows the selected checkpoint", () => {
  assert.equal(clientModelFor(recipe, "w4a8_ascend"), "Eco-Tech/Kimi-K2.6-w4a8");
  assert.equal(clientModelFor(recipe, "fp8"), "RedHatAI/Kimi-K2.6-FP8");
  assert.equal(clientModelFor(recipe, "default"), "moonshotai/Kimi-K2.6");
});
test("served alias takes precedence across deployment shapes", () => {
  for (const r of [{ argv: ["--served-model-name", "api-model"] }, { headArgv: ["--served-model-name", "api-model"] }, { prefill: { argv: ["--served-model-name", "api-model"] } }, { vllm_argv: ["--served-model-name=api-model"] }]) {
    assert.equal(clientModelFor(recipe, "fp8", r), "api-model");
  }
});
test("placeholder emits an escaped string and leaves other models untouched", () => {
  assert.equal(JSON.parse(fillClientModel('model={{client_model}}', 'a"b').slice(6)), 'a"b');
  assert.equal(fillClientModel('model="fixed"', "other"), 'model="fixed"');
});
