import assert from "node:assert/strict";
import test from "node:test";
import fs from "node:fs";
import { execFileSync } from "node:child_process";
import yaml from "js-yaml";
import * as synthesis from "../src/lib/command-synthesis.js";
import { loadStrategies } from "../src/lib/strategies.js";

const taxonomy = yaml.load(fs.readFileSync("taxonomy.yaml", "utf8"));
const strategies = loadStrategies();
const cases = [
  ["GLM-5", "ascend_w4a4_local"],
  ["GLM-5.1", "ascend_w4a4_mxfp4"],
];
const readRecipe = (name) => yaml.load(fs.readFileSync(`models/zai-org/${name}.yaml`, "utf8"));
function render(recipe, variant, frontend) {
  return synthesis.resolveCommand(recipe, variant, "single_node_tep", "ascend_950dt",
    [], strategies, taxonomy, [], 1, null, {}, null, null, frontend);
}

for (const [name, variantKey] of cases) {
  const recipe = readRecipe(name);
  test(`${name}: A5 defaults to Python in the shared frontend resolver`, () => {
    assert.equal(synthesis.resolveFrontend(recipe, undefined, "ascend_950dt", "single_node_tep"), "python");
    assert.equal(synthesis.resolveFrontend(recipe, undefined, "h200", "single_node_tep"), "rust");
  });
  test(`${name}: explicit Rust is not silently replaced with Python`, () => {
    assert.equal(render(recipe, variantKey, "rust").env.VLLM_USE_RUST_FRONTEND, "1");
    assert.equal(render(recipe, variantKey, "python").env.VLLM_USE_RUST_FRONTEND, undefined);
    assert.equal(render(recipe, variantKey).env.VLLM_USE_RUST_FRONTEND, undefined);
  });
  test(`${name}: complete host setup and in-container serve are separate launch steps`, () => {
    const meta = synthesis.computeDockerMeta(recipe, recipe.variants[variantKey], taxonomy.hardware_profiles.ascend_950dt, "ascend_950dt");
    assert.equal(meta.execution?.context, "container");
    const result = render(recipe, variantKey);
    const steps = synthesis.buildExecutionSteps(meta, result.command, result.env);
    assert.deepEqual(steps.map((s) => s.context), ["host", "container"]);
    for (const path of ["/dev/ummu", "/dev/uburma", "/etc/hccl_rootinfo.json", "/etc/hixlep/", "/usr/bin/urma_admin", "~/.cache:/root/.cache"]) {
      assert(steps[0].command.includes(path), `Missing setup mount: ${path}`);
    }
    assert.match(steps[0].command, /--entrypoint \/bin\/bash/);
    assert(!steps[0].command.includes("vllm serve"));
    assert.match(steps[1].command, /mkdir -p \/dev\/shm\/vllm_metrics/);
    assert.match(steps[1].command, /export HCCL_IF_IP=\$\{HCCL_IF_IP:\?Set_HCCL_IF_IP\}/);
    assert.match(steps[1].command, /vllm serve /);
    assert.match(steps[1].command, /--tensor-parallel-size 8/);
    assert.match(steps[1].command, /--data-parallel-size 1/);
    assert(!steps[1].command.includes("docker run"));
    assert(!steps[1].command.includes("--nnodes"));
    if (name === "GLM-5") {
      assert.match(steps[0].command, /GLM5_MODEL_PATH:\?Set_GLM5_MODEL_PATH/);
      assert.match(steps[0].command, /dst=\/models\/GLM5-w4a4,readonly/);
      assert.match(steps[1].command, /cd \/models/);
    } else {
      assert.match(steps[1].command, /export VLLM_USE_MODELSCOPE=True/);
    }
  });
}

test("frontend precedence: explicit choice, strategy hardware, hardware, recipe", () => {
  const recipe = {
    model: { default_frontend: "rust" },
    hardware_overrides: { a5: { frontend: "python" } },
    strategy_overrides: { single_node_tp: { hardware_overrides: { a5: { frontend: "rust" } } } },
  };
  assert.equal(synthesis.resolveFrontend(recipe, "python", "a5", "single_node_tp"), "python");
  assert.equal(synthesis.resolveFrontend(recipe, undefined, "a5", "single_node_tp"), "rust");
  assert.equal(synthesis.resolveFrontend(recipe, undefined, "a5", "single_node_tep"), "python");
  assert.equal(synthesis.resolveFrontend(recipe, undefined, "h200"), "rust");
});

test("API publishes the shared launch steps instead of an incomplete Docker wrapper", () => {
  execFileSync(process.execPath, ["scripts/build-recipes-api.mjs"], { stdio: "pipe" });
  const api = JSON.parse(fs.readFileSync("public/Eco-Tech/GLM-5.1-w4a4c8-mxfp4/hw/ascend_950dt.json", "utf8"));
  assert.equal(api.execution_context, "container");
  assert.equal(api.docker_command, null);
  assert.equal(api.docker_argv, null);
  assert.deepEqual(api.execution_steps.map((s) => s.context), ["host", "container"]);
  assert(api.execution_steps[0].command.includes("/dev/ummu"));
  assert(api.execution_steps[0].command.includes("~/.cache:/root/.cache"));
  assert(api.execution_steps[1].command.includes(api.command));
  assert.equal(api.env.VLLM_USE_RUST_FRONTEND, undefined);
  const nvidia = JSON.parse(fs.readFileSync("public/zai-org/GLM-5.1/hw/h200.json", "utf8"));
  assert.equal(nvidia.execution_steps, undefined);
  assert.equal(nvidia.execution_context, undefined);
  assert.equal(nvidia.env.VLLM_USE_RUST_FRONTEND, "1");
  assert.equal(synthesis.computeDockerMeta(readRecipe("GLM-5.1"), {}, taxonomy.hardware_profiles.h200, "h200").execution, undefined);
});
