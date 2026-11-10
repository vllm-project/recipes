import assert from "node:assert/strict";
import { test } from "node:test";
import Ajv from "ajv";
import { loadTaxonomy } from "../src/lib/taxonomy.js";
import { loadStrategies } from "../src/lib/strategies.js";
import { resolveCommand } from "../src/lib/command-synthesis.js";
import { renderCommand } from "../src/lib/recipe-render.js";
import { buildRecipeSchema } from "../src/lib/recipe-schema.js";
import { applyOverlay, resolveScope, splitOverlay } from "../src/lib/overlay-merge.js";

const taxonomy = loadTaxonomy();
const strategies = loadStrategies();
const strategy = "multi_node_tp_dp";
const recipe = {
  meta: { title: "Worker args", provider: "Example", description: "Worker launch configuration", date_updated: "2026-11-10", tasks: ["text"] },
  model: { model_id: "example/model", architecture: "dense", parameter_count: "1B", docker_image: "example/serve:test" },
  variants: { default: { precision: "bf16", vram_minimum_gb: 1 } },
  compatible_strategies: ["single_node_tp", "multi_node_tp", strategy, "pd_cluster"],
  guide: "Example recipe.",
};
const overrides = {
  [strategy]: {
    hardware_overrides: {
      ascend_910c: { worker_args: ["--headless"] },
      ascend_950dt: { worker_args: ["--headless"] },
    },
  },
};
const configured = { ...recipe, strategy_overrides: overrides };
const flag = (argv, name) => argv[argv.indexOf(name) + 1];

test("hardware-scoped worker args reach every follower and Docker form, never the head", () => {
  for (const [hardware, tp] of [["ascend_910c", "16"], ["ascend_950dt", "8"]]) {
    for (const nodes of [2, 4]) {
      const result = renderCommand(configured, "default", strategy, hardware, nodes, [], strategies, taxonomy);
      assert.ok(!result.head_argv.includes("--headless"));
      assert.ok(!result.head_docker_argv.includes("--headless"));
      assert.equal(result.worker_argvs.length, nodes - 1);
      assert.deepEqual(result.worker_argv, result.worker_argvs[0]);
      for (const [i, argv] of result.worker_argvs.entries()) {
        assert.equal(argv.filter((arg) => arg === "--headless").length, 1);
        assert.equal(flag(argv, "--data-parallel-start-rank"), String(i + 1));
        assert.equal(flag(argv, "--tensor-parallel-size"), tp);
        assert.equal(flag(argv, "--data-parallel-size"), String(nodes));
        assert.ok(result.worker_docker_argvs[i].includes("--headless"));
        assert.match(result.worker_commands[i], /--headless/);
        assert.match(result.worker_docker_commands[i], /--headless/);
      }
    }
  }
});

test("worker args do not change other hardware or non-multi-node commands", () => {
  const baseline = resolveCommand(recipe, "default", strategy, "h200", [], strategies, taxonomy, [], 4);
  const otherHardware = resolveCommand(configured, "default", strategy, "h200", [], strategies, taxonomy, [], 4);
  assert.deepEqual(otherHardware, baseline);
  for (const selectedStrategy of ["single_node_tp", "pd_cluster"]) {
    const scoped = { ...recipe, strategy_overrides: { [selectedStrategy]: { worker_args: ["--headless"], hardware_overrides: { h200: { worker_args: ["--headless"] } } } } };
    const before = resolveCommand(recipe, "default", selectedStrategy, "h200", [], strategies, taxonomy, [], 1);
    const after = resolveCommand(scoped, "default", selectedStrategy, "h200", [], strategies, taxonomy, [], 1);
    assert.deepEqual(after, before);
  }
  const oneNode = resolveCommand(configured, "default", strategy, "ascend_950dt", [], strategies, taxonomy, [], 1);
  assert.ok(!oneNode.argv.includes("--headless"));
});

test("strategy, generation, and exact-hardware worker args layer without duplicating flags", () => {
  const layered = { ...recipe, strategy_overrides: { [strategy]: {
    worker_args: ["--headless"],
    hardware_overrides: { npu: { worker_args: ["--max-num-seqs", "8"] }, ascend_950dt: { worker_args: ["--headless", "--max-num-seqs", "16"] } },
  } } };
  const result = resolveCommand(layered, "default", strategy, "ascend_950dt", [], strategies, taxonomy, [], 4);
  assert.ok(!result.headArgv.includes("--headless"));
  assert.ok(!result.headArgv.includes("--max-num-seqs"));
  for (const argv of result.workerArgvs) {
    assert.equal(argv.filter((arg) => arg === "--headless").length, 1);
    assert.equal(flag(argv, "--max-num-seqs"), "16");
  }
});

test("a worker-only generation block preserves shared recipe args and environment", () => {
  const baseline = { ...recipe, hardware_overrides: { npu: {
    extra_args: ["--max-num-seqs", "8"], extra_env: { OMP_NUM_THREADS: "1" },
  } } };
  const scoped = { ...baseline, strategy_overrides: { [strategy]: {
    hardware_overrides: { npu: { worker_args: ["--headless"] } },
  } } };
  const before = resolveCommand(baseline, "default", strategy, "ascend_950dt", [], strategies, taxonomy, [], 4);
  const after = resolveCommand(scoped, "default", strategy, "ascend_950dt", [], strategies, taxonomy, [], 4);
  assert.deepEqual(after.headArgv, before.headArgv);
  assert.deepEqual(after.env, before.env);
  for (const argv of after.workerArgvs) {
    assert.equal(flag(argv, "--max-num-seqs"), "8");
    assert.ok(argv.includes("--headless"));
  }
});

test("schema accepts worker argument lists and rejects malformed values", () => {
  const validate = new Ajv({ allErrors: true, strict: false }).compile(buildRecipeSchema({ taxonomy, strategies }));
  assert.equal(validate(configured), true, JSON.stringify(validate.errors));
  const general = { ...recipe, strategy_overrides: { [strategy]: { worker_args: ["--headless"] } } };
  assert.equal(validate(general), true, JSON.stringify(validate.errors));
  for (const invalid of ["--headless", [true], [1]]) {
    const bad = structuredClone(configured);
    bad.strategy_overrides[strategy].hardware_overrides.ascend_950dt.worker_args = invalid;
    assert.equal(validate(bad), false);
  }
});

test("Ascend overlays preserve worker args inside their hardware scope", () => {
  const ctx = resolveScope({ brands: ["Huawei"] }, taxonomy);
  const overlay = { strategy_overrides: overrides };
  const { recipe: merged, errors } = applyOverlay(recipe, overlay, { ctx, source: "vllm-ascend" });
  assert.deepEqual(errors, []);
  assert.deepEqual(merged.strategy_overrides, overrides);
  const split = splitOverlay(merged, { ctx, source: "vllm-ascend" });
  assert.deepEqual(split.overlay.strategy_overrides, overrides);
  const outOfScope = { strategy_overrides: { [strategy]: { hardware_overrides: { h200: { worker_args: ["--headless"] } } } } };
  assert.ok(applyOverlay(recipe, outOfScope, { ctx, source: "vllm-ascend" }).errors.length > 0);
});
