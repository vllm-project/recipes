import assert from "node:assert/strict";
import fs from "node:fs";
import { test } from "node:test";
import yaml from "js-yaml";
import { resolveCommand } from "../src/lib/command-synthesis.js";
import { renderCommand, resolveSelection } from "../src/lib/recipe-render.js";
import { selectionMarkdown } from "../src/lib/recipe-markdown.js";

// Exercise the public renderer against local vocabulary and synthetic model
// inputs. No source checkout, model download, shell launcher or NPU is needed.
const read = (file) => yaml.load(fs.readFileSync(new URL(`../${file}`, import.meta.url), "utf8"));
const taxonomy = read("taxonomy.yaml");
const strategies = Object.fromEntries(["strategies", "kv_store"].flatMap((dir) =>
  fs.readdirSync(new URL(`../${dir}/`, import.meta.url)).filter((file) => file.endsWith(".yaml"))
    .map((file) => { const entry = read(`${dir}/${file}`); return [entry.name, entry]; })));
const DP_STRATEGIES = ["single_node_dep", "multi_node_dep", "multi_node_tp_dp"];
const urls = { page: "https://example.test/test/DP", json: "https://example.test/test/DP.json", md: "https://example.test/test/DP.md" };
const flag = (argv, key) => argv.includes(key) ? argv[argv.lastIndexOf(key) + 1] : undefined;

function fixture() {
  return {
    hf_id: "test/DP",
    meta: { title: "DP topology fixture", provider: "test", description: "Synthetic DP contract", tasks: ["text"],
      hardware: { ascend_910c: "supported", ascend_950dt: "supported" } },
    model: { model_id: "test/DP", architecture: "moe", parameter_count: "1B", context_length: 4096,
      base_args: [], docker_image: "example/vllm:fixture" },
    variants: { default: { precision: "bf16", vram_minimum_gb: 4,
      hardware_overrides: {
        ascend_910c: { docker_image: "example/ascend:fixture", docker_serve_command: "vllm serve" },
        ascend_950dt: { docker_image: "example/ascend:fixture", docker_serve_command: "vllm serve" },
      } } },
    compatible_strategies: [...DP_STRATEGIES],
    default_strategy: "single_node_dep",
    strategy_overrides: {},
    guide: "Synthetic recipe for deterministic topology tests.",
  };
}

function run(recipe = fixture(), { strategy = "multi_node_tp_dp", hw = "ascend_910c", nodes = 2,
  advanced = [], kv = null, preset, point, dcp = null } = {}) {
  return resolveCommand(recipe, "default", strategy, hw, [], strategies, taxonomy,
    advanced, nodes, null, {}, kv, null, "python", null, "http", dcp, preset, point);
}

function nodeArgvs(result) {
  assert.equal(result.deploymentError, undefined, result.deploymentError?.message);
  return result.deployType === "multi_node" ? [result.headArgv, ...result.workerArgvs] : [result.argv];
}

function assertLayout(result, { tp, global, local, nodes = 2, hybrid = false, external = false }) {
  const commands = nodeArgvs(result);
  assert.equal(commands.length, nodes);
  for (const [node, argv] of commands.entries()) {
    assert.equal(flag(argv, "--tensor-parallel-size"), String(tp), `node ${node}: TP`);
    assert.equal(flag(argv, "--data-parallel-size"), String(global), `node ${node}: global DP`);
    assert.equal(flag(argv, "--data-parallel-size-local"), String(local), `node ${node}: local DP`);
    assert.equal(argv.includes("--data-parallel-hybrid-lb"), hybrid);
    assert.equal(argv.includes("--data-parallel-external-lb"), external);
    assert.equal(argv.includes("--headless"), node > 0 && !hybrid && !external);
    if (external) {
      assert.equal(flag(argv, "--data-parallel-rank"), String(node));
      assert.equal(flag(argv, "--data-parallel-start-rank"), undefined);
    } else if (node > 0) {
      assert.equal(flag(argv, "--data-parallel-start-rank"), String(node * local));
      assert.equal(flag(argv, "--data-parallel-rank"), undefined);
    }
  }
}

function assertInvalid(result) {
  assert.equal(result.deploymentError?.code, "DP_INVALID_LAYOUT", JSON.stringify(result));
  assert.ok(result.deploymentError.message);
  const inspect = (object) => {
    if (!object || typeof object !== "object") return;
    for (const [key, value] of Object.entries(object)) {
      if (/(?:command|argv)s?$/i.test(key)) assert.ok(!value || value.length === 0, `${key} must not be runnable`);
      else if (key !== "deploymentError") inspect(value);
    }
  };
  inspect(result);
}

test("TPDP retains one full-node TP replica per node unless TP is declared", () => {
  assertLayout(run(), { tp: 16, global: 2, local: 1 });
  for (const [tp, global, local] of [[8, 4, 2], [4, 8, 4]]) {
    const recipe = fixture();
    recipe.strategy_overrides.multi_node_tp_dp = { tp };
    assertLayout(run(recipe), { tp, global, local });
  }
});

test("four physical nodes use distinct DP starts 2, 4 and 6", () => {
  const recipe = fixture();
  recipe.strategy_overrides.multi_node_tp_dp = { tp: 8 };
  const result = run(recipe, { nodes: 4 });
  assertLayout(result, { tp: 8, global: 8, local: 2, nodes: 4 });
  assert.deepEqual(result.workerArgvs.map((argv) => flag(argv, "--data-parallel-start-rank")), ["2", "4", "6"]);
});

test("all ordinary DP strategies accept scalar and hardware-keyed TP", () => {
  for (const strategy of DP_STRATEGIES) {
    for (const tp of [8, { default: 1, huawei: 2, npu: 4, ascend_910c: 8 }]) {
      const recipe = fixture();
      recipe.strategy_overrides[strategy] = { tp };
      const nodes = strategy === "single_node_dep" ? 1 : 2;
      assertLayout(run(recipe, { strategy, nodes }), {
        tp: 8, global: nodes * 2, local: 2, nodes, hybrid: strategy === "multi_node_dep",
      });
    }
  }
});

test("DEP defaults to TP1 and single-node TP2 consumes sixteen devices as DP8", () => {
  for (const strategy of ["single_node_dep", "multi_node_dep"]) {
    const nodes = strategy === "single_node_dep" ? 1 : 2;
    assertLayout(run(fixture(), { strategy, nodes }), { tp: 1, global: nodes * 16, local: 16, nodes, hybrid: nodes > 1 });
  }
  const recipe = fixture();
  recipe.strategy_overrides.single_node_dep = { tp: 2 };
  assertLayout(run(recipe, { strategy: "single_node_dep", nodes: 1 }), { tp: 2, global: 8, local: 8, nodes: 1 });
});

test("effective raw TP participates in sizing; typed TP overrides earlier base and variant flags", () => {
  const recipe = fixture();
  recipe.model.base_args = ["--tensor-parallel-size", "2"];
  recipe.variants.default.extra_args = ["--tensor-parallel-size", "4"];
  assertLayout(run(recipe), { tp: 4, global: 8, local: 4 });
  recipe.strategy_overrides.multi_node_tp_dp = { tp: 8 };
  assertLayout(run(recipe), { tp: 8, global: 4, local: 2 });
});

test("later strategy, hardware and advanced TP flags retain last-wins precedence", () => {
  const recipe = fixture();
  recipe.strategy_overrides.multi_node_tp_dp = { tp: 8, extra_args: ["--tensor-parallel-size", "4"] };
  assertLayout(run(recipe), { tp: 4, global: 8, local: 4 });
  recipe.strategy_overrides.multi_node_tp_dp.hardware_overrides = { ascend_910c: { extra_args: ["--tensor-parallel-size", "2"] } };
  assertLayout(run(recipe), { tp: 2, global: 16, local: 8 });
  assertLayout(run(recipe, { advanced: ["--tensor-parallel-size", "8"] }), { tp: 8, global: 4, local: 2 });
});

test("explicit global or local DP may use fewer devices and derive the other size", () => {
  for (const args of [["--data-parallel-size", "2"], ["--data-parallel-size-local", "1"],
    ["--data-parallel-size", "2", "--data-parallel-size-local", "1"]]) {
    const recipe = fixture();
    recipe.strategy_overrides.multi_node_tp_dp = { tp: 8, extra_args: args };
    assertLayout(run(recipe), { tp: 8, global: 2, local: 1 });
  }
  const recipe = fixture();
  recipe.model.base_args = ["--tensor-parallel-size", "4", "--data-parallel-size", "2"];
  assertLayout(run(recipe, { strategy: "single_node_dep", hw: "ascend_950dt", nodes: 1 }), {
    tp: 4, global: 2, local: 2, nodes: 1,
  });
});

test("hybrid mode serves HTTP on every node and is removed when global DP is one", () => {
  const recipe = fixture();
  recipe.strategy_overrides.multi_node_tp_dp = { tp: 8, extra_args: ["--data-parallel-hybrid-lb"] };
  assertLayout(run(recipe), { tp: 8, global: 4, local: 2, hybrid: true });
  recipe.strategy_overrides.single_node_dep = { tp: 16, extra_args: ["--data-parallel-hybrid-lb"] };
  assertLayout(run(recipe, { strategy: "single_node_dep", nodes: 1 }), { tp: 16, global: 1, local: 1, nodes: 1 });
});

test("external DP requires one local rank per node and emits rank, not start-rank or headless", () => {
  for (const strategy of ["multi_node_tp_dp", "multi_node_dep"]) {
    const recipe = fixture();
    recipe.strategy_overrides[strategy] = { tp: 16, extra_args: ["--no-data-parallel-hybrid-lb", "--data-parallel-external-lb"] };
    assertLayout(run(recipe, { strategy }), { tp: 16, global: 2, local: 1, external: true });
    recipe.strategy_overrides[strategy].tp = 8;
    assertInvalid(run(recipe, { strategy }));
  }
  const recipe = fixture();
  recipe.strategy_overrides.single_node_dep = { tp: 16, extra_args: ["--data-parallel-external-lb"] };
  assertInvalid(run(recipe, { strategy: "single_node_dep", nodes: 1 }));
});

test("equals, short and underscore CLI spellings obey the same topology contract", () => {
  for (const args of [["--tensor-parallel-size=8"], ["--tensor_parallel_size", "8"], ["-tp", "8"]]) {
    const recipe = fixture();
    recipe.model.base_args = args;
    assertLayout(run(recipe), { tp: 8, global: 4, local: 2 });
  }
  const recipe = fixture();
  recipe.model.base_args = ["--tensor_parallel_size=8", "--data_parallel_size=2", "--data_parallel_size_local=1"];
  assertLayout(run(recipe), { tp: 8, global: 2, local: 1 });
});

test("invalid topology never emits runnable commands", () => {
  const cases = [
    ["zero TP", ["--tensor-parallel-size", "0"]],
    ["fractional TP", ["--tensor-parallel-size", "1.5"]],
    ["non-numeric TP", ["--tensor-parallel-size", "bad"]],
    ["non-divisor TP", ["--tensor-parallel-size", "3"]],
    ["oversized TP", ["--tensor-parallel-size", "32"]],
    ["PP", ["--pipeline-parallel-size", "2"]],
    ["zero local DP", ["--data-parallel-size-local", "0"]],
    ["fractional local DP", ["--data-parallel-size-local", "1.5"]],
    ["negative global DP", ["--data-parallel-size", "-2"]],
    ["fractional global DP", ["--data-parallel-size", "2.5"]],
    ["uneven global DP", ["--data-parallel-size", "3"]],
    ["over budget", ["--data-parallel-size-local", "2"]],
    ["inconsistent sizes", ["--data-parallel-size", "2", "--data-parallel-size-local", "2"]],
    ["negative worker start", ["--data-parallel-start-rank", "-1"]],
    ["wrong worker start", ["--data-parallel-start-rank", "8"]],
    ["rank without external mode", ["--data-parallel-rank", "1"]],
    ["underscore/equals invalid TP", ["--tensor_parallel_size=3"]],
    ["underscore/equals invalid local DP", ["--data_parallel_size_local=0"]],
    ["short invalid PP", ["-pp", "2"]],
  ];
  for (const [label, args] of cases) {
    const recipe = fixture();
    recipe.strategy_overrides.multi_node_tp_dp = { tp: 16, extra_args: args };
    try { assertInvalid(run(recipe)); } catch (error) { error.message = `${label}: ${error.message}`; throw error; }
  }
});

test("typed invalid TP is rejected instead of ignored or clamped", () => {
  for (const tp of [0, -1, 1.5, 3, 32, "8"]) {
    const recipe = fixture();
    recipe.strategy_overrides.multi_node_tp_dp = { tp };
    assertInvalid(run(recipe));
  }
});

test("every follower shares the head's replica layout, LB mode and rendezvous", () => {
  for (const worker_args of [
    ["--tensor-parallel-size", "4"], ["--data-parallel-size-local", "1"],
    ["--data-parallel-address", "another-head"], ["--data-parallel-rpc-port", "15555"],
    ["--data-parallel-hybrid-lb"], ["--no-headless"],
  ]) {
    const recipe = fixture();
    recipe.strategy_overrides.multi_node_tp_dp = { tp: 8,
      hardware_overrides: { ascend_910c: { worker_args } } };
    assertInvalid(run(recipe, { nodes: 4 }));
  }
  const recipe = fixture();
  recipe.strategy_overrides.multi_node_tp_dp = { tp: 8, hardware_overrides: {
    ascend_910c: { worker_args: ["--data-parallel-start-rank", "2"] },
  } };
  assertLayout(run(recipe), { tp: 8, global: 4, local: 2 });
  assertInvalid(run(recipe, { nodes: 4 })); // Rank 2 needs start=4, not start=2.
});

test("literal visible-device masks further constrain the final TP × local DP budget", () => {
  for (const [hw, key] of [["ascend_910c", "ASCEND_RT_VISIBLE_DEVICES"], ["h200", "CUDA_VISIBLE_DEVICES"], ["mi300x", "ROCR_VISIBLE_DEVICES"]]) {
    const recipe = fixture();
    recipe.strategy_overrides.single_node_dep = { tp: 2, extra_env: { [key]: "0,1,2,3" },
      extra_args: ["--data-parallel-size", "4"] };
    assertInvalid(run(recipe, { strategy: "single_node_dep", nodes: 1, hw }));
    recipe.strategy_overrides.single_node_dep.extra_args = ["--data-parallel-size", "2"];
    assertLayout(run(recipe, { strategy: "single_node_dep", nodes: 1, hw }), { tp: 2, global: 2, local: 2, nodes: 1 });
    recipe.strategy_overrides.single_node_dep.extra_env[key] = "0,1,1,2";
    assertInvalid(run(recipe, { strategy: "single_node_dep", nodes: 1, hw }));
  }
});

test("LB conflicts and unavailable HTTP entry points are rejected", () => {
  for (const args of [
    ["--data-parallel-hybrid-lb", "--data-parallel-external-lb"],
    ["--headless"], ["--api_server_count=0"], ["-asc1"],
    ["--data-parallel-multi-port-external-lb"], ["--data-parallel-backend", "ray"],
    ["--prefill-context-parallel-size", "2"], ["--nnodes", "2"],
    ["--data-parallel-size-l", "1"], ["--tensor-parallel-size"],
    ["--data-parallel-hybrid-lb", "false"], ["--no-data-parallel-hybrid-lb", "false"],
    ["--data-parallel-address", "-bad"], ["--data-parallel-address="],
    ...["abc", "0", "-1", "65536"].map((port) => ["--data-parallel-rpc-port", port]),
  ]) {
    const recipe = fixture();
    recipe.model.base_args = args;
    assertInvalid(run(recipe));
  }
  const recipe = fixture();
  recipe.model.architecture = "dense";
  recipe.model.base_args = ["--data-parallel-external-lb"];
  assertInvalid(run(recipe));
  recipe.model.base_args = [];
  assertLayout(run(recipe), { tp: 16, global: 2, local: 1 });
});

test("effective integer values and default rendezvous are identical across node commands", () => {
  const recipe = fixture();
  recipe.strategy_overrides.multi_node_tp_dp = { hardware_overrides: {
    ascend_910c: { worker_args: ["--data-parallel-rpc-port=29550", "--data-parallel-backend", "mp"] },
  } };
  assertLayout(run(recipe), { tp: 16, global: 2, local: 1 });
  recipe.strategy_overrides.single_node_dep = { extra_args: ["--api-server-count", "1.0", "--data-parallel-rpc-port", "15555.0"] };
  const argv = run(recipe, { strategy: "single_node_dep", nodes: 1 }).argv;
  assert.equal(flag(argv, "--api-server-count"), "1");
  assert.equal(flag(argv, "--data-parallel-rpc-port"), "15555");
});

test("preset TP and later preset raw args size public JSON, Docker and Markdown consistently", () => {
  const recipe = fixture();
  recipe.strategy_overrides.multi_node_tp_dp = { tp: 16 };
  recipe.presets = { tuned: { label: "TP4", hardware: "ascend_910c", strategy: "multi_node_tp_dp",
    tp: 8, args: ["--tensor-parallel-size", "4"], select: { nodes: 2 } } };
  assertLayout(run(recipe, { preset: "tuned" }), { tp: 4, global: 8, local: 4 });
  delete recipe.presets.tuned.args;
  assertLayout(run(recipe, { preset: "tuned" }), { tp: 8, global: 4, local: 2 });
  const resolved = resolveSelection(recipe, new URLSearchParams({ hardware: "ascend_910c", strategy: "multi_node_tp_dp", nodes: "2", preset: "tuned" }), strategies, taxonomy);
  const rendering = JSON.parse(JSON.stringify(resolved.rendering));
  assert.ok(rendering);
  for (const argv of [rendering.head_argv, ...rendering.worker_argvs, rendering.head_docker_argv, ...rendering.worker_docker_argvs]) {
    assert.equal(flag(argv, "--tensor-parallel-size"), "8");
    assert.equal(flag(argv, "--data-parallel-size"), "4");
    assert.equal(flag(argv, "--data-parallel-size-local"), "2");
  }
  assert.equal(flag(rendering.worker_argvs[0], "--data-parallel-start-rank"), "2");
  const markdown = selectionMarkdown(recipe, resolved, { strategies, taxonomy, urls });
  assert.match(markdown, /--tensor-parallel-size 8/);
  assert.match(markdown, /--data-parallel-size-local 2/);
  assert.match(markdown, /--data-parallel-start-rank 2/);
});

test("invalid layouts propagate through static rendering, dynamic selection and Markdown", () => {
  const recipe = fixture();
  recipe.strategy_overrides.multi_node_tp_dp = { tp: 3 };
  assert.throws(() => renderCommand(recipe, "default", "multi_node_tp_dp", "ascend_910c", 2, [], strategies, taxonomy,
    null, null, undefined, undefined, true), /Invalid deployment/);
  const resolved = resolveSelection(recipe, new URLSearchParams({ hardware: "ascend_910c", strategy: "multi_node_tp_dp", nodes: "2" }), strategies, taxonomy);
  assert.equal(resolved.rendering, null);
  assert.equal(resolved.selection.error?.code, "DP_INVALID_LAYOUT");
  const markdown = selectionMarkdown(recipe, resolved, { strategies, taxonomy, urls });
  assert.ok(markdown.includes(resolved.selection.error.message));
  assert.doesNotMatch(markdown, /vllm serve test\/DP/);
});

test("Mooncake composition cannot hide an invalid serving topology", () => {
  const recipe = fixture();
  recipe.strategy_overrides.multi_node_tp_dp = { tp: 3 };
  assertInvalid(run(recipe, { hw: "h200", kv: "kv_store_distributed_mooncake" }));
});

test("DCP choices are computed from the final effective TP", () => {
  const recipe = fixture();
  recipe.decode_context_parallel = { sizes: { h200: [2, 4, 8] }, comm_backend: "a2a", match_tp: true };
  recipe.strategy_overrides.multi_node_tp_dp = { tp: 4 };
  const on = run(recipe, { hw: "h200", dcp: 4 });
  assert.deepEqual(on.dcp.sizes, [4]);
  assert.deepEqual(on.dcp.tp, [4]);
  assert.equal(on.dcp.selected, 4);
  for (const argv of nodeArgvs(on)) assert.equal(flag(argv, "--decode-context-parallel-size"), "4");
  const changed = run(recipe, { hw: "h200", dcp: 4, advanced: ["--tensor-parallel-size", "2"] });
  assert.deepEqual(changed.dcp.sizes, [2]);
  assert.equal(changed.dcp.selected, 0);
  for (const argv of nodeArgvs(changed)) assert.equal(flag(argv, "--decode-context-parallel-size"), undefined);
});
