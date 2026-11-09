import assert from "node:assert/strict";
import fs from "node:fs";
import { test } from "node:test";
import yaml from "js-yaml";
import { resolveCommand } from "../src/lib/command-synthesis.js";
import { resolveSelection } from "../src/lib/recipe-render.js";
import { selectionMarkdown } from "../src/lib/recipe-markdown.js";

const read = (file) => yaml.load(fs.readFileSync(new URL(`../${file}`, import.meta.url), "utf8"));
const taxonomy = read("taxonomy.yaml");
const strategies = Object.fromEntries(["multi_node_tp_dp", "multi_node_dep"].map((name) => [name, read(`strategies/${name}.yaml`)]));
const flag = (args, key) => {
  const index = args.lastIndexOf(key);
  return index < 0 ? undefined : args[index + 1];
};
const fixture = () => ({
  hf_id: "test/Local-DP", meta: { title: "Local DP" },
  model: { model_id: "test/Local-DP", architecture: "moe", base_args: [] },
  variants: { default: { precision: "bf16", vram_minimum_gb: 4 } },
  compatible_strategies: Object.keys(strategies),
});
const render = (recipe, strategy, hardware = "h200", preset, point) => resolveCommand(
  recipe, "default", strategy, hardware, [], strategies, taxonomy,
  [], 4, null, {}, null, null, "python", null, "http", null, preset, point,
);
function check(result, { tp, local, global = local * 4 }) {
  const commands = [result.headArgv, ...result.workerArgvs];
  assert.equal(commands.length, 4);
  for (const [rank, args] of commands.entries()) {
    assert.equal(Number(flag(args, "--tensor-parallel-size") || 1), tp);
    assert.equal(Number(flag(args, "--data-parallel-size-local")), local);
    assert.equal(Number(flag(args, "--data-parallel-size")), global);
    if (rank) assert.equal(Number(flag(args, "--data-parallel-start-rank")), rank * local);
    assert.equal(flag(args, "--data-parallel-address"), "$HEAD_IP");
  }
}

test("ordinary multi-node DP defaults follow the final TP on GPU and NPU", () => {
  for (const strategy of Object.keys(strategies)) for (const hardware of ["h200", "ascend_950dt"]) {
    check(render(fixture(), strategy, hardware), { tp: strategy === "multi_node_tp_dp" ? 8 : 1, local: strategy === "multi_node_tp_dp" ? 1 : 8 });
    const recipe = fixture();
    recipe.strategy_overrides = { [strategy]: { extra_args: ["--tensor-parallel-size", "4"] } };
    check(render(recipe, strategy, hardware), { tp: 4, local: 2 });
    recipe.strategy_overrides[strategy] = { tp: { [hardware]: 4 } };
    check(render(recipe, strategy, hardware), { tp: 4, local: 2 });
  }
});

test("late hardware and preset arguments keep local, global and follower DP ranks consistent", () => {
  for (const strategy of Object.keys(strategies)) {
    const recipe = fixture();
    recipe.strategy_overrides = { [strategy]: { hardware_overrides: { h200: { extra_args: ["--tensor-parallel-size", "4", "--data-parallel-size-local", "2"] } } } };
    check(render(recipe, strategy), { tp: 4, local: 2 });
    recipe.presets = { small: { label: "Partial node", hardware: "h200", strategy,
      args: ["--data-parallel-size-local", "1"],
      points: { tp2: { args: ["--tensor-parallel-size", "2"] } },
    } };
    check(render(recipe, strategy, "h200", "small", "tp2"), { tp: 2, local: 1 });
    delete recipe.strategy_overrides;
    recipe.presets.small.args = ["--tensor-parallel-size", "4", "--data-parallel-size", "4"];
    delete recipe.presets.small.points;
    check(render(recipe, strategy, "h200", "small"), { tp: 4, local: 1, global: 4 });
    recipe.presets.small.args = ["--tensor-parallel-size", "8", "-tp=4", "-dpl", "2"];
    check(render(recipe, strategy, "h200", "small"), { tp: 4, local: 2 });
    recipe.presets.small.args = ["-tp", "2", "--tensor-parallel-size=4", "-dpl=1", "-dp=4"];
    check(render(recipe, strategy, "h200", "small"), { tp: 4, local: 1, global: 4 });
  }
});

test("local DP layout reaches page-query JSON, Docker and Markdown exports", () => {
  for (const strategy of Object.keys(strategies)) {
    const recipe = fixture();
    recipe.strategy_min_gpus = 32;
    recipe.strategy_overrides = { [strategy]: { extra_args: ["--tensor-parallel-size", "4", "--data-parallel-size-local", "2"] } };
    const resolved = resolveSelection(recipe, new URLSearchParams({ hardware: "h200", strategy, nodes: "4" }), strategies, taxonomy);
    const output = resolved.rendering;
    assert.equal(Number(flag(output.head_argv, "--data-parallel-size")), 8);
    assert.equal(output.worker_argvs.length, 3);
    for (const [index, args] of output.worker_docker_argvs.entries()) {
      assert.equal(Number(flag(args, "--data-parallel-size-local")), 2);
      assert.equal(Number(flag(args, "--data-parallel-start-rank")), (index + 1) * 2);
    }
    const markdown = selectionMarkdown(recipe, resolved, { strategies, taxonomy,
      urls: { page: "https://example.test/test/Local-DP", md: "https://example.test/test/Local-DP.md", json: "https://example.test/test/Local-DP.json" } });
    assert.match(markdown, /--data-parallel-size-local 2/);
    assert.match(markdown, /--data-parallel-start-rank 6/);
  }
});
