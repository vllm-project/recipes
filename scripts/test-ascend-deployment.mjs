import assert from "node:assert/strict";
import { execFileSync, spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { test } from "node:test";
import yaml from "js-yaml";
import Ajv from "ajv";
import { createAscendPdRecipe, createCustomConnectorRecipe, ascendPdCases } from "./fixtures/ascend-deployment.mjs";
import { buildRecipeSchema } from "../src/lib/recipe-schema.js";
import { resolvePdRoleLayout } from "../src/lib/command-synthesis.js";
import { bindRouterPort } from "../src/lib/cluster-endpoints.js";
import { renderCommand, resolveSelection } from "../src/lib/recipe-render.js";
import { selectionMarkdown } from "../src/lib/recipe-markdown.js";

const root = fileURLToPath(new URL("../", import.meta.url));
const read = (file) => yaml.load(fs.readFileSync(path.join(root, file), "utf8"));
const taxonomy = read("taxonomy.yaml");
const strategies = Object.fromEntries(["strategies", "kv_store"].flatMap((dir) => fs.readdirSync(path.join(root, dir))
  .filter((file) => file.endsWith(".yaml")).map((file) => { const value = read(`${dir}/${file}`); return [value.name, value]; })));
const flag = (argv, key) => argv[argv.lastIndexOf(key) + 1];
const select = (recipe, params) => resolveSelection(recipe, new URLSearchParams(params), strategies, taxonomy);
const pageQuery = { hardware: "ascend_950dt", strategy: "pd_cluster", prefill_nodes: "5", decode_nodes: "5", prefill_rank: "7", decode_rank: "999", prefill_mode: "dep", decode_mode: "tp" };
const urls = { page: "https://example.test/test/Ascend-PD", md: "https://example.test/test/Ascend-PD.md", json: "https://example.test/test/Ascend-PD.json" };

test("shared layouts and selection round five nodes to eight, with group-local rank and master", () => {
  const recipe = createAscendPdRecipe();
  recipe.hf_id = "test/Ascend-PD";
  const resolved = select(recipe, pageQuery);
  assert.equal(resolved.selection.variant, "default");
  for (const role of ["prefill", "decode"]) {
    const layout = resolvePdRoleLayout({ recipe, strategy: strategies.pd_cluster, role,
      hwProfile: taxonomy.hardware_profiles.ascend_950dt, hwProfileId: "ascend_950dt",
      selection: { nodes: 5, rank: role === "prefill" ? 7 : 999, parallelism: role === "prefill" ? "dep" : "tp" } });
    assert.equal(layout.nodes, 8);
    assert.equal(layout.nodeStep, 4);
    assert.equal(layout.currentNode, 7);
    assert.equal(layout.instanceNodeRank, 3);
    assert.equal(layout.instanceStart, 5);
    assert.deepEqual(resolved.selection.pd[role], { nodes: layout.nodes, rank: layout.currentNode, parallelism: layout.parallelism });
    assert.equal(resolved.rendering[role].nodes, 8);
    const argv = resolved.rendering[role].argv;
    assert.equal(flag(argv, role === "prefill" ? "--node-rank" : "--data-parallel-rank"), "3");
    assert.equal(flag(argv, role === "prefill" ? "--master-addr" : "--data-parallel-address"), `$${role.toUpperCase()}_NODE_5`);
  }
  for (const param of ["prefill_nodes", "decode_nodes", "prefill_mode", "decode_mode", "decode_rank"]) {
    assert.ok(resolved.selection.adjusted.some((a) => a.param === param), param);
  }
  const markdown = selectionMarkdown(recipe, resolved, { strategies, taxonomy, urls });
  assert.match(markdown, /--dp-rank-start 3/);
  assert.match(markdown, /\$PREFILL_NODE_5/);
});

test("ordinary GPU colocated pools retain zero-node half-GPU semantics", () => {
  const recipe = createAscendPdRecipe();
  const layout = resolvePdRoleLayout({ recipe, strategy: strategies.pd_cluster, role: "prefill",
    hwProfile: taxonomy.hardware_profiles.h200, hwProfileId: "h200", selection: { nodes: 0, rank: 7, parallelism: "tp" } });
  assert.equal(layout.nodes, 0);
  assert.equal(layout.poolGpus, 4);
  assert.equal(layout.tp, 4);
  assert.equal(layout.currentNode, 0);
  assert.equal(layout.nodeStep, 1);
});

test("Ascend proxy port binding agrees in command, argv, Docker and public renderings", () => {
  const recipe = createAscendPdRecipe();
  const resolved = select(recipe, pageQuery);
  const bound = bindRouterPort({ ...resolved.rendering.router,
    command: "python proxy --port $ROUTER_PORT", argv: ["python", "proxy", "--port", "$ROUTER_PORT"], dockerCommand: "docker run image --port $ROUTER_PORT" }, "18000");
  assert.equal(bound.port, 18000);
  assert.match(bound.command, /--port 18000/);
  assert.equal(flag(bound.argv, "--port"), "18000");
  assert.match(bound.dockerCommand, /--port 18000/);
  const render = (routerPort) => renderCommand(recipe, "default", "pd_cluster", "ascend_950dt", 1, [], strategies, taxonomy,
    null, null, undefined, undefined, true, { routerPort });
  for (const [requested, expected] of [[undefined, 8000], [18000, 18000]]) {
    const router = render(requested).router;
    assert.equal(router.port, expected);
    assert.equal(flag(router.argv, "--port"), String(expected));
    for (const text of [router.command, router.dockerCommand]) assert.match(text, new RegExp(`--port ${expected}\\b`));
  }
  for (const port of [0, -1, 65536, 1.2, "invalid", Infinity]) assert.throws(() => bindRouterPort(bound, port), /1 to 65535/);
});

test("local DP exports every external process and keeps Markdown launcher steps", () => {
  const recipe = createAscendPdRecipe();
  recipe.hf_id = "test/Ascend-PD";
  recipe.strategy_overrides.pd_cluster.decode.tp.ascend_950dt = 4;
  const resolved = select(recipe, pageQuery);
  const { prefill, decode, router, start_server } = resolved.rendering;
  assert.ok(prefill.command);
  assert.equal(decode.command, undefined);
  assert.equal(decode.argv, undefined);
  assert.equal(decode.docker_command, undefined);
  assert.equal(decode.processes.length, 2);
  assert.equal(decode.dpLocal, 2);
  assert.equal(decode.dpSize, 8);
  for (const [i, process] of decode.processes.entries()) {
    const devices = i === 0 ? "0,1,2,3" : "4,5,6,7";
    assert.equal(process.localRank, i);
    assert.equal(process.dpRank, 6 + i);
    assert.equal(process.port, 8002 + i);
    assert.equal(process.env.ASCEND_RT_VISIBLE_DEVICES, devices);
    assert.equal(flag(process.argv, "--data-parallel-size-local"), "1");
    assert.equal(flag(process.argv, "--data-parallel-rank"), String(6 + i));
    assert.equal(flag(process.argv, "--data-parallel-address"), "$DECODE_NODE_5");
    assert.equal(flag(process.argv, "--port"), String(8002 + i));
    assert.equal(flag(process.docker_argv, "--data-parallel-rank"), String(6 + i));
    assert.equal(flag(process.docker_argv, "--port"), String(8002 + i));
    assert.ok(process.docker_argv.includes(`ASCEND_RT_VISIBLE_DEVICES=${devices}`));
    assert.match(process.docker_command, new RegExp(`ASCEND_RT_VISIBLE_DEVICES=${devices}`));
    for (const argv of [process.argv, process.docker_argv]) {
      assert.equal(JSON.parse(flag(argv, "--speculative-config")).draft_tensor_parallel_size, 4);
      const topology = JSON.parse(flag(argv, "--kv-transfer-config")).kv_connector_extra_config;
      assert.equal(topology.prefill.tp_size, 8);
      assert.equal(topology.decode.tp_size, 4);
    }
  }
  assert.match(start_server.command, /--dp-size 8 --tp-size 4 --dp-size-local 2/);
  assert.match(start_server.command, /--dp-rank-start 6/);
  const markdown = selectionMarkdown(recipe, resolved, { strategies, taxonomy, urls });
  assert.match(markdown, /--dp-size 8 --tp-size 4 --dp-size-local 2/);
  assert.match(markdown, /--decoder-ports (8002 8003 ){7}8002 8003/);
  assert.match(router.command, /--decoder-hosts \$DECODE_NODE_1 \$DECODE_NODE_1/);
  assert.deepEqual(JSON.parse(JSON.stringify(resolved.rendering)).decode.processes, decode.processes);
});

test("local DP1 keeps singular public commands identical to its sole process", () => {
  const recipe = createAscendPdRecipe();
  const resolved = select(recipe, pageQuery);
  const decode = resolved.rendering.decode;
  assert.equal(decode.dpLocal, 1);
  assert.equal(decode.processes.length, 1);
  for (const key of ["command", "argv", "env", "docker_command", "docker_argv"]) {
    assert.deepEqual(decode[key], decode.processes[0][key], key);
  }
  assert.equal(flag(decode.argv, "--data-parallel-size-local"), "1");
  assert.equal(flag(decode.argv, "--data-parallel-rank"), "3");
  assert.equal(decode.env.ASCEND_RT_VISIBLE_DEVICES, "0,1,2,3,4,5,6,7");
});

test("local DP4 uses all device slices and accepts the last legal HTTP port", () => {
  const recipe = createAscendPdRecipe();
  const cfg = recipe.strategy_overrides.pd_cluster.decode;
  cfg.tp.ascend_950dt = 2;
  cfg.hardware_overrides.ascend_950dt.extra_args.push("--port", "65532");
  const { decode, router, start_server } = select(recipe, pageQuery).rendering;
  assert.equal(decode.dpLocal, 4);
  assert.equal(decode.dpSize, 16);
  assert.deepEqual(decode.processes.map((process) => process.dpRank), [12, 13, 14, 15]);
  assert.deepEqual(decode.processes.map((process) => process.port), [65532, 65533, 65534, 65535]);
  assert.deepEqual(decode.processes.map((process) => process.env.ASCEND_RT_VISIBLE_DEVICES), ["0,1", "2,3", "4,5", "6,7"]);
  for (const process of decode.processes) {
    for (const argv of [process.argv, process.docker_argv]) {
      assert.equal(flag(argv, "--port"), String(process.port));
      assert.equal(JSON.parse(flag(argv, "--speculative-config")).draft_tensor_parallel_size, 2);
    }
  }
  assert.match(start_server.command, /--dp-size 16 --tp-size 2 --dp-size-local 4/);
  assert.match(start_server.command, /--dp-rank-start 12/);
  assert.match(router.command, /--decoder-ports (65532 65533 65534 65535 ){7}65532 65533 65534 65535/);
});

test("invalid layouts fail static rendering and return actionable selection errors", () => {
  const recipe = createAscendPdRecipe();
  recipe.hf_id = "test/Ascend-PD";
  const cfg = recipe.strategy_overrides.pd_cluster.decode;
  cfg.tp.ascend_950dt = 4;
  cfg.hardware_overrides.ascend_950dt.extra_args.push("--pipeline-parallel-size", "2");
  for (const strict of [false, true]) {
    assert.throws(() => renderCommand(recipe, "default", "pd_cluster", "ascend_950dt", 1,
      ["spec_decoding"], strategies, taxonomy, null, null, undefined, undefined, strict), /PP|pipeline/i);
  }
  const resolved = select(recipe, pageQuery);
  assert.equal(resolved.rendering, null);
  assert.equal(resolved.selection.error.code, "ASCEND_PD_INVALID_LAYOUT");
  assert.match(resolved.selection.error.message, /decode/i);
  assert.match(resolved.selection.error.message, /PP|pipeline/i);
  const markdown = selectionMarkdown(recipe, resolved, { strategies, taxonomy, urls });
  assert.ok(markdown.includes(resolved.selection.error.message));
  assert.doesNotMatch(markdown, /launch_online_dp\.py|run_dp_template\.sh|```bash\nvllm serve/);
});

test("distinct dense and MoE PD layouts validate and export their selected checkpoint", () => {
  const validate = new Ajv({ allErrors: true, strict: false }).compile(buildRecipeSchema({ taxonomy, strategies }));
  for (const { recipe, hardware, variant } of ascendPdCases()) {
    assert.ok(validate(recipe), JSON.stringify(validate.errors));
    recipe.hf_id = recipe.model.model_id;
    const resolved = select(recipe, { hardware, variant, strategy: "pd_cluster" });
    assert.equal(resolved.selection.variant, variant);
    assert.equal(resolved.selection.hardware, hardware);
    const rendering = JSON.parse(JSON.stringify(resolved.rendering));
    assert.ok(rendering.ascend_scripts && rendering.start_server);
    const modelId = recipe.variants[variant].model_id || recipe.model.model_id;
    for (const role of ["prefill", "decode"]) {
      const pool = rendering[role];
      for (const process of pool.processes || [pool]) {
        assert.equal(process.argv[process.argv.indexOf("serve") + 1], modelId);
        assert.ok(process.docker_argv.includes(modelId));
      }
      assert.match(pool.templateCommand, new RegExp(modelId));
    }
    const markdown = selectionMarkdown(recipe, resolved, { strategies, taxonomy, urls });
    assert.ok(markdown.includes(modelId));
    for (const [index, title] of ["Scripts", "Prefill", "Decode", "Start_Server", "Start_Proxy"].entries()) {
      assert.ok(markdown.includes(`### ${index + 1}. ${title}`));
    }
  }
});

test("custom recipe connectors export truthful metadata and preserve Docker/Markdown arguments", () => {
  const recipe = createCustomConnectorRecipe();
  const validate = new Ajv({ allErrors: true, strict: false }).compile(buildRecipeSchema({ taxonomy, strategies }));
  assert.ok(validate(recipe), JSON.stringify(validate.errors));
  assert.deepEqual(recipe.presets.ascend.select, { pd_router: "ascend" });
  recipe.hf_id = recipe.model.model_id;
  const resolved = select(recipe, { hardware: "ascend_950dt", strategy: "pd_cluster" });
  assert.equal(resolved.selection.error, undefined);
  const rendering = JSON.parse(JSON.stringify(resolved.rendering));
  assert.equal(rendering.pd_connector, null);
  assert.deepEqual(rendering.pd_connector_configured, { prefill: "TestConnector", decode: "TestDecodeConnector" });
  assert.ok(rendering.ascend_scripts && rendering.start_server);
  assert.equal(rendering.decode.processes.length, 2);
  for (const role of ["prefill", "decode"]) {
    const expected = JSON.parse(flag(role === "prefill" ? recipe.model.base_args
      : recipe.strategy_overrides.pd_cluster.decode.hardware_overrides.ascend_950dt.extra_args, "--kv-transfer-config"));
    for (const process of rendering[role].processes || [rendering[role]]) {
      for (const argv of [process.argv, process.docker_argv]) {
        assert.deepEqual(JSON.parse(flag(argv, "--kv-transfer-config")), expected);
      }
      assert.equal(process.env.TEST_TRANSFER_ROLE, role);
      assert.ok(process.docker_argv.includes(`TEST_TRANSFER_ROLE=${role}`));
      assert.match(process.docker_command, new RegExp(expected.kv_connector));
    }
  }
  const markdown = selectionMarkdown(recipe, resolved, { strategies, taxonomy, urls });
  for (const name of ["TestConnector", "TestDecodeConnector"]) assert.ok(markdown.includes(name));
  assert.match(markdown, /--dp-size 8 --tp-size 4 --dp-size-local 2/);
  assert.doesNotMatch(markdown, /MooncakeConnectorV2|VLLM_NIXL_SIDE_CHANNEL/);
  const disabled = select(recipe, { hardware: "ascend_950dt", strategy: "pd_cluster", pd_router: "vllm" });
  assert.equal(disabled.rendering?.ascend_scripts, undefined, "an explicit URL router overrides the preset");
});

test("the real static API builder exports custom grouped PD and rejects an invalid authored layout", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "recipes-ascend-api-"));
  try {
    for (const file of ["taxonomy.yaml", "platforms.yaml", "strategies", "kv_store"]) fs.cpSync(path.join(root, file), path.join(dir, file), { recursive: true });
    fs.mkdirSync(path.join(dir, "models", "test"), { recursive: true });
    const recipe = createCustomConnectorRecipe();
    const recipeFile = path.join(dir, "models/test/Custom-PD.yaml");
    fs.writeFileSync(recipeFile, yaml.dump(recipe));
    const out = path.join(dir, "api");
    const env = { ...process.env, RECIPES_CONTENT_DIR: dir, RECIPES_PUBLIC_DIR: out };
    execFileSync(process.execPath, ["scripts/build-recipes-api.mjs"], { cwd: root, env, stdio: "pipe", timeout: 60000 });
    const exported = JSON.parse(fs.readFileSync(path.join(out, "test/Custom-PD/hw/ascend_950dt.json"), "utf8"));
    assert.equal(exported.deploy_type, "pd_cluster");
    assert.ok(exported.ascend_scripts && exported.start_server);
    assert.equal(exported.decode.processes.length, 2);
    assert.equal(exported.pd_connector, null);
    assert.deepEqual(exported.pd_connector_configured, { prefill: "TestConnector", decode: "TestDecodeConnector" });
    recipe.strategy_overrides.pd_cluster.decode.pp.ascend_950dt = 2;
    fs.writeFileSync(recipeFile, yaml.dump(recipe));
    const invalid = spawnSync(process.execPath, ["scripts/build-recipes-api.mjs"], { cwd: root, env, encoding: "utf8", timeout: 60000 });
    assert.notEqual(invalid.status, 0, "invalid PD topology must fail the build");
    assert.match(invalid.stdout + invalid.stderr, /decode/i);
    assert.match(invalid.stdout + invalid.stderr, /PP|pipeline/i);
  } finally {
    assert.equal(path.dirname(dir), path.resolve(os.tmpdir()));
    assert.ok(path.basename(dir).startsWith("recipes-ascend-api-"));
    fs.rmSync(dir, { recursive: true, force: true });
  }
});
