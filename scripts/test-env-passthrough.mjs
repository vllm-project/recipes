import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { test } from "node:test";
import { buildDockerArgv, buildDockerRun, resolveCommand } from "../src/lib/command-synthesis.js";
import { envToExports } from "../src/lib/env-format.js";
import { loadTaxonomy } from "../src/lib/taxonomy.js";
import { loadStrategies } from "../src/lib/strategies.js";

test("self-references pass through in Docker shell and argv forms", () => {
  const env = { NCCL_IB_HCA: "$NCCL_IB_HCA", NCCL_SOCKET_IFNAME: "$IFACE_NAME", FIXED: "1" };
  const meta = { image: "example/image:tag", gpuFlags: "--gpus all" };
  const shell = buildDockerRun({ command: "vllm serve org/model", env, ...meta });
  const argv = buildDockerArgv({ argv: ["vllm", "serve", "org/model"], env, meta });
  assert.match(shell, /-e NCCL_IB_HCA(?:\s|\\)/);
  assert.doesNotMatch(shell, /NCCL_IB_HCA=/);
  assert.match(shell, /-e NCCL_SOCKET_IFNAME=\$IFACE_NAME/);
  assert.deepEqual(argv.slice(argv.indexOf("NCCL_IB_HCA") - 1, argv.indexOf("NCCL_IB_HCA") + 1), ["-e", "NCCL_IB_HCA"]);
  assert.ok(argv.includes("NCCL_SOCKET_IFNAME=$IFACE_NAME"));
});

test("bare commands preserve unset, set and empty host variables", () => {
  const prelude = envToExports({ NCCL_IB_HCA: "${NCCL_IB_HCA}" });
  const script = `${prelude}\nsh -c 'if [ "\${NCCL_IB_HCA+x}" = x ]; then printf "set:%s" "$NCCL_IB_HCA"; else printf unset; fi'`;
  for (const [setup, expected] of [["", "unset"], ["NCCL_IB_HCA=fic2,vsolar\n", "set:fic2,vsolar"], ["NCCL_IB_HCA=''\n", "set:"]]) {
    const result = spawnSync("sh", ["-eu", "-c", setup + script], { encoding: "utf8", env: { PATH: process.env.PATH } });
    assert.equal(result.status, 0, result.stderr);
    assert.equal(result.stdout, expected);
  }
});

test("810E multi-node roles pass through HCA without overriding explicit values", () => {
  const recipe = {
    model: { model_id: "org/model" },
    variants: { default: { precision: "bf16" } },
  };
  const taxonomy = loadTaxonomy();
  const strategies = loadStrategies();
  const get = (hw, strategy, nodes, pdNodes = null, configured = undefined) => {
    const model = configured ? { ...recipe, model: { ...recipe.model, base_env: { NCCL_IB_HCA: configured } } } : recipe;
    return resolveCommand(model, "default", strategy, hw, [], strategies, taxonomy, [], nodes, pdNodes);
  };
  const multi = get("810e", "multi_node_tep", 2);
  assert.equal(multi.env.NCCL_IB_HCA, "$NCCL_IB_HCA");
  assert.equal(multi.env.PCCL_NET_PLUGIN_ENABLE, "1");
  assert.equal(get("810e", "multi_node_tep", 2, null, "custom").env.NCCL_IB_HCA, "custom");
  assert.equal(get("810e", "multi_node_tep", 1).env.NCCL_IB_HCA, undefined);
  assert.equal(get("810e", "single_node_tp", 1).env.NCCL_IB_HCA, undefined);
  assert.equal(get("810e", "single_node_tp", 1).env.PCCL_NET_PLUGIN_ENABLE, undefined);
  assert.equal(get("m890", "multi_node_tep", 2).env.NCCL_IB_HCA, undefined);
  assert.equal(get("m890", "multi_node_tep", 2).env.PCCL_NET_PLUGIN_ENABLE, undefined);
  const pd = get("810e", "pd_cluster", 3, { prefill: { nodes: 2 }, decode: { nodes: 1 } });
  assert.equal(pd.prefill.env.NCCL_IB_HCA, "$NCCL_IB_HCA");
  assert.equal(pd.prefill.env.PCCL_NET_PLUGIN_ENABLE, "1");
  assert.equal(pd.decode.env.NCCL_IB_HCA, undefined);
  assert.equal(pd.decode.env.PCCL_NET_PLUGIN_ENABLE, undefined);
  const explicit = { ...recipe, model: { ...recipe.model, base_env: { PCCL_NET_PLUGIN_ENABLE: "0" } } };
  assert.equal(resolveCommand(explicit, "default", "multi_node_tep", "810e", [], strategies, taxonomy, [], 2).env.PCCL_NET_PLUGIN_ENABLE, "0");
});
