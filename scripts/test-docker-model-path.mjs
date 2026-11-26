import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import path from "node:path";
import { test } from "node:test";
import { substitute } from "../src/lib/cluster-endpoints.js";
import { buildDockerRun, computeDockerMeta, resolveOmniCommand } from "../src/lib/command-synthesis.js";
import { renderCommand } from "../src/lib/recipe-render.js";
import { loadStrategies } from "../src/lib/strategies.js";
import { loadTaxonomy } from "../src/lib/taxonomy.js";

const taxonomy = loadTaxonomy();
const profiles = taxonomy.hardware_profiles;
const strategies = loadStrategies();

test("local checkpoint paths reach the server and its Docker mount as single literal arguments", (t) => {
  const bash = process.platform === "win32" ? path.join(process.env.ProgramFiles || "C:/Program Files", "Git/bin/bash.exe") : "bash";
  if (spawnSync(bash, ["--version"]).status !== 0) return t.skip("Bash is unavailable");
  const checkpoint = "/weights/O'Brien $ROUTER_HOST [v2] --port 9123 $(echo model)\nsecond line";
  const r = {
    model: { model_id: checkpoint, base_args: ["--dtype", "auto"] },
    variants: { default: { precision: "fp8", vram_minimum_gb: 100 } },
    compatible_strategies: ["single_node_tp", "multi_node_tp", "pd_cluster"],
  };
  const run = (name, command) => {
    const script = name + '() { printf "%s\\0" "$@"; }\n' + substitute(command, { ROUTER_HOST: "localhost" });
    const { status, stdout, stderr } = spawnSync(bash, ["-c", script], {
      encoding: "utf8", env: { ...process.env, HEAD_IP: "$HEAD_IP" },
    });
    assert.equal(status, 0, stderr);
    return stdout.split("\0").slice(0, -1);
  };
  const meta = computeDockerMeta(r, r.variants.default, profiles.h200, "h200");
  function check(command, argv, dockerCommand, dockerArgv, port = 8000) {
    assert.deepEqual(run("vllm", command), argv.slice(1));
    const args = run("docker", dockerCommand);
    const image = args.indexOf(meta.image);
    assert.ok(image >= 0);
    assert.deepEqual(args.slice(image), dockerArgv.slice(dockerArgv.indexOf(meta.image)));
    const mount = `${checkpoint}:${checkpoint}:ro`;
    assert.ok(args.includes(mount));
    assert.ok(dockerArgv.includes(mount));
    assert.ok(args.includes(`${port}:${port}`), "a flag-like substring in the path must not publish a port");
  }
  const single = renderCommand(r, "default", "single_node_tp", "h200", 1, [], strategies, taxonomy);
  check(single.command, single.argv, single.docker_command, single.docker_argv);
  const multi = renderCommand(r, "default", "multi_node_tp", "h200", 2, [], strategies, taxonomy);
  check(multi.head_command, multi.head_argv, multi.head_docker_command, multi.head_docker_argv);
  check(multi.worker_command, multi.worker_argv, multi.worker_docker_command, multi.worker_docker_argv);
  const pd = renderCommand(r, "default", "pd_cluster", "h200", 2, [], strategies, taxonomy,
    { prefill: { nodes: 1 }, decode: { nodes: 1 } });
  for (const [pool, port] of [[pd.prefill, 8001], [pd.decode, 8002]]) {
    check(pool.command, pool.argv, pool.docker_command, pool.docker_argv, port);
  }
  const kv = renderCommand(r, "default", "multi_node_tp", "h200", 2, [],
    strategies, taxonomy, null, "kv_store_distributed_mooncake");
  check(kv.vllm_command, kv.vllm_argv, kv.vllm_docker_command, kv.vllm_docker_argv);
  check(kv.vllm_worker_command, kv.vllm_worker_argv, kv.vllm_worker_docker_command, kv.vllm_worker_docker_argv);

  const omni = resolveOmniCommand(r, "default", { modelId: checkpoint }, profiles.h200, "h200");
  const cmd = buildDockerRun({ command: omni.command, modelId: omni.modelId, ...meta });
  const args = run("docker", cmd);
  assert.deepEqual(args.slice(args.indexOf(meta.image) + 1), [checkpoint, "--dtype", "auto", "--omni"]);
  assert.ok(args.includes(`${checkpoint}:${checkpoint}:ro`));
});
