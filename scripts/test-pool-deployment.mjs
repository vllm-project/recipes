import test from "node:test";
import assert from "node:assert/strict";
import { applyPoolDeployment, resolvePoolDeployment, validatePoolDeployment, validatePoolProfile } from "../src/lib/pool-deployment.js";
import { inputError } from "../src/lib/pool-inputs.js";

const profile = {
  schema_version: 1, backend: "memcache", label: "Memcache",
  inputs: {
    address: { scope: "host", type: "host", label: "IP" },
    weights: { scope: "host", type: "path", label: "Weights" },
    disk: { scope: "host", type: "device", label: "SSD" },
    port: { scope: "cluster", type: "port", label: "Meta port", default: 25250 },
  },
  store: { prefill: { backend: "memcache" }, decode: { backend: "memcache" } },
  mounts: [{ source: "{{weights}}", target: "/checkpoint", read_only: true }],
  devices: [{ source: "{{disk}}" }],
  steps: [
    { id: "container", stage: "container", target: "containers", action: "container", title: "Create container", check: "Running" },
    { id: "meta", stage: "control", target: "control", mode: "new", title: "Start Meta", command: "start-meta --port {{port}}", check: "Listening" },
    { id: "disk-check", stage: "config", target: "containers", title: "Mapped disk", command: "lsblk {{pool_device_paths}}", check: "Device present" },
  ],
};
const server = (role) => ({
  role, container: "model", devices: role === "prefill" ? [0, 1] : [2, 3],
  launch: {
    argv: ["vllm", "serve", "/checkpoint", "--tensor-parallel-size", "2", "--config", '{"path":"{{weights}}"}'],
    env: { VLLM_HOST_IP: "{{address}}" },
    transfer: { kv_connector: "MooncakeConnectorV2", kv_port: "{{port}}" },
    store: { lookup_rpc_port: role },
  },
});
const deployment = {
  hosts: ["h1"], containers: { model: { host: "h1" } },
  placements: [server("prefill"), server("decode"), { role: "meta", mode: "new", container: "model" },
    { role: "proxy", container: "model", launch: { argv: ["python", "proxy.py", "--port", "19080", "--prefiller-hosts", "{{hosts.h1.address}}"] } }],
  steps: ["prefill", "decode", "proxy"].map((role) => ({
    id: "start-" + role, target: role, stage: "serve", action: "launch", title: role, check: "HTTP health",
    depends_on: role === "proxy" ? ["start-prefill", "start-decode"] : ["disk-check"],
  })),
};
const recipe = {
  _poolProfiles: { "vllm-ascend/memcache": profile },
  presets: { qwen: { deployment, roles: { prefill: { parallelism: "tep" } }, image: "ascend:rc1", select: { kv_offload: "vllm-ascend/memcache" } } },
};
const inputs = { "hosts.h1.weights": '/data/O\'Brien "weights"', "hosts.h1.address": "192.0.2.5", "hosts.h1.disk": "/dev/nvme1n1" };
const resolve = (mode, supplied = inputs) => resolvePoolDeployment({ recipe, presetKey: "qwen", mode, inputs: supplied });

test("new and existing modes share a single physical container and use the assigned devices", () => {
  const reuse = resolve("reuse");
  assert.deepEqual(reuse.errors, []);
  assert.deepEqual(reuse.missing, []);
  assert.ok(!reuse.steps.some((s) => s.id.startsWith("meta/")));
  assert.equal(resolve("new").steps.filter((s) => s.id.startsWith("meta/")).length, 1);
  assert.equal(resolve("new").placements.find((p) => p.role === "meta").host, "h1");
  assert.ok(reuse.placements.every((p) => p.role !== "meta"));
  assert.equal(resolve(null).steps.find((s) => s.id.startsWith("meta/")).mode, "new");
  const create = reuse.steps.find((s) => s.id === "container/model");
  assert.ok(create.argv.includes(inputs["hosts.h1.weights"] + ":/checkpoint:ro"));
  assert.equal(create.container, null);
  assert.ok(reuse.steps.find((s) => s.id === "disk-check/model").command.includes("/dev/nvme1n1"));
  assert.deepEqual(reuse.steps.at(-1).depends_on, ["start-prefill/prefill", "start-decode/decode"]);
});
test("typed inputs survive JSON and shell serialization, and old API command fields match", () => {
  const plan = resolve("new");
  const json = applyPoolDeployment({ hardware: "huawei" }, plan);
  for (const [role, kvRole] of [["prefill", "kv_producer"], ["decode", "kv_consumer"]]) {
    const launch = json[role], args = launch.argv;
    assert.equal(args.filter((a) => a === "--kv-transfer-config").length, 1);
    const config = JSON.parse(args[args.indexOf("--kv-transfer-config") + 1]);
    assert.equal(config.kv_role, kvRole);
    const [direct, store] = config.kv_connector_extra_config.connectors;
    assert.equal(direct.kv_role, kvRole);
    assert.equal(direct.kv_port, 25250);
    assert.equal(store.kv_connector, "AscendStoreConnector");
    assert.equal(store.kv_connector_extra_config.lookup_rpc_port, role);
    assert.equal(JSON.parse(args[args.indexOf("--config") + 1]).path, inputs["hosts.h1.weights"]);
    assert.deepEqual(launch.docker_argv.slice(0, 4), ["docker", "exec", "model", "bash"]);
    assert.equal(launch.command, launch.node_commands[0].command);
  }
  assert.equal(json.prefill.env.ASCEND_RT_VISIBLE_DEVICES, "0,1");
  assert.equal(json.decode.env.ASCEND_RT_VISIBLE_DEVICES, "2,3");
  assert.equal(json.prefill.parallelism, "tep");
  assert.equal(json.router.port, 19080);
});
test("a physical node runs every colocated Decode process and budgets its assigned workers once", () => {
  const multiple = structuredClone(recipe);
  const layout = multiple.presets.qwen.deployment;
  layout.placements[1].id = "d0";
  layout.placements[1].devices = [2];
  layout.placements.push({ ...structuredClone(layout.placements[1]), id: "d1", devices: [3] });
  multiple._poolProfiles["vllm-ascend/memcache"].capacity = [
    { label: "Worker DRAM", count: "workers", value: 2, unit: "GiB" },
  ];
  const plan = resolvePoolDeployment({ recipe: multiple, presetKey: "qwen", mode: "new", inputs });
  const { decode } = applyPoolDeployment({}, plan);
  assert.equal(decode.nodes, 1);
  assert.equal(decode.node_commands.length, 1);
  assert.deepEqual(decode.process_commands.map((p) => [p.placement, p.env.ASCEND_RT_VISIBLE_DEVICES]), [["d0", "2"], ["d1", "3"]]);
  assert.deepEqual(decode.docker_argv.slice(0, 5), ["docker", "exec", "model", "bash", "-c"]);
  assert.deepEqual(plan.capacity, [{ host: "h1", label: "Worker DRAM", count: 4, per_instance: 2, total: 8, unit: "GiB" }]);
});
test("missing inputs are named in templates and invalid addresses never become executable values", () => {
  const plan = resolve(null, { "hosts.h1.address": "invalid host" });
  assert.ok(plan.errors.length);
  assert.ok(plan.missing.includes("hosts.h1.weights"));
  assert.equal(plan.launches[0].env.VLLM_HOST_IP, "{{hosts.h1.address}}");
  assert.ok(plan.steps.find((s) => s.action !== "container" && s.id.startsWith("start-prefill")).errors.length);
  assert.equal(inputError("port", "65536") !== null, true);
  assert.equal(inputError("integer", "2e3") !== null, true);
});
test("the preset owns configuration names shared by preparation and serving", () => {
  const fixed = structuredClone(recipe);
  const layout = fixed.presets.qwen.deployment;
  layout.constants = { ini_name: "mmc-local.conf" };
  layout.steps[0].check = "Readable: {{ini_name}}";
  layout.placements[0].launch.env.CONFIG = "{{ini_name}}";
  assert.deepEqual(validatePoolDeployment(profile, layout, { gpu_count: 4 }), []);
  const plan = resolvePoolDeployment({ recipe: fixed, presetKey: "qwen", mode: "reuse", inputs: { ...inputs, ini_name: "override" } });
  assert.ok(!plan.inputs.some((f) => f.key === "ini_name"));
  assert.equal(plan.launches[0].env.CONFIG, "mmc-local.conf");
  assert.equal(plan.steps.find((s) => s.id === "start-prefill/prefill").check, "Readable: mmc-local.conf");
  layout.constants.port = 10;
  assert.match(validatePoolDeployment(profile, layout, { gpu_count: 4 }).join(" "), /conflicting deployment constant port/);
});
test("source validation rejects cycles, unresolved input references and overlapping physical devices", () => {
  assert.deepEqual(validatePoolProfile(profile), []);
  assert.deepEqual(validatePoolDeployment(profile, deployment, { gpu_count: 4 }), []);
  const bad = structuredClone(deployment);
  bad.placements[1].devices = [0, 8];
  bad.steps[0].depends_on = [bad.steps[0].id];
  bad.steps[1].check = "{{hosts.unknown.disk}}";
  const errors = validatePoolDeployment(profile, bad, { gpu_count: 4 }).join("\n");
  for (const pattern of [/cycle/, /overlapping device/, /Unknown input reference/]) assert.match(errors, pattern);
});
