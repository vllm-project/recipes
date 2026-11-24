// Compile a source-owned deployment once for the interactive guide, shell,
// raw argv, and API exports. No host input is evaluated or sent to a server.
import { makeBindings, poolInputs, quoteWord } from "./pool-inputs.js";
import { orderSteps, placementsFor } from "./pool-profile-validation.js";
import { multiConnector } from "./kv-connector.js";
export { validatePoolDeployment, validatePoolProfile } from "./pool-profile-validation.js";

const shellCommand = (argv) => argv.map(quoteWord).join(" ");
const unique = (array) => [...new Set(array)];
const idOf = (placement) => placement.id ?? placement.role;

function contextsFor(target, deployment, mode) {
  if (target === "hosts" || target.startsWith("host:")) {
    const hosts = target === "hosts" ? deployment.hosts : [target.slice(5)];
    return hosts.map((host) => ({ host }));
  }
  if (target === "containers") return Object.entries(deployment.containers).map(([container, c]) => ({ host: c.host, container }));
  return placementsFor(target, deployment.placements, mode).map((placement) => ({
    placement, role: placement.role, host: deployment.containers[placement.container].host, container: placement.container,
  }));
}

function bindingsFor(fields, ctx, devices) {
  const bindings = makeBindings(fields, ctx);
  // Device values are resolved lazily: an empty SSD field should hold only
  // the steps that use it, not an unrelated host's connectivity checks.
  const extra = Object.defineProperty({}, "pool_device_paths", { enumerable: true, get() {
    return devices.map((d) => bindings.bind(d.target ?? d.source)).join(":");
  } });
  const main = makeBindings(fields, ctx, extra);
  return { bind: main.bind, issues: () => ({
    missing: unique([...bindings.missing, ...main.missing]),
    errors: unique([...bindings.errors, ...main.errors]),
  }) };
}

function containerArgv({ container }, spec, profile, deployment, image, bind) {
  const argv = ["docker", "run", "-d", "--name", container,
    ...(spec.docker_args || ["--network", "host", "--ipc", "host"]).map((arg) => String(bind(arg)))];
  for (const m of [...(profile.mounts || []), ...(deployment.mounts || [])]) {
    argv.push("-v", bind(m.source) + ":" + bind(m.target) + (m.read_only ? ":ro" : ""));
  }
  for (const d of [...(profile.devices || []), ...(deployment.devices || [])]) {
    argv.push("--device", bind(d.source) + ":" + bind(d.target ?? d.source));
  }
  return [...argv, "--entrypoint", "/bin/bash", image, "-c", "exec sleep infinity"];
}

function bindArgument(value, bind) {
  // A JSON-valued CLI flag is one argv entry. Interpolate inside the parsed
  // structure so a quote or backslash in a checkpoint path stays data.
  if (typeof value === "string" && /^[\[{]/.test(value) && !value.startsWith("{{")) {
    let parsed;
    try { parsed = JSON.parse(value); } catch { /* ordinary command argument */ }
    if (parsed !== undefined) return JSON.stringify(bind(parsed));
  }
  return String(bind(value));
}

function launchProcess(ctx, profile, bind) {
  const p = ctx.placement;
  const launch = p.launch;
  const argv = launch.argv.map((v) => bindArgument(v, bind));
  const env = Object.fromEntries(Object.entries(launch.env || {}).map(([k, v]) => [k, String(bind(v))]));
  if (p.devices?.length) env.ASCEND_RT_VISIBLE_DEVICES = p.devices.join(",");
  let transfer;
  if (p.role === "prefill" || p.role === "decode") {
    const kvRole = p.role === "prefill" ? "kv_producer" : "kv_consumer";
    const direct = bind(launch.transfer ?? p.transfer);
    const store = { ...bind(profile.store[p.role]), ...bind(launch.store ?? p.store ?? {}) };
    transfer = multiConnector([
      { ...direct, kv_role: kvRole }, { kv_connector: "AscendStoreConnector", kv_role: kvRole, kv_connector_extra_config: store },
    ], kvRole);
    argv.push("--kv-transfer-config", JSON.stringify(transfer));
  }
  const exports = Object.entries(env).map(([k, v]) => "export " + k + "=" + quoteWord(v));
  const setup = launch.setup ? bind(launch.setup, true).trim() : "";
  const command = [...(setup ? [setup] : []), ...exports, shellCommand(argv)].join("\n");
  const dockerArgv = ["docker", "exec", ctx.container, "bash", "-c", "set -e\n" + command];
  return { id: idOf(p), role: p.role, host: ctx.host, container: ctx.container, devices: p.devices || [],
    ...(p.rank != null && { rank: p.rank }), argv, env, command, docker_argv: dockerArgv,
    docker_command: shellCommand(dockerArgv), ...(transfer && { transfer }) };
}

function hostBudgets(hosts, budgets, fields, launches) {
  const readNumber = (host, name) => {
    const field = fields.get("hosts." + host + "." + name) ?? fields.get(name);
    return field && !field.missing && !field.error ? Number(field.value) : NaN;
  };
  return hosts.flatMap((host) => {
    const workers = launches.filter((l) => l.host === host && ["prefill", "decode"].includes(l.role))
      .reduce((sum, l) => sum + l.devices.length, 0);
    return budgets.flatMap((b) => {
      const count = b.count === "workers" ? workers : readNumber(host, b.count);
      const per = b.input == null ? b.value : readNumber(host, b.input);
      return Number.isFinite(per) && Number.isFinite(count) && count > 0
        ? [{ host, label: b.label, count, per_instance: per, total: count * per, unit: b.unit }] : [];
    });
  });
}

export function resolvePoolDeployment({ recipe, presetKey, mode, inputs = {} }) {
  const preset = recipe?.presets?.[presetKey];
  const profileId = preset?.select?.kv_offload;
  const profile = recipe?._poolProfiles?.[profileId];
  const deployment = preset?.deployment;
  if (!profile || !deployment) return null;
  const selectedMode = ["new", "reuse"].includes(mode) ? mode : null;
  const fields = poolInputs(profile, deployment, selectedMode, inputs);
  const index = new Map(fields.map((f) => [f.key, f]));
  for (const [key, value] of Object.entries(deployment.constants || {})) index.set(key, { key, value });
  const devices = [...(profile.devices || []), ...(deployment.devices || [])];
  const activePlacements = deployment.placements.filter((p) => !selectedMode || !p.mode || p.mode === selectedMode);
  const placements = activePlacements.map((p) => ({
    id: idOf(p), role: p.role, host: deployment.containers[p.container].host, container: p.container,
    devices: p.devices || [], mode: p.mode || null, ...(p.rank != null && { rank: p.rank }),
  }));
  const launches = activePlacements.filter((p) => p.launch).map((p) => {
    const ctx = { placement: p, role: p.role, host: deployment.containers[p.container].host, container: p.container };
    const b = bindingsFor(index, ctx, devices);
    return { ...launchProcess(ctx, profile, b.bind), ...b.issues() };
  });
  const image = typeof preset.image === "string" ? preset.image : preset.image?.vllm;
  const steps = [];
  const expanded = new Map();
  for (const template of orderSteps([...profile.steps, ...(deployment.steps || [])])) {
    if (selectedMode && template.mode && selectedMode !== template.mode) continue;
    const batch = [];
    for (const ctx of contextsFor(template.target, deployment, selectedMode)) {
      const b = bindingsFor(index, ctx, devices);
      const place = ctx.placement ? idOf(ctx.placement) : ctx.container ?? ctx.host;
      const step = { id: template.id + "/" + place, title: template.title, stage: template.stage,
        host: ctx.host, container: ctx.container || null, role: ctx.role || null, mode: template.mode ?? ctx.placement?.mode ?? null,
        check: String(b.bind(template.check)), command: "" };
      let launch;
      if (template.action === "launch") {
        launch = launches.find((l) => l.id === place);
        if (!launch) throw new Error("Missing launch for " + step.id);
        Object.assign(step, { command: launch.command, argv: launch.argv });
      } else if (template.action === "container") {
        step.argv = containerArgv(ctx, deployment.containers[ctx.container], profile, deployment, image, b.bind);
        step.command = shellCommand(step.argv);
        // docker run is issued on the host, even when it creates one specific container.
        step.container = null;
      } else if (template.command) {
        step.command = String(b.bind(template.command, true)).trimEnd();
      }
      const prerequisites = (template.depends_on || []).flatMap((id) => {
        const candidates = expanded.get(id) || [];
        const sameHost = candidates.filter((s) => s.host === ctx.host);
        return (sameHost.length && !template.target.startsWith("placement:") ? sameHost : candidates).map((s) => s.id);
      });
      const issues = b.issues();
      step.depends_on = unique(prerequisites);
      step.missing = unique([...issues.missing, ...(launch?.missing || [])]);
      step.errors = unique([...issues.errors, ...(launch?.errors || [])]);
      batch.push(step);
    }
    expanded.set(template.id, batch);
    steps.push(...batch);
  }
  return { profile: profileId, label: profile.label, source: profile.source || null,
    ascend_version: deployment.ascend_version, tutorial: deployment.tutorial, mode: selectedMode,
    hosts: deployment.hosts, containers: deployment.containers, placements, inputs: fields, steps, launches,
    parallelism: Object.fromEntries(Object.entries(preset.roles || {}).map(([role, settings]) => [role, settings.parallelism])),
    capacity: hostBudgets(deployment.hosts, profile.capacity || [], index, launches),
    missing: unique([...fields.filter((f) => f.missing).map((f) => f.key), ...steps.flatMap((s) => s.missing)]),
    errors: unique([...fields.filter((f) => f.error).map((f) => f.key + ": " + f.error), ...steps.flatMap((s) => s.errors)]) };
}

// Each node may run more than one independent service process. Preserve all
// processes rather than replaying rank zero, including their per-rank env/IPC.
function hostCommand(processes, node_rank) {
  const process_commands = processes.map(({ argv, env, command, docker_argv, docker_command, id, host, container, devices }) =>
    ({ placement: id, host, container, devices, argv, env, command, docker_argv, docker_command }));
  const first = process_commands[0];
  if (process_commands.length === 1) return { ...first, node_rank };
  // Per-host exports must actually start every local rank. Keep each rank's
  // setup/env in its own shell and stop sibling ranks if one process exits.
  const sameContainer = process_commands.every((p) => p.container === first.container);
  const script = [
    "set -e", "pids=()", "trap 'kill \"${pids[@]}\" 2>/dev/null || true' EXIT",
    ...process_commands.map((p) => "(\n" + (sameContainer ? p.command : p.docker_command) + "\n) &\npids+=(\"$!\")"),
    "wait -n",
  ].join("\n");
  const argv = ["bash", "-c", script];
  const docker_argv = sameContainer ? ["docker", "exec", first.container, ...argv] : argv;
  return { node_rank, host: first.host, container: sameContainer ? first.container : null,
    env: {}, argv, command: shellCommand(argv), docker_argv, docker_command: shellCommand(docker_argv), process_commands };
}

function roleView(plan, role, rank = 0) {
  const launches = plan.launches.filter((l) => l.role === role);
  const hosts = unique(launches.map((l) => l.host));
  const currentNode = Math.min(Math.max(Number(rank) || 0, 0), hosts.length - 1);
  const node_commands = hosts.map((host, node_rank) => hostCommand(launches.filter((l) => l.host === host), node_rank));
  const first = launches[0], selected = node_commands.find((n) => n.node_rank === currentNode);
  const topology = first.transfer.kv_connector_extra_config.connectors[0].kv_connector_extra_config?.[role] || {};
  const flag = (name) => {
    const i = first.argv.indexOf(name);
    return i < 0 ? undefined : first.argv[i + 1];
  };
  const tp = Number(flag("--tensor-parallel-size") ?? flag("-tp") ?? topology.tp_size ?? 1);
  const pp = Number(flag("--pipeline-parallel-size") ?? flag("-pp") ?? topology.pp_size ?? 1);
  const dpSize = Number(flag("--data-parallel-size") ?? flag("-dp") ?? topology.dp_size ?? 1);
  return { ...selected, nodes: hosts.length, parallelism: plan.parallelism[role] ?? (dpSize > 1 ? "dep" : "tp"), tp, pp, dpSize,
    dpLocal: dpSize / hosts.length, currentNode, startRank: currentNode * dpSize / hosts.length,
    poolGpus: tp * pp * dpSize, node_commands };
}

export function poolCommand(plan, pdNodes = {}) {
  const proxy = plan.launches.find((l) => l.role === "proxy");
  const portFlag = proxy.argv.indexOf("--port");
  const port = portFlag >= 0 ? Number(proxy.argv[portFlag + 1]) : null;
  return { deployType: "pd_cluster", nodeCount: plan.hosts.length, poolDeployment: plan,
    pdRouter: "ascend", orchestrator: "ascend-proxy", frontendRequired: "python",
    prefill: roleView(plan, "prefill", pdNodes?.prefill?.rank), decode: roleView(plan, "decode", pdNodes?.decode?.rank),
    router: { ...proxy, label: "Ascend Router", dockerCommand: proxy.docker_command,
      ...(Number.isInteger(port) && port > 0 && { port }) },
    kvConnector: { selected: "mooncake_ascend", install: [] } };
}

export function applyPoolDeployment(rendering, plan) {
  if (!plan) return rendering;
  const result = poolCommand(plan);
  return { ...rendering, pool_deployment: plan, deploy_type: "pd_cluster", node_count: plan.hosts.length,
    pd_router: result.pdRouter, pd_connector: result.kvConnector.selected,
    prefill: roleView(plan, "prefill", rendering?.prefill?.currentNode),
    decode: roleView(plan, "decode", rendering?.decode?.currentNode), router: result.router };
}
