import { resolveDeploymentScript, resolveLaunchScript } from "./scripted-strategies.js";

export function ascendPdPreset(recipe, hardware) {
  const preset = recipe.scripted_strategies?.pd_cluster?.[hardware];
  return preset?.router?.id === "ascend_proxy" ? preset : null;
}

const endpointTokens = {
  PREFILL_NODE_1: "<P0_IP>", PREFILL_NODE_2: "<P1_IP>",
  DECODE_NODE_1: "<D0_IP>", DECODE_NODE_2: "<D1_IP>",
  PREFILL_IFACE_1: "<P0_NIC>", PREFILL_IFACE_2: "<P1_NIC>",
  DECODE_IFACE_1: "<D0_NIC>", DECODE_IFACE_2: "<D1_NIC>", MODEL_PATH: "<MODEL_PATH>",
};

// Only Ascend's fixed workflow uses literal, editable values. Empty inputs must
// not silently turn into references to an environment prepared elsewhere.
export function substituteAscendPd(text, endpoints) {
  if (!text) return text;
  return text.replace(/<P[01]_(?:IP|NIC)>|<D[01]_(?:IP|NIC)>|<MODEL_PATH>/g, token => {
    const key = Object.keys(endpointTokens).find(k => endpointTokens[k] === token);
    return endpoints?.[key] || token;
  });
}

// This adapter emits only the declared, fixed deployment. It does not reuse
// GPU NIXL flags or scale a validated PP/DP topology into arbitrary layouts.
export function resolveAscendPd(recipe, variant, hardware, pdNodes = null) {
  const preset = ascendPdPreset(recipe, hardware);
  if (!preset) return null;
  if (!preset.source || !preset.preparation || !preset.workdir) throw new Error("Ascend Proxy requires pinned source, preparation and workdir");
  if (variant !== preset.variant) throw new Error(`Ascend Proxy requires variant ${preset.variant}`);
  const { workdir } = preset;
  const block = ref => resolveDeploymentScript(recipe, hardware, ref);
  const shellQuote = value => `'${String(value).replaceAll("'", "'\\''")}'`;
  const roleNodes = role => preset.steps.filter(s => s.role === role).sort((a, b) => a.node - b.node);
  const roles = { prefill: roleNodes("prefill"), decode: roleNodes("decode") };
  for (const role of Object.keys(roles)) {
    if (!roles[role].length || roles[role].some((s, i) => s.node !== i || (s.launch_script === undefined && s.launch_block === undefined))) {
      throw new Error(`Invalid Ascend ${role} nodes or missing launch block`);
    }
  }
  const nodeData = (step, role) => {
    const label = `${role === "prefill" ? "P" : "D"}${step.node}`;
    const ip = endpointTokens[`${role === "prefill" ? "PREFILL" : "DECODE"}_NODE_${step.node + 1}`];
    const nic = endpointTokens[`${role === "prefill" ? "PREFILL" : "DECODE"}_IFACE_${step.node + 1}`];
    let template = block(step)
      .replace(/^nic_name=.*$/m, `nic_name="${nic}"`)
      .replace(/^local_ip=.*$/m, `local_ip="${ip}"`)
      .replace(/^node_p0_ip=.*$/m, 'node_p0_ip="<P0_IP>"');
    const checkpointPath = template.match(/vllm serve\s+(\S+)/)?.[1];
    if (!checkpointPath) throw new Error(`Missing serve command in ${label}`);
    template = template.replace(`vllm serve ${checkpointPath}`, 'vllm serve "$MODEL_PATH"');
    template = `#!/usr/bin/env bash\n# ${label}: container vllm-ascend, ${workdir}/run_dp_template.sh\n# Fill this node's <...> values here, or use Cluster endpoints before copying.\nexport VLLM_USE_RUST_FRONTEND=0\nMODEL_PATH="<MODEL_PATH>"\n` + template;
    if (step.visible_devices) template = template.replace('export VLLM_USE_RUST_FRONTEND=0',
      `export VLLM_USE_RUST_FRONTEND=0\nexport ASCEND_RT_VISIBLE_DEVICES=${step.visible_devices}`);
    let launch = resolveLaunchScript(recipe, hardware, step)
      .replaceAll("$node_d0_ip", '"<D0_IP>"')
      .replaceAll("$node_p0_ip", '"<P0_IP>"')
      .replaceAll("$node_p1_ip", '"<P1_IP>"');
    if (step.launch_type === "direct") {
      if (launch.trim() !== "bash run_dp_template.sh") throw new Error(`Invalid direct launch in ${label}`);
    } else if (step.launch_type === "launcher") {
      if (!launch.includes("python launch_online_dp.py")) throw new Error(`Invalid DP launch in ${label}`);
    } else throw new Error(`Missing launch type in ${label}`);
    if (template.includes("python launch_online_dp.py") || template.includes("bash run_dp_template.sh")) {
      throw new Error(`Recursive launcher in ${label} template`);
    }
    const saveCommand = `# Inside ${label}'s vllm-ascend container. Save the template; do not start it yet.\nmkdir -p ${shellQuote(workdir)}\ncd ${shellQuote(workdir)}\ncat > run_dp_template.sh <<'RECIPE_TEMPLATE_EOF'\n${template}\nRECIPE_TEMPLATE_EOF`;
    const launcherNote = step.launch_type === "launcher" ? `# First save the common launcher as ${workdir}/launch_online_dp.py, alongside the template.\n` : "";
    const launchCommand = `# Inside ${label}'s vllm-ascend container, after saving ${workdir}/run_dp_template.sh.\n${launcherNote}cd ${shellQuote(workdir)}\n${launch}`;
    return { label, node: step.node, template, command: saveCommand, saveCommand, launchCommand,
      launchType: step.launch_type, checkpointPath };
  };
  const all = Object.fromEntries(Object.entries(roles).map(([role, steps]) => [role, steps.map(s => nodeData(s, role))]));
  const selected = role => {
    const raw = pdNodes?.[role]?.rank ?? 0;
    const currentNode = Math.max(0, Math.min(all[role].length - 1, Math.floor(Number(raw) || 0)));
    return { ...all[role][currentNode], env: {}, nodes: all[role].length, currentNode,
      fixed: true, nodeHint: all[role][currentNode].label,
      nodeLabels: all[role].map(n => n.label), nodeCommands: all[role],
      description: `${all[role][currentNode].label} · ${preset.model_id}` };
  };
  const proxyStep = preset.steps.find(s => s.role === "proxy");
  if (!proxyStep) throw new Error("Missing Ascend Proxy command");
  const proxy = block(proxyStep);
  return {
    deployType: "pd_cluster", orchestrator: "ascend_proxy", fixed: true,
    modelId: preset.model_id, servedModelName: all.prefill[0].template.match(/--served-model-name\s+(\S+)/)?.[1] || preset.model_id, source: preset.source,
    prefill: selected("prefill"), decode: selected("decode"),
    router: { id: "ascend_proxy", label: "Ascend Proxy", port: preset.router.port,
      command: proxy, env: {} },
    preparation: {
      hardwareLabel: preset.hardware_label,
      containerNodes: [...all.prefill, ...all.decode].map(n => n.label),
      launcherNodes: [...all.prefill, ...all.decode].filter(n => n.launchType === "launcher").map(n => n.label),
      directNodes: [...all.prefill, ...all.decode].filter(n => n.launchType === "direct").map(n => n.label),
      hostCommand: block(preset.preparation.docker),
      launcherSource: block(preset.preparation.launcher),
      launcherPath: `${workdir}/launch_online_dp.py`,
      workdir, revision: preset.source.revision,
    },
    endpointNames: ["PREFILL_NODE_1", "PREFILL_NODE_2", "DECODE_NODE_1", "DECODE_NODE_2", "PREFILL_IFACE_1", "PREFILL_IFACE_2", "DECODE_IFACE_1", "DECODE_IFACE_2", "MODEL_PATH", "ROUTER_HOST"],
    constraints: preset.constraints,
  };
}
