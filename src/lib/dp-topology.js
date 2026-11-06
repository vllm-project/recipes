// Ordinary TP+DP/DEP launches one vLLM supervisor per node. Each local DP
// replica owns TP devices; global DP counts replicas, never physical devices.
export const DP_STRATEGIES = new Set(["multi_node_tp_dp", "single_node_dep", "multi_node_dep"]);

export class DpLayoutError extends Error {}

const aliases = {
  "-tp": "--tensor-parallel-size", "-pp": "--pipeline-parallel-size",
  "-dp": "--data-parallel-size", "-dpl": "--data-parallel-size-local",
  "-dpn": "--data-parallel-rank", "-dpr": "--data-parallel-start-rank",
  "-dpa": "--data-parallel-address", "-dpp": "--data-parallel-rpc-port",
  "-dpb": "--data-parallel-backend", "-dph": "--data-parallel-hybrid-lb",
  "-dpe": "--data-parallel-external-lb", "-dpm": "--data-parallel-multi-port-external-lb",
  "-pcp": "--prefill-context-parallel-size", "-n": "--nnodes", "-r": "--node-rank",
  "-asc": "--api-server-count",
};
const bools = new Set(["--headless", "--data-parallel-hybrid-lb", "--data-parallel-external-lb", "--data-parallel-multi-port-external-lb"]);
const owned = new Set([
  ...Object.values(aliases), ...bools, "--master-addr",
  ...[...bools].map((f) => `--no-${f.slice(2)}`),
]);

// Normalize only topology flags, including vLLM's underscore / short forms.
// Keep other tokens intact (JSON strings and option values are not flags).
function canonicalArgs(args, fail) {
  return args.flatMap((arg) => {
    if (typeof arg !== "string" || !arg.startsWith("-")) return [arg];
    const attached = arg.match(/^(-dpl|-dpn|-dpr|-dpp|-dp|-tp|-pp|-pcp|-asc|-n|-r)(-?\d+(?:\.\d+)?)$/);
    if (attached) return [aliases[attached[1]], attached[2]];
    const equals = arg.indexOf("=");
    const spelling = (equals < 0 ? arg : arg.slice(0, equals)).replaceAll("_", "-");
    const flag = aliases[spelling] || spelling;
    if (!owned.has(flag)) {
      if (spelling.startsWith("--") && [...owned].some((f) => f.startsWith(spelling))) {
        fail(`Use the full topology flag name instead of ${spelling}.`);
      }
      return [arg];
    }
    if (bools.has(flag) || flag.startsWith("--no-")) {
      if (equals >= 0) fail(`${flag} does not take a value.`);
      return [flag];
    }
    return equals < 0 ? [flag] : [flag, arg.slice(equals + 1)];
  });
}

export function normalizeDpArgs(args, { strategyName, gpuCount, nodes, nodeRank, address, architecture, brand, env = {} }) {
  const fail = (reason) => { throw new DpLayoutError(`${strategyName}: ${reason}`); };
  const tokens = canonicalArgs(args, fail);
  const values = new Map();
  const rest = [];
  for (let i = 0; i < tokens.length; i++) {
    const flag = tokens[i];
    if (!owned.has(flag)) { rest.push(flag); continue; }
    if (bools.has(flag) || flag.startsWith("--no-")) {
      if (tokens[i + 1] != null && !String(tokens[i + 1]).startsWith("-")) fail(`${flag} does not take a value.`);
      if (bools.has(flag)) values.set(flag, true);
      else values.set(`--${flag.slice(5)}`, false);
    }
    else {
      const value = tokens[++i];
      if (value == null || (String(value).startsWith("-") && !/^-\d/.test(String(value)))) fail(`${flag} requires a value.`);
      values.set(flag, value);
    }
  }
  const integer = (name, fallback, min = 1) => {
    const n = values.has(name) ? Number(values.get(name)) : fallback;
    if (!Number.isInteger(n) || n < min) fail(`${name} must be an integer >= ${min} (got ${n}).`);
    return n;
  };
  if (!Number.isInteger(nodes) || nodes < 1 || !Number.isInteger(gpuCount) || gpuCount < 1) {
    fail("Node and device counts must be positive integers.");
  }
  const tp = integer("--tensor-parallel-size", strategyName === "multi_node_tp_dp" ? gpuCount : 1);
  if (tp > gpuCount || gpuCount % tp !== 0) fail(`TP must divide the ${gpuCount} devices per node (got ${tp}).`);
  if (integer("--pipeline-parallel-size", 1) !== 1) fail("Ordinary TP+DP/DEP currently requires PP1.");
  if (integer("--prefill-context-parallel-size", 1) !== 1) fail("Ordinary TP+DP/DEP currently requires PCP1.");
  if (integer("--nnodes", 1) !== 1 || integer("--node-rank", 0, 0) !== 0 || values.has("--master-addr")) {
    fail("Use DP address/start ranks for node-local replicas; cross-node TP/PP rendezvous is unsupported here.");
  }
  if (values.has("--data-parallel-backend") && values.get("--data-parallel-backend") !== "mp") {
    fail("Ordinary TP+DP/DEP requires the mp DP backend.");
  }
  if (values.has("--data-parallel-rpc-port")) {
    const flag = "--data-parallel-rpc-port";
    const raw = String(values.get(flag));
    if (!/^\$(?:[A-Za-z_][A-Za-z0-9_]*|\{[A-Za-z_][A-Za-z0-9_]*\})$/.test(raw)) {
      const port = integer(flag);
      if (port > 65535) fail(`${flag} must be in 1..65535.`);
      values.set(flag, String(port));
    }
  }
  if (values.has("--data-parallel-address") && !String(values.get("--data-parallel-address")).trim()) {
    fail("DP address must not be empty.");
  }
  const localFlag = "--data-parallel-size-local";
  const globalFlag = "--data-parallel-size";
  const local = integer(localFlag, values.has(globalFlag) ? integer(globalFlag) / nodes : gpuCount / tp);
  const global = integer(globalFlag, local * nodes);
  if (global !== local * nodes) fail(`Global DP ${global} must equal local DP ${local} × ${nodes} nodes.`);
  if (local * tp > gpuCount) fail(`TP ${tp} × local DP ${local} exceeds ${gpuCount} devices per node.`);
  const visibleKeys = brand === "Huawei" ? ["ASCEND_RT_VISIBLE_DEVICES"]
    : brand === "AMD" ? ["CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"]
      : ["CUDA_VISIBLE_DEVICES"];
  for (const key of visibleKeys) {
    if (env[key] == null) continue;
    const mask = String(env[key]).trim();
    // Dynamic shell expressions cannot be counted statically. Literal lists
    // can restrict the budget further, but never enlarge the hardware profile.
    if (!/^(?:\d+(?:\s*,\s*\d+)*)?$/.test(mask) && mask !== "-1") continue;
    const ids = mask && mask !== "-1" ? mask.split(",").map((s) => Number(s.trim())) : [];
    if (new Set(ids).size !== ids.length || local * tp > ids.length) {
      fail(`TP ${tp} × local DP ${local} exceeds or conflicts with ${key}=${mask}.`);
    }
  }
  const start = nodeRank * local;
  if (!Number.isInteger(nodeRank) || nodeRank < 0 || nodeRank >= nodes) fail("Node rank is outside the selected cluster.");
  if (values.has("--data-parallel-start-rank") && integer("--data-parallel-start-rank", 0, 0) !== start) {
    fail(`DP start rank must be node index × local DP (${start}).`);
  }
  let hybrid = values.get("--data-parallel-hybrid-lb") === true;
  const external = values.get("--data-parallel-external-lb") === true || values.has("--data-parallel-rank");
  if (values.get("--data-parallel-multi-port-external-lb")) fail("Multi-port external LB requires a separate supervisor/endpoint layout.");
  if (external) {
    if (global <= 1 || local !== 1) fail("External LB requires global DP > 1 and local DP = 1.");
    if (hybrid || values.get("--headless") || values.has("--data-parallel-start-rank")) {
      fail("External LB cannot combine hybrid LB, headless or DP start rank; use one DP rank per node.");
    }
    if (values.has("--data-parallel-rank") && integer("--data-parallel-rank", 0, 0) !== nodeRank) fail(`External DP rank must equal node index (${nodeRank}).`);
  }
  if (global === 1) hybrid = false;
  if (values.get("--headless") && (nodeRank === 0 || hybrid)) fail("Headless is only valid on internal-LB follower nodes.");
  // With all replicas on one node, vLLM uses internal LB even if hybrid was
  // requested. Avoid its localDP=1 hybrid→external conversion for DP1.
  if (nodes === 1) hybrid = false;
  if (nodeRank > 0 && !hybrid && !external && values.get("--headless") === false) {
    fail("Internal-LB follower nodes require headless mode.");
  }
  if (architecture === "dense" && (external || (hybrid && local === 1))) {
    fail("External DP load balancing requires a MoE model; use internal LB for dense models.");
  }
  if (values.has("--api-server-count")) {
    const count = integer("--api-server-count", 1, 0);
    const headless = nodeRank > 0 && !hybrid && !external;
    if (headless ? count > 0 : count === 0) fail("API server count conflicts with the selected DP load-balancing mode.");
    values.set("--api-server-count", String(count));
  }
  const out = [...rest, "--tensor-parallel-size", String(tp), globalFlag, String(global), localFlag, String(local)];
  if (nodes > 1 || values.has("--data-parallel-address")) out.push("--data-parallel-address", values.get("--data-parallel-address") || address);
  for (const flag of ["--data-parallel-rpc-port", "--data-parallel-backend", "--api-server-count"]) {
    if (values.has(flag)) out.push(flag, String(values.get(flag)));
  }
  if (external) out.push("--data-parallel-external-lb", "--data-parallel-rank", String(nodeRank));
  else {
    if (hybrid) out.push("--data-parallel-hybrid-lb");
    if (nodeRank > 0) {
      out.push("--data-parallel-start-rank", String(start));
      if (!hybrid) out.push("--headless");
    }
  }
  return out;
}
