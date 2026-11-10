// Opt-in JSON clients shared by the interactive builder and both JSON/Markdown
// renderers. Recipes describe the payload and benchmark workload; the selected
// checkpoint and existing cluster endpoint controls own the target.

export const CLIENT_ENDPOINT_PATTERN = "^/(?!/)[A-Za-z0-9._~/-]*$";
export const CLIENT_RESERVED_BENCHMARK_ARG_PATTERN = "^(?:--(?:model|host|port|endpoint|backend|base[-_]url|dataset(?:[-_]name)?)(?:[=\\s]|$)|-m)";
const RESERVED_BENCHMARK_ARG_RE = new RegExp(CLIENT_RESERVED_BENCHMARK_ARG_PATTERN);

/** Preserve the builder's existing client routing, including routerless KV. */
export function resolveClientTarget(result, endpoints = {}) {
  const deployType = result?.deployType ?? result?.deploy_type;
  const isPd = deployType === "pd_cluster";
  const hasKvRouter = deployType === "kv_store_lb" && !!result.router;
  const routed = isPd || hasKvRouter;
  const hostKey = routed ? "ROUTER_HOST" : deployType === "multi_node" ? "HEAD_IP" : "VLLM_HOST";
  const portKey = routed ? "ROUTER_PORT" : "VLLM_PORT";
  const defaultPort = isPd ? 30000 : hasKvRouter ? (result.router.port || 30080) : 8000;
  return {
    hostKey,
    portKey,
    host: String(endpoints[hostKey] || "localhost"),
    port: String(endpoints[portKey] || defaultPort),
  };
}

// Request data must stay literal, even when it contains $HEAD_IP or shell
// syntax. Unlike serve-command quoting, this never expands placeholders.
function shellQuote(value) {
  const str = String(value);
  return /^[A-Za-z0-9_./=:@,+%-]+$/.test(str)
    ? str
    : `'${str.replace(/'/g, "'\\''")}'`;
}

/** null preserves the legacy text and Omni paths for recipes without client. */
export function renderRecipeClients(recipe, variantKey, result, endpoints = {}) {
  const client = recipe.client;
  if (!client || (recipe.meta?.tasks || []).includes("omni")) return null;
  // Also enforce the selected-target invariants for programmatic callers.
  if (Object.prototype.hasOwnProperty.call(client.request_body, "model")) {
    throw new Error("client.request_body must not declare model; the selected variant supplies it");
  }
  const extraArgs = client.benchmark.args || [];
  if (extraArgs.some((arg) => RESERVED_BENCHMARK_ARG_RE.test(arg))) {
    throw new Error("client.benchmark.args must not override the selected model, target, endpoint, backend or dataset");
  }
  const { host, port } = resolveClientTarget(result, endpoints);
  const modelId = recipe.variants?.[variantKey]?.model_id || recipe.model?.model_id || "model";
  const body = JSON.stringify({ model: modelId, ...client.request_body }, null, 2);
  // Keep the join at runtime: Next 15's minifier folds a literal array into
  // a template whose backslash-newlines become JavaScript line continuations.
  const verifyLines = [
    `curl --fail-with-body -sS -X POST ${shellQuote(`http://${host}:${port}${client.endpoint}`)}`,
    "  -H 'Content-Type: application/json'",
  ];
  verifyLines.push(`  --data-raw ${shellQuote(body)}`);
  const verifyCommand = verifyLines.join(" \\\n");
  const args = [
    ["--model", modelId],
    ["--host", host],
    ["--port", port],
    ["--backend", client.benchmark.backend],
    ["--dataset-name", client.benchmark.dataset_name],
    ["--endpoint", client.endpoint],
    ...(extraArgs.length ? [extraArgs] : []),
  ];
  const benchmarkCommand = ["vllm bench serve", ...args.map((parts) => `  ${parts.map(shellQuote).join(" ")}`)].join(" \\\n");
  return {
    endpoint: client.endpoint,
    model_id: modelId,
    host,
    port,
    verify_command: verifyCommand,
    benchmark_command: benchmarkCommand,
  };
}
