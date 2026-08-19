/**
 * Cluster endpoints: lets users fill in real IPs/ports for the $VAR
 * placeholders that command-synthesis emits ($HEAD_IP, $PREFILL_HOST_IP,
 * PREFILL_NODE_1, …). Values persist in localStorage so a user fills
 * once per cluster and every recipe page substitutes their values into
 * commands, env exports, curl tests, and bench runs.
 *
 * Empty/unset fields stay as placeholders — the rendered command is still
 * valid shell (variables a user can later `export`).
 */

const STORAGE_KEY = "vllm-recipes:cluster-endpoints";

/**
 * Detect $VAR placeholders in a piece of rendered text. Returns a deduped
 * list of { kind: "var", name, label } — `name` is the bare identifier the
 * panel stores values against; `label` is what we show on the input row.
 *
 * The router's --prefill / --decode endpoints use $PREFILL_NODE_N /
 * $DECODE_NODE_N, the same names the per-node commands consume — so one
 * regex covers every placeholder in the rendered output.
 */
export function detectPlaceholders(text) {
  if (!text) return [];
  const found = new Map();
  const varRe = /\$([A-Z_][A-Z0-9_]*)/g;
  let m;
  while ((m = varRe.exec(text)) !== null) {
    const name = m[1];
    if (!found.has(name)) {
      found.set(name, { kind: "var", name, label: `$${name}` });
    }
  }
  return [...found.values()];
}

/**
 * Detect placeholders across many strings (commands, env values, curl, bench).
 * Order is preserved by first appearance so the panel renders in the order
 * the user encounters them.
 */
export function detectPlaceholdersAll(...texts) {
  const seen = new Map();
  for (const t of texts) {
    for (const p of detectPlaceholders(t)) {
      if (!seen.has(p.name)) seen.set(p.name, p);
    }
  }
  return [...seen.values()];
}

/**
 * Replace $VAR / NODE_N placeholders in `text` with values from `endpoints`
 * (a flat `{ name: value }` map keyed by the placeholder name without `$`).
 *
 * Empty/missing values are left untouched — the rendered output keeps the
 * placeholder so users still get a valid shell-variable form.
 */
export function substitute(text, endpoints) {
  if (!text || !endpoints) return text;
  let out = text;
  // $VAR — only replace whole identifiers, longest-key first to avoid
  // $PREFILL_HOST_IP being clipped by $PREFILL.
  const keys = Object.keys(endpoints).sort((a, b) => b.length - a.length);
  for (const k of keys) {
    const v = endpoints[k];
    if (v === undefined || v === null || v === "") continue;
    // Word-boundary aware: $FOO must not chew into $FOOBAR.
    const re = new RegExp(`\\$${escapeRe(k)}(?![A-Z0-9_])`, "g");
    out = out.replace(re, v);
  }
  return out;
}

/**
 * Same as substitute() but for an env map ({ KEY: value }). Values are
 * passed through substitute(), keys are left alone.
 */
export function substituteEnv(env, endpoints) {
  if (!env) return env;
  const out = {};
  for (const [k, v] of Object.entries(env)) {
    out[k] = typeof v === "string" ? substitute(v, endpoints) : v;
  }
  return out;
}

function escapeRe(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

export function loadEndpoints() {
  if (typeof window === "undefined") return {};
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : {};
  } catch {
    return {};
  }
}

export function saveEndpoint(name, value) {
  if (typeof window === "undefined") return;
  try {
    const cur = loadEndpoints();
    if (value === undefined || value === null || value === "") {
      delete cur[name];
    } else {
      cur[name] = value;
    }
    localStorage.setItem(STORAGE_KEY, JSON.stringify(cur));
  } catch {}
}

export function clearEndpoints() {
  if (typeof window === "undefined") return;
  try {
    localStorage.removeItem(STORAGE_KEY);
  } catch {}
}
