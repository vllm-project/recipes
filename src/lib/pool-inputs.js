// The same typed fields drive the browser form and all template/filled exports.
// Bind shell words at the boundary. JSON values and raw argv never use shell quoting.
export const inputTypes = ["host", "port", "path", "device", "interface", "integer", "text", "size", "boolean", "id"];
export const identifier = /^[a-zA-Z][a-zA-Z0-9_-]*$/;
export const fieldReference = /\{\{([\w.-]+)\}\}/g;
const wholeField = /^\{\{([\w.-]+)\}\}$/;

export function quoteWord(value) {
  const word = String(value);
  return /^[a-zA-Z0-9_@%+=:,./-]+$/.test(word) ? word : "'" + word.replaceAll("'", "'\"'\"'") + "'";
}

function isHost(value) {
  // The Ascend recipes use the same address for bind(), URLs and host:port
  // endpoints. IPv6 needs distinct formatting for those three contexts.
  if (/^[\d.]+$/.test(value)) return /^(?:\d{1,3}\.){3}\d{1,3}$/.test(value)
    && value.split(".").every((part) => String(Number(part)) === part && Number(part) <= 255);
  return value.length <= 253 && value.split(".").every((part) =>
    /^[a-z\d](?:[a-z\d-]{0,61}[a-z\d])?$/i.test(part));
}

export function inputError(type, value) {
  const word = String(value);
  if (Array.isArray(value) || (value !== null && typeof value === "object")
      || /[\x00-\x1f\x7f]/.test(word)) return "Expected a single-line value.";
  switch (type) {
    case "host": return isHost(word) ? null : "Enter an IPv4 address or DNS name.";
    case "port": return /^\d+$/.test(word) && Number(word) >= 1 && Number(word) <= 65535
      ? null : "Enter a port from 1 to 65535.";
    case "integer": return /^\d+$/.test(word) && Number.isSafeInteger(Number(word)) ? null : "Enter a non-negative integer.";
    case "path":
    case "device": {
      if (!word.startsWith("/") || word.includes(":") || word.split("/").includes(".."))
        return "Enter an absolute Linux path without a colon or a parent traversal.";
      return type === "device" && !/^\/dev\/.+/.test(word) ? "Enter an absolute block device path under /dev/." : null;
    }
    case "interface": return /^[\w][\w.:-]{0,14}$/.test(word) ? null : "Enter a network interface name.";
    case "size": return /^\d+(?:\.\d+)?(?:[kmgt]i?b?|b)?$/i.test(word) ? null : "Enter a capacity with its unit.";
    case "boolean": return /^(true|false)$/.test(word) ? null : "Choose true or false.";
    case "id": return /^[a-zA-Z0-9_][\w.-]*$/.test(word) ? null : "Use letters, numbers, dots, hyphens or underscores.";
    case "text": return null;
    default: return "Unknown input type: " + type;
  }
}

export function poolInputs(profile, deployment, mode, supplied) {
  const specs = { ...profile.inputs, ...deployment.inputs };
  return Object.entries(specs).flatMap(([name, spec]) => {
    if (mode && spec.mode && mode !== spec.mode) return [];
    return (spec.scope === "host" ? deployment.hosts : [null]).map((host) => {
      const key = host ? "hosts." + host + "." + name : name;
      const defaults = deployment.defaults || {};
      const raw = Object.hasOwn(supplied, key) ? supplied[key] : defaults[key] ?? defaults[name] ?? spec.default;
      const missing = raw === undefined || raw === null || raw === "";
      const error = missing ? null : inputError(spec.type, raw);
      let value = missing || error ? "" : raw;
      if (!missing && !error) {
        if (spec.type === "integer" || spec.type === "port") value = Number(value);
        if (spec.type === "boolean") value = String(value) === "true";
      }
      return { key, label: spec.label, type: spec.type, host, mode: spec.mode || null, value, missing,
        ...(error && { error, value: typeof raw === "string" ? raw : "" }) };
    });
  });
}

export function makeBindings(fields, context, extras = {}) {
  const missing = new Set(), errors = new Set();
  const read = (name) => {
    if (Object.hasOwn(extras, name)) return extras[name];
    const { host, container, placement } = context;
    if (name === "host_id" && host) return host;
    if (name === "container_id" && container) return container;
    if (name.startsWith("placement.") && placement) {
      const key = name.slice("placement.".length);
      const vars = { ...placement.vars, id: placement.id ?? placement.role, rank: placement.rank ?? 0 };
      if (Object.hasOwn(vars, key)) return vars[key];
    }
    const scoped = "hosts." + host + "." + name;
    const key = fields.has(scoped) ? scoped : name;
    const field = fields.get(key);
    if (!field) errors.add("Unknown input " + name + " in " + (host || "cluster"));
    else if (field.error) errors.add(field.key + ": " + field.error);
    else if (!field.missing) return field.value;
    missing.add(key);
    return "{{" + key + "}}";
  };
  function bind(value, shell = false) {
    if (typeof value === "string") {
      const whole = wholeField.exec(value);
      if (whole && !shell) return read(whole[1]);
      return value.replace(fieldReference, (_, name) => shell ? quoteWord(read(name)) : String(read(name)));
    }
    if (Array.isArray(value)) return value.map((item) => bind(item));
    if (value !== null && typeof value === "object")
      return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, bind(item)]));
    return value;
  }
  return { bind, missing, errors };
}
